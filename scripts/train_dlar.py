"""
LEPO Training Script with Multiple Reward Functions

Supported reward types:
    - standard: Standard correctness reward only
    - relative: Relative length penalty/reward (LEPO paper)
    - contrastive: Contrastive length optimization (Idea 3) - DEPRECATED
    - hybrid: Hybrid of relative and contrastive - DEPRECATED
    - quality_filtered: Quality-filtered contrastive learning (Idea 2) - NEW
    - confidence_aware: Confidence-aware length reward (Idea 5) - NEW
    - quality_confidence: Quality + Confidence hybrid (Recommended) - NEW

Usage:
    # Standard training
    accelerate launch --config_file configs/accelerate/ds_zero2.yaml \
        scripts/train_dlar.py --config configs/lepo/llama3_1b.yaml --reward_type standard

    # Quality-filtered contrastive learning
    accelerate launch --config_file configs/accelerate/ds_zero2.yaml \
        scripts/train_dlar.py --config configs/lepo/llama3_1b_quality.yaml --reward_type quality_filtered

    # Confidence-aware length reward
    accelerate launch --config_file configs/accelerate/ds_zero2.yaml \
        scripts/train_dlar.py --config configs/lepo/llama3_1b_confidence.yaml --reward_type confidence_aware

    # Quality + Confidence hybrid (recommended)
    accelerate launch --config_file configs/accelerate/ds_zero2.yaml \
        scripts/train_dlar.py --config configs/lepo/llama3_1b_quality_confidence.yaml --reward_type quality_confidence
"""

import os
import sys
import argparse
import datetime
import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from src.models import LatentQwen2ForCausalLM, LatentLlamaForCausalLM
from src.trainers import DLaRTrainer, DLaRConfig
from src.data import get_dataset
from src.rewards import correctness_reward_func
from src.rewards.math_reward import (
    # 原有奖励器
    RelativeLengthRewarder,
    ContrastiveLengthRewarder,
    HybridContrastiveLengthRewarder,
    # 新奖励器 (思路2和思路5)
    QualityFilteredContrastiveRewarder,
    ConfidenceAwareLengthRewarder,
    QualityConfidenceHybridRewarder,
    count_reasoning_tokens
)
from src.utils import set_deterministic_seed


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_special_token_ids(tokenizer):
    """获取特殊token IDs用于长度计算"""
    latent_tokens = ['<|START|>', '<|CONTINUE|>', '<|STOP|>']
    special_token_ids = [
        tokenizer.convert_tokens_to_ids(tok)
        for tok in latent_tokens
        if tokenizer.convert_tokens_to_ids(tok) is not None
    ]
    return special_token_ids


def create_length_judge(tokenizer, config: dict):
    """创建长度判断函数"""
    special_token_ids = get_special_token_ids(tokenizer)

    def length_judge(completion_ids):
        """Judge the reasoning length of a completion."""
        if special_token_ids:
            return float(count_reasoning_tokens(completion_ids, special_token_ids))
        else:
            return float(len(completion_ids))

    return length_judge


def create_relative_length_rewarder(tokenizer, config: dict):
    """Create the RelativeLengthRewarder from math_reward.py"""
    reward_config = config.get('reward', {})
    train_cfg = config['training']
    length_judge = create_length_judge(tokenizer, config)

    return RelativeLengthRewarder(
        num_generations=train_cfg['num_generations'],
        length_judge=length_judge,
        answer_reward=reward_config.get('answer_reward', 1.0),
        format_penalty=reward_config.get('format_penalty', -1.0),
        relative_length_penalty=reward_config.get('relative_length_penalty', 1.0),
        relative_length_reward=reward_config.get('relative_length_reward', 'same'),
        accuracy_requirement=reward_config.get('accuracy_requirement', 0.5),
        all_gather=reward_config.get('all_gather', False),
    )


def create_contrastive_rewarder(tokenizer, config: dict):
    """Create the ContrastiveLengthRewarder (Idea 3) - DEPRECATED"""
    reward_config = config.get('reward', {})
    train_cfg = config['training']
    length_judge = create_length_judge(tokenizer, config)

    return ContrastiveLengthRewarder(
        num_generations=train_cfg['num_generations'],
        length_judge=length_judge,
        answer_reward=reward_config.get('answer_reward', 1.0),
        format_penalty=reward_config.get('format_penalty', -1.0),
        contrastive_coefficient=reward_config.get('contrastive_coefficient', 0.5),
        top_k_short=reward_config.get('top_k_short', 2),
        length_ratio_threshold=reward_config.get('length_ratio_threshold', 1.5),
        temperature=reward_config.get('temperature', 0.1),
    )


def create_hybrid_rewarder(tokenizer, config: dict):
    """Create the HybridContrastiveLengthRewarder - DEPRECATED"""
    reward_config = config.get('reward', {})
    train_cfg = config['training']
    length_judge = create_length_judge(tokenizer, config)

    return HybridContrastiveLengthRewarder(
        num_generations=train_cfg['num_generations'],
        length_judge=length_judge,
        answer_reward=reward_config.get('answer_reward', 1.0),
        format_penalty=reward_config.get('format_penalty', -1.0),
        relative_length_penalty=reward_config.get('relative_length_penalty', 1.0),
        accuracy_requirement=reward_config.get('accuracy_requirement', 0.5),
        contrastive_coefficient=reward_config.get('contrastive_coefficient', 0.5),
        top_k_short=reward_config.get('top_k_short', 2),
        length_ratio_threshold=reward_config.get('length_ratio_threshold', 1.5),
        use_contrastive_for_hard=reward_config.get('use_contrastive_for_hard', True),
        min_length_diff=reward_config.get('min_length_diff', 20.0),
        length_ratio_threshold_abs=reward_config.get('length_ratio_threshold_abs', 1.3),
        min_group_accuracy=reward_config.get('min_group_accuracy', 0.2),
    )


def create_quality_filtered_rewarder(tokenizer, config: dict):
    """
    创建质量筛选对比奖励器 (思路2)

    Args:
        tokenizer: Tokenizer
        config: 配置字典
    """
    reward_config = config.get('reward', {})
    train_cfg = config['training']
    length_judge = create_length_judge(tokenizer, config)

    return QualityFilteredContrastiveRewarder(
        num_generations=train_cfg['num_generations'],
        length_judge=length_judge,
        answer_reward=reward_config.get('answer_reward', 1.0),
        format_penalty=reward_config.get('format_penalty', -0.3),
        quality_threshold=reward_config.get('quality_threshold', 0.6),
        contrastive_coefficient=reward_config.get('contrastive_coefficient', 0.2),
        top_k_short=reward_config.get('top_k_short', 1),
        length_ratio_threshold=reward_config.get('length_ratio_threshold', 1.5),
        min_correct_samples=reward_config.get('min_correct_samples', 2),
        min_high_quality_samples=reward_config.get('min_high_quality_samples', 2),
        min_length_diff=reward_config.get('min_length_diff', 30.0),
        min_steps=reward_config.get('min_steps', 2),
        max_steps=reward_config.get('max_steps', 8),
    )


def create_confidence_aware_rewarder(tokenizer, config: dict):
    """
    创建置信度感知长度奖励器 (思路5)

    Args:
        tokenizer: Tokenizer
        config: 配置字典
    """
    reward_config = config.get('reward', {})
    train_cfg = config['training']
    length_judge = create_length_judge(tokenizer, config)

    return ConfidenceAwareLengthRewarder(
        num_generations=train_cfg['num_generations'],
        length_judge=length_judge,
        answer_reward=reward_config.get('answer_reward', 1.0),
        format_penalty=reward_config.get('format_penalty', -0.3),
        confidence_threshold=reward_config.get('confidence_threshold', 0.7),
        optimal_length_min=reward_config.get('optimal_length_min', 150),
        optimal_length_max=reward_config.get('optimal_length_max', 350),
        short_length_reward=reward_config.get('short_length_reward', 0.1),
        optimal_length_reward=reward_config.get('optimal_length_reward', 0.15),
        long_length_reward=reward_config.get('long_length_reward', 0.05),
        use_entropy=reward_config.get('use_entropy', True),
        entropy_threshold=reward_config.get('entropy_threshold', 2.0),
    )


def create_quality_confidence_rewarder(tokenizer, config: dict):
    """
    创建质量筛选 + 置信度感知 混合奖励器 (推荐)

    Args:
        tokenizer: Tokenizer
        config: 配置字典
    """
    reward_config = config.get('reward', {})
    train_cfg = config['training']
    length_judge = create_length_judge(tokenizer, config)

    return QualityConfidenceHybridRewarder(
        num_generations=train_cfg['num_generations'],
        length_judge=length_judge,
        answer_reward=reward_config.get('answer_reward', 1.0),
        format_penalty=reward_config.get('format_penalty', -0.3),
        # 质量筛选参数
        quality_threshold=reward_config.get('quality_threshold', 0.6),
        min_steps=reward_config.get('min_steps', 2),
        max_steps=reward_config.get('max_steps', 8),
        # 置信度感知参数
        confidence_threshold=reward_config.get('confidence_threshold', 0.7),
        optimal_length_min=reward_config.get('optimal_length_min', 150),
        optimal_length_max=reward_config.get('optimal_length_max', 350),
        optimal_length_reward=reward_config.get('optimal_length_reward', 0.15),
        short_length_reward=reward_config.get('short_length_reward', 0.1),
        long_length_reward=reward_config.get('long_length_reward', 0.05),
    )


def create_rewarder(tokenizer, config: dict, reward_type: str = "standard"):
    """
    Create reward function based on type

    Args:
        tokenizer: The tokenizer for decoding
        config: Training configuration dictionary
        reward_type: Type of reward function
            - "standard": Standard correctness reward
            - "relative": Relative length penalty/reward (LEPO paper)
            - "contrastive": Contrastive length optimization (DEPRECATED)
            - "hybrid": Hybrid of relative and contrastive (DEPRECATED)
            - "quality_filtered": Quality-filtered contrastive learning (Idea 2)
            - "confidence_aware": Confidence-aware length reward (Idea 5)
            - "quality_confidence": Quality + Confidence hybrid (Recommended)

    Returns:
        Reward function instance
    """
    if reward_type == "standard":
        return correctness_reward_func
    elif reward_type == "relative":
        return create_relative_length_rewarder(tokenizer, config)
    elif reward_type == "contrastive":
        print("Warning: 'contrastive' is deprecated. Consider using 'quality_filtered' instead.")
        return create_contrastive_rewarder(tokenizer, config)
    elif reward_type == "hybrid":
        print("Warning: 'hybrid' is deprecated. Consider using 'quality_confidence' instead.")
        return create_hybrid_rewarder(tokenizer, config)
    elif reward_type == "quality_filtered":
        return create_quality_filtered_rewarder(tokenizer, config)
    elif reward_type == "confidence_aware":
        return create_confidence_aware_rewarder(tokenizer, config)
    elif reward_type == "quality_confidence":
        return create_quality_confidence_rewarder(tokenizer, config)
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}. "
                         f"Available: standard, relative, contrastive, hybrid, "
                         f"quality_filtered, confidence_aware, quality_confidence")


def print_reward_config(reward_type: str, train_cfg: dict, reward_config: dict):
    """打印奖励配置信息"""
    print(f"Using {reward_type} reward function...")

    if reward_type == "standard":
        print("  - Using standard correctness reward")

    elif reward_type == "relative":
        print(f"  - num_generations: {train_cfg['num_generations']}")
        print(f"  - answer_reward: {reward_config.get('answer_reward', 1.0)}")
        print(f"  - format_penalty: {reward_config.get('format_penalty', -1.0)}")
        print(f"  - relative_length_penalty: {reward_config.get('relative_length_penalty', 1.0)}")
        print(f"  - relative_length_reward: {reward_config.get('relative_length_reward', 'same')}")
        print(f"  - accuracy_requirement: {reward_config.get('accuracy_requirement', 0.5)}")

    elif reward_type == "contrastive":
        print(f"  - num_generations: {train_cfg['num_generations']}")
        print(f"  - answer_reward: {reward_config.get('answer_reward', 1.0)}")
        print(f"  - format_penalty: {reward_config.get('format_penalty', -1.0)}")
        print(f"  - contrastive_coefficient: {reward_config.get('contrastive_coefficient', 0.5)}")
        print(f"  - top_k_short: {reward_config.get('top_k_short', 2)}")
        print(f"  - length_ratio_threshold: {reward_config.get('length_ratio_threshold', 1.5)}")
        print("  - ⚠️  DEPRECATED: Consider using 'quality_filtered'")

    elif reward_type == "hybrid":
        print(f"  - num_generations: {train_cfg['num_generations']}")
        print(f"  - answer_reward: {reward_config.get('answer_reward', 1.0)}")
        print(f"  - format_penalty: {reward_config.get('format_penalty', -1.0)}")
        print(f"  - relative_length_penalty: {reward_config.get('relative_length_penalty', 1.0)}")
        print(f"  - accuracy_requirement: {reward_config.get('accuracy_requirement', 0.5)}")
        print(f"  - contrastive_coefficient: {reward_config.get('contrastive_coefficient', 0.5)}")
        print(f"  - top_k_short: {reward_config.get('top_k_short', 2)}")
        print(f"  - length_ratio_threshold: {reward_config.get('length_ratio_threshold', 1.5)}")
        print("  - ⚠️  DEPRECATED: Consider using 'quality_confidence'")

    elif reward_type == "quality_filtered":
        print(f"  - num_generations: {train_cfg['num_generations']}")
        print(f"  - answer_reward: {reward_config.get('answer_reward', 1.0)}")
        print(f"  - format_penalty: {reward_config.get('format_penalty', -0.3)}")
        print(f"  - quality_threshold: {reward_config.get('quality_threshold', 0.6)}")
        print(f"  - contrastive_coefficient: {reward_config.get('contrastive_coefficient', 0.2)}")
        print(f"  - top_k_short: {reward_config.get('top_k_short', 1)}")
        print(f"  - length_ratio_threshold: {reward_config.get('length_ratio_threshold', 1.5)}")
        print(f"  - min_correct_samples: {reward_config.get('min_correct_samples', 2)}")
        print(f"  - min_high_quality_samples: {reward_config.get('min_high_quality_samples', 2)}")
        print(f"  - min_length_diff: {reward_config.get('min_length_diff', 30.0)}")
        print(f"  - min_steps: {reward_config.get('min_steps', 2)}")
        print(f"  - max_steps: {reward_config.get('max_steps', 8)}")

    elif reward_type == "confidence_aware":
        print(f"  - num_generations: {train_cfg['num_generations']}")
        print(f"  - answer_reward: {reward_config.get('answer_reward', 1.0)}")
        print(f"  - format_penalty: {reward_config.get('format_penalty', -0.3)}")
        print(f"  - confidence_threshold: {reward_config.get('confidence_threshold', 0.7)}")
        print(f"  - optimal_length_min: {reward_config.get('optimal_length_min', 150)}")
        print(f"  - optimal_length_max: {reward_config.get('optimal_length_max', 350)}")
        print(f"  - optimal_length_reward: {reward_config.get('optimal_length_reward', 0.15)}")
        print(f"  - short_length_reward: {reward_config.get('short_length_reward', 0.1)}")
        print(f"  - long_length_reward: {reward_config.get('long_length_reward', 0.05)}")
        print(f"  - use_entropy: {reward_config.get('use_entropy', True)}")
        print(f"  - entropy_threshold: {reward_config.get('entropy_threshold', 2.0)}")

    elif reward_type == "quality_confidence":
        print(f"  - num_generations: {train_cfg['num_generations']}")
        print(f"  - answer_reward: {reward_config.get('answer_reward', 1.0)}")
        print(f"  - format_penalty: {reward_config.get('format_penalty', -0.3)}")
        print(f"  - quality_threshold: {reward_config.get('quality_threshold', 0.6)}")
        print(f"  - min_steps: {reward_config.get('min_steps', 2)}")
        print(f"  - max_steps: {reward_config.get('max_steps', 8)}")
        print(f"  - confidence_threshold: {reward_config.get('confidence_threshold', 0.7)}")
        print(f"  - optimal_length_min: {reward_config.get('optimal_length_min', 150)}")
        print(f"  - optimal_length_max: {reward_config.get('optimal_length_max', 350)}")
        print(f"  - optimal_length_reward: {reward_config.get('optimal_length_reward', 0.15)}")
        print(f"  - short_length_reward: {reward_config.get('short_length_reward', 0.1)}")
        print(f"  - long_length_reward: {reward_config.get('long_length_reward', 0.05)}")
        print("  - ⭐ RECOMMENDED for small models")


def main():
    parser = argparse.ArgumentParser(description='LEPO Training with Multiple Reward Functions')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--reward_type', type=str, default='standard',
                        choices=[
                            'standard', 'relative', 'contrastive', 'hybrid',
                            'quality_filtered', 'confidence_aware', 'quality_confidence'
                        ],
                        help='Type of reward function to use')
    # 保留兼容性参数
    parser.add_argument('--use_relative_length_reward', action='store_true',
                        help='[Deprecated] Use --reward_type relative instead')
    parser.add_argument('--no_relative_length_reward', action='store_true',
                        help='[Deprecated] Use --reward_type standard instead')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # 兼容旧参数
    if args.use_relative_length_reward and args.reward_type == 'standard':
        args.reward_type = 'relative'
    if args.no_relative_length_reward:
        args.reward_type = 'standard'

    # Determine reward type from config if not specified
    if args.reward_type == 'standard':
        config_reward_type = config.get('reward_type', 'standard')
        if config_reward_type != 'standard':
            args.reward_type = config_reward_type

    # Set random seed
    set_deterministic_seed(config['seed'])

    # Set wandb project
    os.environ["WANDB_PROJECT"] = config.get('wandb_project', 'LEPO-clean')

    # Load dataset
    print("Loading dataset...")
    train_data_cfg = config['data']['train']
    eval_data_cfg = config['data']['eval']

    train_dataset = get_dataset(
        name=train_data_cfg['name'],
        split=train_data_cfg['split'],
        subset=train_data_cfg['subset'],
        shot=train_data_cfg['shot']
    )
    eval_dataset = get_dataset(
        name=eval_data_cfg['name'],
        split=eval_data_cfg['split'],
        subset=eval_data_cfg['subset'],
        shot=eval_data_cfg['shot']
    )
    print("Dataset loaded.")

    # Model configuration
    model_path = config['model']['path']
    model_name = model_path.split("/")[-1]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Tokenizer loaded.")

    # Generation configuration
    gen_cfg = config['generation']
    latent_length = gen_cfg.get('latent_length', 32)
    noise_strength = gen_cfg.get('noise_strength', 0.30)
    noise_type = gen_cfg.get('noise_type', 'gumbel')
    noise_str = str(int(noise_strength * 100)).zfill(3)

    # 构建 generation_kwargs，包含置信度引导相关参数
    generation_kwargs = {
        "max_new_tokens": gen_cfg['max_new_tokens'],
        "max_completion_length": gen_cfg['max_new_tokens'],
        "temperature": gen_cfg['temperature'],
        "top_p": gen_cfg['top_p'],
        "top_k": gen_cfg['top_k'],
        "latent_length": latent_length,
        "do_latent_sample": gen_cfg.get('do_latent_sample', False),
        "do_discrete_sample": gen_cfg.get('do_discrete_sample', True),
        "noise_type": noise_type,
        "noise_strength": noise_strength,
        "pad_token_id": tokenizer.eos_token_id,
        # 置信度引导相关
        "use_confidence_guided_noise": gen_cfg.get('use_confidence_guided_noise', False),
        "confidence_weight": gen_cfg.get('confidence_weight', 0.7),
        "entropy_weight": gen_cfg.get('entropy_weight', 0.3),
        "min_noise_strength": gen_cfg.get('min_noise_strength', 0.01),
        "max_noise_strength": gen_cfg.get('max_noise_strength', 0.3),
        "noise_schedule": gen_cfg.get('noise_schedule', None),
        "noise_warmup_steps": gen_cfg.get('noise_warmup_steps', 1000),
        "noise_start_strength": gen_cfg.get('noise_start_strength', None),
        "noise_end_strength": gen_cfg.get('noise_end_strength', None),
    }

    # Training configuration
    train_cfg = config['training']
    eval_cfg = config['eval']
    output_cfg = config['output']

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    data_name = train_data_cfg['name']
    sub_name = train_data_cfg['subset']
    split = train_data_cfg['split']

    # Add suffix to output dir based on reward type
    reward_suffix = f"-{args.reward_type.capitalize()}" if args.reward_type != 'standard' else ""
    model_type = config['model'].get('model_type', 'qwen')
    model_prefix = "qwen" if model_type == "qwen" else "llama"
    output_dir = f"{output_cfg['base_dir']}/{model_prefix}-CLaR-0shot-{latent_length}-{noise_type}{noise_str}-{data_name}-{sub_name}-{split}{reward_suffix}-{timestamp}"
    run_name = f"{model_prefix}-CLaR-0shot-{latent_length}-{noise_type}{noise_str}-{data_name}-{sub_name}-{split}{reward_suffix}-{timestamp}"

    training_args = DLaRConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=float(train_cfg['learning_rate']),
        adam_beta1=float(train_cfg['adam_beta1']),
        adam_beta2=float(train_cfg['adam_beta2']),
        weight_decay=float(train_cfg['weight_decay']),
        warmup_ratio=float(train_cfg['warmup_ratio']),
        lr_scheduler_type=train_cfg['lr_scheduler_type'],
        logging_steps=train_cfg['logging_steps'],
        bf16=train_cfg['bf16'],
        per_device_train_batch_size=train_cfg['per_device_train_batch_size'],
        gradient_accumulation_steps=train_cfg['gradient_accumulation_steps'],
        steps_per_generation=train_cfg['steps_per_generation'],
        num_iterations=train_cfg['num_iterations'],
        num_generations=train_cfg['num_generations'],
        max_prompt_length=train_cfg['max_prompt_length'],
        num_train_epochs=train_cfg['num_train_epochs'],
        save_steps=train_cfg['save_steps'],
        max_grad_norm=train_cfg['max_grad_norm'],
        generation_kwargs=generation_kwargs,
        report_to=train_cfg['report_to'],
        do_eval=eval_cfg['do_eval'],
        eval_strategy=eval_cfg['eval_strategy'],
        eval_steps=eval_cfg['eval_steps'],
        per_device_eval_batch_size=eval_cfg['per_device_eval_batch_size'],
        eval_accumulation_steps=eval_cfg['eval_accumulation_steps'],
        eval_on_start=eval_cfg['eval_on_start'],
        max_steps=train_cfg.get('max_steps', -1),
        logging_dir=train_cfg.get('logging_dir', None),
    )

    # Load model
    print("Loading model...")
    torch_dtype = getattr(torch, config['model']['torch_dtype'])
    model_type = config['model'].get('model_type', 'qwen')

    if model_type == 'llama':
        model = LatentLlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=None
        )
    else:
        model = LatentQwen2ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=None
        )

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
    print("Model loaded.")

    # Create reward function
    reward_config = config.get('reward', {})
    print_reward_config(args.reward_type, train_cfg, reward_config)
    reward_func = create_rewarder(tokenizer, config, args.reward_type)

    # Training
    trainer = DLaRTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Training...")
    trainer.train()
    print("Training complete.")


if __name__ == "__main__":
    main()