"""
GRPO Training Script with Multiple Reward Functions

Supported reward types:
    - standard: Standard correctness reward only
    - relative: Relative length penalty/reward (LEPO paper)
    - contrastive: Contrastive length optimization (Idea 3)
    - hybrid: Hybrid of relative and contrastive (recommended)

Usage:
    accelerate launch --config_file configs/accelerate/ds_zero2.yaml \
        scripts/train_grpo.py --config configs/grpo/qwen2_3b_grpo.yaml --reward_type standard

    # With relative length reward
    accelerate launch --config_file configs/accelerate/ds_zero2.yaml \
        scripts/train_grpo.py --config configs/grpo/qwen2_3b_grpo.yaml --reward_type relative

    # With contrastive length optimization (Idea 3)
    accelerate launch --config_file configs/accelerate/ds_zero2.yaml \
        scripts/train_grpo.py --config configs/grpo/qwen2_3b_grpo_contrastive.yaml --reward_type contrastive
"""

import os
import sys
import argparse
import datetime
import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.data import get_dataset
from src.rewards import correctness_reward_func
from src.rewards.math_reward import (
    RelativeLengthRewarder,
    ContrastiveLengthRewarder,
    HybridContrastiveLengthRewarder,
    count_reasoning_tokens
)
from src.utils import set_deterministic_seed


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_relative_length_rewarder(tokenizer, config: dict):
    """
    Create the RelativeLengthRewarder from math_reward.py

    Args:
        tokenizer: The tokenizer for decoding
        config: Training configuration dictionary

    Returns:
        RelativeLengthRewarder instance
    """
    reward_config = config.get('reward', {})
    train_cfg = config['training']

    # Get special token IDs for latent tokens (if using latent model)
    special_token_ids = None
    if 'latent' in config.get('model', {}):
        latent_tokens = ['<|START|>', '<|CONTINUE|>', '<|STOP|>']
        special_token_ids = [
            tokenizer.convert_tokens_to_ids(tok)
            for tok in latent_tokens
            if tokenizer.convert_tokens_to_ids(tok) is not None
        ]

    def length_judge(completion_ids):
        """Judge the reasoning length of a completion."""
        if special_token_ids:
            return float(count_reasoning_tokens(completion_ids, special_token_ids))
        else:
            return float(len(completion_ids))

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
    """
    Create the ContrastiveLengthRewarder (Idea 3)

    Args:
        tokenizer: The tokenizer for decoding
        config: Training configuration dictionary

    Returns:
        ContrastiveLengthRewarder instance
    """
    reward_config = config.get('reward', {})
    train_cfg = config['training']

    # Get special token IDs for latent tokens
    special_token_ids = None
    if 'latent' in config.get('model', {}):
        latent_tokens = ['<|START|>', '<|CONTINUE|>', '<|STOP|>']
        special_token_ids = [
            tokenizer.convert_tokens_to_ids(tok)
            for tok in latent_tokens
            if tokenizer.convert_tokens_to_ids(tok) is not None
        ]

    def length_judge(completion_ids):
        """Judge the reasoning length of a completion."""
        if special_token_ids:
            return float(count_reasoning_tokens(completion_ids, special_token_ids))
        else:
            return float(len(completion_ids))

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
    """
    Create the HybridContrastiveLengthRewarder

    Args:
        tokenizer: The tokenizer for decoding
        config: Training configuration dictionary

    Returns:
        HybridContrastiveLengthRewarder instance
    """
    reward_config = config.get('reward', {})
    train_cfg = config['training']

    # Get special token IDs for latent tokens
    special_token_ids = None
    if 'latent' in config.get('model', {}):
        latent_tokens = ['<|START|>', '<|CONTINUE|>', '<|STOP|>']
        special_token_ids = [
            tokenizer.convert_tokens_to_ids(tok)
            for tok in latent_tokens
            if tokenizer.convert_tokens_to_ids(tok) is not None
        ]

    def length_judge(completion_ids):
        """Judge the reasoning length of a completion."""
        if special_token_ids:
            return float(count_reasoning_tokens(completion_ids, special_token_ids))
        else:
            return float(len(completion_ids))

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
            - "contrastive": Contrastive length optimization (Idea 3)
            - "hybrid": Hybrid of relative and contrastive

    Returns:
        Reward function instance
    """
    if reward_type == "standard":
        return correctness_reward_func
    elif reward_type == "relative":
        return create_relative_length_rewarder(tokenizer, config)
    elif reward_type == "contrastive":
        return create_contrastive_rewarder(tokenizer, config)
    elif reward_type == "hybrid":
        return create_hybrid_rewarder(tokenizer, config)
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}. "
                         f"Available: standard, relative, contrastive, hybrid")


def main():
    parser = argparse.ArgumentParser(description='GRPO Training with Multiple Reward Functions')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--reward_type', type=str, default='standard',
                        choices=['standard', 'relative', 'contrastive', 'hybrid'],
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
    generation_kwargs = {
        "max_new_tokens": gen_cfg['max_new_tokens'],
        "max_completion_length": gen_cfg['max_new_tokens'],
        "temperature": gen_cfg['temperature'],
        "top_p": gen_cfg['top_p'],
        "top_k": gen_cfg['top_k'],
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
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
    output_dir = f"{output_cfg['base_dir']}/{model_name}-GRPO-0shot-{data_name}-{sub_name}-{split}{reward_suffix}-{timestamp}"
    run_name = f"{model_name}-GRPO-0shot-{data_name}-{sub_name}-{split}{reward_suffix}-{timestamp}"

    training_args = GRPOConfig(
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
        max_steps=train_cfg.get('max_steps', -1),          # ← 新增：透传 max_steps
        logging_dir=train_cfg.get('logging_dir', None),    # ← 新增：透传 logging_dir
        max_grad_norm=train_cfg['max_grad_norm'],
        generation_kwargs=generation_kwargs,
        report_to=train_cfg['report_to'],
        do_eval=eval_cfg['do_eval'],
        eval_strategy=eval_cfg['eval_strategy'],
        eval_steps=eval_cfg['eval_steps'],
        per_device_eval_batch_size=eval_cfg['per_device_eval_batch_size'],
        eval_accumulation_steps=eval_cfg['eval_accumulation_steps'],
        eval_on_start=eval_cfg['eval_on_start'],
    )

    # Load model
    print("Loading model...")
    torch_dtype = getattr(torch, config['model']['torch_dtype'])
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=None
    )

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
    print("Model loaded.")

    # Create reward function
    print(f"Using {args.reward_type} reward function...")
    reward_func = create_rewarder(tokenizer, config, args.reward_type)

    # Print configuration
    if args.reward_type == 'relative':
        reward_config = config.get('reward', {})
        print(f"  - num_generations: {train_cfg['num_generations']}")
        print(f"  - answer_reward: {reward_config.get('answer_reward', 1.0)}")
        print(f"  - format_penalty: {reward_config.get('format_penalty', -1.0)}")
        print(f"  - relative_length_penalty: {reward_config.get('relative_length_penalty', 1.0)}")
        print(f"  - relative_length_reward: {reward_config.get('relative_length_reward', 'same')}")
        print(f"  - accuracy_requirement: {reward_config.get('accuracy_requirement', 0.5)}")
    elif args.reward_type == 'contrastive':
        reward_config = config.get('reward', {})
        print(f"  - num_generations: {train_cfg['num_generations']}")
        print(f"  - answer_reward: {reward_config.get('answer_reward', 1.0)}")
        print(f"  - format_penalty: {reward_config.get('format_penalty', -1.0)}")
        print(f"  - contrastive_coefficient: {reward_config.get('contrastive_coefficient', 0.5)}")
        print(f"  - top_k_short: {reward_config.get('top_k_short', 2)}")
        print(f"  - length_ratio_threshold: {reward_config.get('length_ratio_threshold', 1.5)}")
    elif args.reward_type == 'hybrid':
        reward_config = config.get('reward', {})
        print(f"  - num_generations: {train_cfg['num_generations']}")
        print(f"  - answer_reward: {reward_config.get('answer_reward', 1.0)}")
        print(f"  - format_penalty: {reward_config.get('format_penalty', -1.0)}")
        print(f"  - relative_length_penalty: {reward_config.get('relative_length_penalty', 1.0)}")
        print(f"  - accuracy_requirement: {reward_config.get('accuracy_requirement', 0.5)}")
        print(f"  - contrastive_coefficient: {reward_config.get('contrastive_coefficient', 0.5)}")
        print(f"  - top_k_short: {reward_config.get('top_k_short', 2)}")
        print(f"  - length_ratio_threshold: {reward_config.get('length_ratio_threshold', 1.5)}")
    else:
        print("  - Using standard correctness reward")

    # Training
    trainer = GRPOTrainer(
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