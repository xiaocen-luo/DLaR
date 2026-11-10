"""
Distributed Latent Model Evaluation Script

Usage:
    torchrun --nproc_per_node=8 scripts/eval_latent.py --config configs/eval/xxx.yaml

Or use the launch script:
    ./scripts/run_eval.sh configs/eval/xxx.yaml
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
import os
import json
import yaml
from tqdm import tqdm
import datetime
import argparse
import time
import numpy as np

from src.models import LatentQwen2ForCausalLM, LatentLlamaForCausalLM, LatentGenerationConfig
from src.data.datasets import get_dataset
from src.rewards import eval_correctness_check
from src.utils.functions import set_deterministic_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LLM on math datasets')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    parser.add_argument('--load_balance_method', type=str, default='interleaved',
                        choices=['interleaved', 'sequential'],
                        help='Load balancing method for distributed evaluation')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples per question')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0

    torch.cuda.set_device(gpu)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        timeout=datetime.timedelta(minutes=300)
    )
    dist.barrier()

    return True, rank, world_size, gpu


def get_gpu_memory_mb(gpu_id):
    """Get current GPU memory usage (MB)"""
    allocated = torch.cuda.memory_allocated(gpu_id) / (1024 * 1024)
    reserved = torch.cuda.memory_reserved(gpu_id) / (1024 * 1024)
    return allocated, reserved


def get_peak_gpu_memory_mb(gpu_id):
    """Get peak GPU memory usage (MB)"""
    peak_reserved = torch.cuda.max_memory_reserved(gpu_id) / (1024 * 1024)
    return peak_reserved


def get_interleaved_indices(total_size, world_size, rank):
    """Distribute indices in interleaved fashion for load balancing"""
    indices = list(range(rank, total_size, world_size))
    return indices


def get_response_transformers(prompts, model, tokenizer, generation_config, gpu, batch_size=32):
    """Generate responses in batches"""
    responses = []
    actual_model = model.module if hasattr(model, 'module') else model

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]

        batch_input_texts = [
            tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            ) for prompt in batch_prompts
        ]

        batch_inputs = tokenizer(
            batch_input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length
        ).to(f'cuda:{gpu}')

        with torch.no_grad():
            outputs = actual_model.generate(
                **batch_inputs,
                generation_config=generation_config,
                pad_token_id=tokenizer.eos_token_id
            )
            batch_outputs = outputs['ids']

        input_length = batch_inputs.input_ids.shape[1]

        for j in range(len(batch_prompts)):
            decoded = tokenizer.decode(
                batch_outputs[j, input_length:],
                skip_special_tokens=True
            )
            responses.append(decoded)

        del batch_inputs, batch_outputs
        torch.cuda.empty_cache()

    return responses


def compute_pass_at_k(correct_counts, num_samples):
    """
    Compute Pass@k metric.
    Pass@k = proportion of problems where at least one sample is correct.
    """
    if num_samples == 1:
        # For k=1, Pass@1 = Mean@1
        return sum(correct_counts) / len(correct_counts)

    pass_count = sum(1 for c in correct_counts if c > 0)
    return pass_count / len(correct_counts)


def compute_mean_at_k(correct_counts, num_samples):
    """
    Compute Mean@k metric.
    Mean@k = average accuracy across all samples.
    """
    return sum(correct_counts) / (len(correct_counts) * num_samples)


def main():
    args = parse_args()

    # Initialize distributed environment
    is_distributed, rank, world_size, gpu = setup_distributed()

    # Record start time
    eval_start_time = time.time()

    def print_rank0(*print_args, **kwargs):
        if rank == 0:
            print(*print_args, **kwargs)

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed
    set_deterministic_seed(args.seed)

    # Get parameters from configuration
    MODEL_PATH = config['model_path']
    LOG_DIR = config.get('log_dir', './eval_logs')
    DATASET_NAME = config['dataset']['name']
    DATASET_SPLIT = config['dataset'].get('split', 'test')
    DATASET_SUBSET = config['dataset'].get('subset', 'none')
    BATCH_SIZE = config.get('batch_size', 64)
    # Priority: command line args > config file > default values
    NUM_SAMPLES = args.num_samples if args.num_samples != 1 else config.get('num_samples', 1)
    SEED = args.seed if args.seed != 0 else config.get('seed', 0)

    # Generation configuration
    g_config = config.get('generation_config', {})
    latent_length = g_config.get('latent_length', 0)
    do_latent_sample = g_config.get('do_latent_sample', False)
    do_discrete_sample = g_config.get('do_discrete_sample', True)
    noise_type = g_config.get('noise_type', 'gumbel')
    noise_strength = g_config.get('noise_strength', 0.0)
    temperature = g_config.get('temperature', 1.0)
    max_new_tokens = g_config.get('max_new_tokens', 2048)

    noise_str = str(int(noise_strength * 100)).zfill(3)
    temp_str = str(int(temperature * 100)).zfill(3)

    # Model name
    MODEL_NAME = MODEL_PATH.split("/")[-1]

    # Ensure log directory exists
    if rank == 0:
        os.makedirs(LOG_DIR, exist_ok=True)

    if is_distributed:
        dist.barrier()

    # Create log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{MODEL_NAME}_{DATASET_NAME}_{DATASET_SPLIT}_{DATASET_SUBSET}_latent{latent_length}_noise_{noise_type}_{noise_str}_temp{temp_str}_{NUM_SAMPLES}samples_seed{SEED}_{timestamp}_rank{rank}.jsonl"
    log_path = os.path.join(LOG_DIR, log_filename)
    summary_path = os.path.join(LOG_DIR,
                                f"summary_{MODEL_NAME}_{DATASET_NAME}_{DATASET_SPLIT}_{DATASET_SUBSET}_latent{latent_length}_noise_{noise_type}_{noise_str}_temp{temp_str}_{NUM_SAMPLES}samples_seed{SEED}_{timestamp}.json")

    # Load tokenizer
    print_rank0(f"Loading model {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = 'left'

    # Load model
    if 'llama' in MODEL_PATH.lower():
        model = LatentLlamaForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map={"": gpu},
        )
        print("llama")
    else:
        model = LatentQwen2ForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map={"": gpu},
        )
        print("Qwen")

    if is_distributed:
        model = DDP(model, device_ids=[gpu])

    model.eval()
    print_rank0("Model loaded.")

    # Configure generation parameters
    generation_config = LatentGenerationConfig(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_k=30,
        top_p=0.95,
        latent_length=latent_length,
        do_latent_sample=do_latent_sample,
        do_discrete_sample=do_discrete_sample,
        noise_type=noise_type,
        noise_strength=noise_strength,
    )

    # Load dataset
    print_rank0("Loading dataset...")
    dataset = get_dataset(DATASET_NAME, DATASET_SPLIT, DATASET_SUBSET, shot=0)

    print_rank0(f"Dataset loaded. Total size: {len(dataset)}")
    print_rank0(f"Load balancing method: {args.load_balance_method}")
    print_rank0(f"Number of samples per question: {NUM_SAMPLES}")

    total_size = len(dataset)
    local_total_correct = 0.0  # For Mean@k
    local_pass_count = 0  # For Pass@k
    local_total_questions = 0
    local_total_response_length = 0.0
    local_correct_counts = []  # Store per-question correct counts for both metrics

    # Distribute data
    if args.load_balance_method == 'interleaved':
        local_indices = get_interleaved_indices(total_size, world_size, rank)
        print(f"Rank {rank}: Processing {len(local_indices)} questions with interleaved indices")
    else:
        per_rank_size = total_size // world_size
        start_idx = rank * per_rank_size
        end_idx = start_idx + per_rank_size if rank < world_size - 1 else total_size
        local_indices = list(range(start_idx, end_idx))
        print(f"Rank {rank}: Processing questions {start_idx} to {end_idx}")

    local_tot = len(local_indices)
    local_total_questions = local_tot

    # Progress bar (main process only)
    if rank == 0:
        pbar = tqdm(total=local_tot, desc=f"GPU 0 Progress")

    # Start evaluation
    with open(log_path, 'w', encoding='utf-8') as log_file:
        for question_idx, global_idx in enumerate(local_indices):
            question_data = dataset[global_idx]
            prompt = question_data["prompt"]
            solution = question_data["solution"]
            question = question_data["question"]

            question_responses = []
            question_lengths = []
            question_correct_count = 0
            sample_details = []

            prompts_batch = [prompt] * NUM_SAMPLES

            if rank == 0:
                pbar.set_description(
                    f"GPU 0 - Processing Question {global_idx} ({question_idx + 1}/{local_tot}) with {NUM_SAMPLES} samples"
                )

            start_time = time.time()

            effective_batch_size = min(NUM_SAMPLES, BATCH_SIZE)
            responses_batch = get_response_transformers(
                prompts_batch, model, tokenizer, generation_config, gpu, batch_size=effective_batch_size
            )

            inference_time = time.time() - start_time
            avg_inference_time_per_sample = inference_time / NUM_SAMPLES

            # Process each response
            for sample_idx, response in enumerate(responses_batch):
                response_tokenized = tokenizer(response, return_tensors="pt")
                response_length = response_tokenized['attention_mask'].sum().item()
                question_lengths.append(response_length)

                is_correct, extracted_response, extracted_solution = eval_correctness_check(
                    [response], [solution], DATASET_NAME
                )

                if is_correct[0]:
                    question_correct_count += 1

                question_responses.append(response)

                sample_details.append({
                    "sample_idx": sample_idx,
                    "response": response,
                    "response_length": response_length,
                    "extracted_response": extracted_response[0],
                    "is_correct": bool(is_correct[0]),
                    "inference_time": avg_inference_time_per_sample
                })

            # Compute metrics for this question
            question_accuracy = question_correct_count / NUM_SAMPLES  # Mean@k per question
            question_pass = 1 if question_correct_count > 0 else 0  # Pass@k per question

            avg_response_length = sum(question_lengths) / NUM_SAMPLES

            # Accumulate for Mean@k
            local_total_correct += question_accuracy

            # Accumulate for Pass@k
            local_pass_count += question_pass

            # Store for later computation
            local_correct_counts.append(question_correct_count)

            local_total_response_length += avg_response_length

            log_entry = {
                "global_index": global_idx,
                "question": question,
                "solution": solution,
                "extracted_solution": extracted_solution[0],
                "num_samples": NUM_SAMPLES,
                "correct_count": question_correct_count,
                "mean_accuracy": question_accuracy,  # Mean@k per question
                "pass": question_pass,  # Pass@k per question
                "avg_response_length": avg_response_length,
                "response_lengths": question_lengths,
                "total_inference_time": inference_time,
                "avg_inference_time_per_sample": avg_inference_time_per_sample,
                "rank": rank,
                "samples": sample_details
            }
            log_file.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

            if rank == 0:
                pbar.update(1)
                pbar.set_postfix({
                    'Mean@k': f"{question_accuracy:.3f}",
                    'Pass@k': f"{question_pass:.3f}",
                    'Avg_Length': f"{avg_response_length:.1f}",
                    'Time': f"{inference_time:.2f}s"
                })

    if rank == 0:
        pbar.close()

    # Compute local metrics
    local_mean_at_k = local_total_correct / local_total_questions if local_total_questions > 0 else 0.0
    local_pass_at_k = local_pass_count / local_total_questions if local_total_questions > 0 else 0.0

    # Collect results from all processes
    if is_distributed:
        # Collect for Mean@k
        local_correct_tensor = torch.tensor([local_total_correct], dtype=torch.float32).cuda(gpu)
        # Collect for Pass@k
        local_pass_tensor = torch.tensor([local_pass_count], dtype=torch.float32).cuda(gpu)
        local_questions_tensor = torch.tensor([local_total_questions], dtype=torch.float32).cuda(gpu)
        local_length_tensor = torch.tensor([local_total_response_length], dtype=torch.float32).cuda(gpu)

        all_corrects = [torch.zeros_like(local_correct_tensor) for _ in range(world_size)]
        all_passes = [torch.zeros_like(local_pass_tensor) for _ in range(world_size)]
        all_questions = [torch.zeros_like(local_questions_tensor) for _ in range(world_size)]
        all_lengths = [torch.zeros_like(local_length_tensor) for _ in range(world_size)]

        dist.all_gather(all_corrects, local_correct_tensor)
        dist.all_gather(all_passes, local_pass_tensor)
        dist.all_gather(all_questions, local_questions_tensor)
        dist.all_gather(all_lengths, local_length_tensor)

        total_correct = sum(correct.item() for correct in all_corrects)
        total_pass = sum(pass_.item() for pass_ in all_passes)
        total_questions = sum(questions.item() for questions in all_questions)
        total_length = sum(length.item() for length in all_lengths)

        if rank == 0:
            print("\n" + "=" * 50)
            print("Processing Summary:")
            for i in range(world_size):
                q_count = all_questions[i].item()
                if q_count > 0:
                    avg_acc = all_corrects[i].item() / q_count
                    pass_acc = all_passes[i].item() / q_count
                    avg_len = all_lengths[i].item() / q_count
                    print(
                        f"GPU {i}: Processed {int(q_count)} questions, Mean@k: {avg_acc:.4f}, Pass@k: {pass_acc:.4f}, Avg Length: {avg_len:.2f}")
            print("=" * 50 + "\n")
    else:
        total_correct = local_total_correct
        total_pass = local_pass_count
        total_questions = local_total_questions
        total_length = local_total_response_length

    overall_mean_at_k = total_correct / total_questions if total_questions > 0 else 0.0
    overall_pass_at_k = total_pass / total_questions if total_questions > 0 else 0.0
    overall_avg_length = total_length / total_questions if total_questions > 0 else 0.0

    # Compute Pass@k using unbiased estimator (optional, for k < num_samples)
    # Unbiased Pass@k estimator: 1 - C(n-c, k) / C(n, k)
    # where n = num_samples, c = correct_count
    def compute_unbiased_pass_at_k(correct_counts, num_samples, k):
        """Compute unbiased Pass@k estimator."""
        from math import comb
        if k > num_samples:
            return 1.0
        pass_count = 0
        for c in correct_counts:
            if c == 0:
                continue
            if num_samples - c < k:
                pass_count += 1
            else:
                prob_fail = comb(num_samples - c, k) / comb(num_samples, k)
                pass_count += 1 - prob_fail
        return pass_count / len(correct_counts)

    # Collect all correct counts from all processes
    if is_distributed and rank == 0:
        # For unbiased Pass@k, we need to gather all correct counts
        # This is done by reading the merged log file
        pass

    # Collect peak GPU memory
    if is_distributed:
        peak_gpu_memory = get_peak_gpu_memory_mb(gpu)
        peak_gpu_memory_tensor = torch.tensor([peak_gpu_memory], dtype=torch.float32).cuda(gpu)
        all_peak_memories = [torch.zeros_like(peak_gpu_memory_tensor) for _ in range(world_size)]
        dist.all_gather(all_peak_memories, peak_gpu_memory_tensor)
        all_peak_memories_list = [t.item() for t in all_peak_memories]
        max_peak_gpu_memory = max(all_peak_memories_list)
    else:
        all_peak_memories_list = [get_peak_gpu_memory_mb(gpu)]
        max_peak_gpu_memory = all_peak_memories_list[0]

    # Main process saves summary results
    if rank == 0:
        print_rank0(f"Overall Mean@k: {overall_mean_at_k:.4f}")
        print_rank0(f"Overall Pass@k: {overall_pass_at_k:.4f}")
        print_rank0(f"Overall Average Response Length: {overall_avg_length:.2f}")

        # Merge log files from all processes
        all_logs = []
        question_level_stats = {}
        all_correct_counts = []

        for r in range(world_size):
            rank_log_file = os.path.join(LOG_DIR,
                                         f"{MODEL_NAME}_{DATASET_NAME}_{DATASET_SPLIT}_{DATASET_SUBSET}_latent{latent_length}_noise_{noise_type}_{noise_str}_temp{temp_str}_{NUM_SAMPLES}samples_seed{SEED}_{timestamp}_rank{r}.jsonl")
            if os.path.exists(rank_log_file):
                with open(rank_log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        log_entry = json.loads(line)
                        all_logs.append(log_entry)

                        global_idx = log_entry['global_index']
                        question_level_stats[global_idx] = {
                            "mean_accuracy": log_entry['mean_accuracy'],
                            "pass": log_entry['pass'],
                            "avg_response_length": log_entry['avg_response_length'],
                            "correct_count": log_entry['correct_count'],
                            "num_samples": log_entry['num_samples']
                        }
                        all_correct_counts.append(log_entry['correct_count'])

        all_logs.sort(key=lambda x: x['global_index'])

        merged_log_path = os.path.join(LOG_DIR,
                                       f"{MODEL_NAME}_{DATASET_NAME}_{DATASET_SPLIT}_{DATASET_SUBSET}_latent{latent_length}_noise_{noise_type}_{noise_str}_temp{temp_str}_{NUM_SAMPLES}samples_seed{SEED}_{timestamp}_merged.jsonl")
        with open(merged_log_path, 'w', encoding='utf-8') as f:
            for log in all_logs:
                f.write(json.dumps(log, ensure_ascii=False) + '\n')

        # Calculate statistical distributions
        mean_accuracies = [stats['mean_accuracy'] for stats in question_level_stats.values()]
        pass_values = [stats['pass'] for stats in question_level_stats.values()]
        lengths = [stats['avg_response_length'] for stats in question_level_stats.values()]

        # Recompute metrics from merged data for verification
        recomputed_mean_at_k = sum(mean_accuracies) / len(mean_accuracies) if mean_accuracies else 0.0
        recomputed_pass_at_k = sum(pass_values) / len(pass_values) if pass_values else 0.0

        # Compute unbiased Pass@8 (if k < num_samples)
        if NUM_SAMPLES > 1:
            try:
                from math import comb
                k = min(NUM_SAMPLES, 8)  # Typically report Pass@1, Pass@8, Pass@32
                unbiased_pass_at_k = 0.0
                for c in all_correct_counts:
                    if c == 0:
                        continue
                    if NUM_SAMPLES - c < k:
                        unbiased_pass_at_k += 1
                    else:
                        prob_fail = comb(NUM_SAMPLES - c, k) / comb(NUM_SAMPLES, k)
                        unbiased_pass_at_k += 1 - prob_fail
                unbiased_pass_at_k = unbiased_pass_at_k / len(all_correct_counts) if all_correct_counts else 0.0
            except:
                unbiased_pass_at_k = recomputed_pass_at_k
        else:
            unbiased_pass_at_k = recomputed_pass_at_k

        accuracy_distribution = {
            "mean": float(np.mean(mean_accuracies)),
            "std": float(np.std(mean_accuracies)),
            "min": float(np.min(mean_accuracies)),
            "max": float(np.max(mean_accuracies)),
            "median": float(np.median(mean_accuracies)),
            "quantiles": {
                "25%": float(np.percentile(mean_accuracies, 25)),
                "50%": float(np.percentile(mean_accuracies, 50)),
                "75%": float(np.percentile(mean_accuracies, 75))
            }
        }

        length_distribution = {
            "mean": float(np.mean(lengths)),
            "std": float(np.std(lengths)),
            "min": float(np.min(lengths)),
            "max": float(np.max(lengths)),
            "median": float(np.median(lengths)),
            "quantiles": {
                "25%": float(np.percentile(lengths, 25)),
                "50%": float(np.percentile(lengths, 50)),
                "75%": float(np.percentile(lengths, 75))
            }
        }

        total_execution_time = time.time() - eval_start_time

        summary = {
            "model": MODEL_NAME,
            "model_path": MODEL_PATH,
            "dataset": DATASET_NAME,
            "split": DATASET_SPLIT,
            "subset": DATASET_SUBSET if DATASET_NAME in ["math", "dapo"] else "N/A",
            "num_samples_per_question": NUM_SAMPLES,
            "seed": SEED,
            "total_questions": int(total_questions),
            # Main metrics
            "overall_mean_at_k": overall_mean_at_k,
            "overall_pass_at_k": overall_pass_at_k,
            "unbiased_pass_at_8": unbiased_pass_at_k if NUM_SAMPLES >= 8 else None,
            "overall_avg_response_length": overall_avg_length,
            # Distributions
            "mean_accuracy_distribution": accuracy_distribution,
            "pass_distribution": {
                "mean": float(np.mean(pass_values)),
                "std": float(np.std(pass_values)),
                "pass_count": int(sum(pass_values)),
                "total_questions": int(total_questions)
            },
            "response_length_distribution": length_distribution,
            # Configuration
            "batch_size": BATCH_SIZE,
            "world_size": world_size,
            "load_balance_method": args.load_balance_method,
            "config_file": os.path.basename(args.config),
            "log_file": os.path.basename(merged_log_path),
            "timestamp": timestamp,
            "generation_config": {
                "latent_length": latent_length,
                "noise_type": noise_type,
                "noise_strength": noise_strength,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "do_latent_sample": do_latent_sample,
                "do_discrete_sample": do_discrete_sample,
            },
            "total_execution_time_seconds": total_execution_time,
            "peak_gpu_memory_per_gpu_mb": all_peak_memories_list,
            "max_peak_gpu_memory_mb": max_peak_gpu_memory,
        }

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Delete temporary log files
        for r in range(world_size):
            rank_log_file = os.path.join(LOG_DIR,
                                         f"{MODEL_NAME}_{DATASET_NAME}_{DATASET_SPLIT}_{DATASET_SUBSET}_latent{latent_length}_noise_{noise_type}_{noise_str}_temp{temp_str}_{NUM_SAMPLES}samples_seed{SEED}_{timestamp}_rank{r}.jsonl")
            if os.path.exists(rank_log_file):
                os.remove(rank_log_file)

        print(f"\nEvaluation complete. Logs saved to {LOG_DIR}")
        print(f"  Merged logs: {merged_log_path}")
        print(f"  Summary: {summary_path}")
        print(f"  Mean@{NUM_SAMPLES}: {overall_mean_at_k:.4f}")
        print(f"  Pass@{NUM_SAMPLES}: {overall_pass_at_k:.4f}")
        if NUM_SAMPLES >= 8:
            print(f"  Unbiased Pass@8: {unbiased_pass_at_k:.4f}")
        print(f"Total Execution Time: {total_execution_time:.2f} seconds")

    # Clean up distributed environment
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()