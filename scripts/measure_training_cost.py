"""
Measure training cost for DLaR vs baselines.

Usage:
    python scripts/measure_training_cost.py \
        --config configs/lepo/llama3_1b.yaml \
        --reward_type standard \
        --model_name DLaR \
        --output_dir ./cost_measurements
"""

import os
import sys
import json
import time
import argparse
import subprocess
import datetime

import torch
import pynvml

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/lepo/llama3_1b.yaml")
    parser.add_argument('--reward_type', type=str, default='standard')
    parser.add_argument('--model_name', type=str, default="DLaR",
                        help='Name for logging (e.g., GRPO, LEPO, DLaR)')
    parser.add_argument('--accelerate_config', type=str,
                        default='configs/accelerate/ds_zero2.yaml')
    parser.add_argument('--output_dir', type=str, default='./cost_measurements')
    parser.add_argument('--num_gpus', type=int, default=2)
    return parser.parse_args()


def get_gpu_memory_mb(gpu_id=0):
    """Get current GPU memory usage in MB."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return {
        'used_mb': info.used / (1024 ** 2),
        'total_mb': info.total / (1024 ** 2),
        'free_mb': info.free / (1024 ** 2),
    }


def measure_inference_latency(model_path, tokenizer_path, n_tokens=32, n_runs=20):
    """
    Measure inference latency for generating n_tokens.
    Returns average latency per token in ms.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"  Loading model for latency test...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    model.eval()

    # Simple prompt
    prompt = "Please reason step by step, and put your final answer within \\boxed{}.\n\nWhat is 2 + 2?"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    torch.cuda.synchronize()

    # Measure
    latencies = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        latencies.append(elapsed / n_tokens * 1000)  # ms per token

    avg_latency = sum(latencies) / len(latencies)
    print(f"  Average latency: {avg_latency:.2f} ms/token")

    del model
    torch.cuda.empty_cache()

    return avg_latency


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(f"Measuring training cost for: {args.model_name}")
    print("=" * 70)

    # ===== 1. Record start time and initial memory =====
    start_time = time.time()
    initial_memory = get_gpu_memory_mb(0)

    print(f"\n[Before training]")
    print(f"  GPU 0 memory: {initial_memory['used_mb']:.0f} / {initial_memory['total_mb']:.0f} MB")

    # ===== 2. Run training as subprocess =====
    cmd = [
        'accelerate', 'launch',
        '--config_file', args.accelerate_config,
        '--num_processes', str(args.num_gpus),
        'scripts/train_grpo.py',
        '--config', args.config,
        '--reward_type', args.reward_type,
    ]

    print(f"\n[Training command]")
    print(f"  {' '.join(cmd)}")
    print(f"\n[Training started]")

    # Monitor memory in background while training
    peak_memory_mb = [0.0] * args.num_gpus
    monitoring = True

    import threading

    def monitor_memory():
        while monitoring:
            for gpu_id in range(args.num_gpus):
                try:
                    mem = get_gpu_memory_mb(gpu_id)
                    peak_memory_mb[gpu_id] = max(peak_memory_mb[gpu_id], mem['used_mb'])
                except Exception:
                    pass
            time.sleep(5)

    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()

    # Run training
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='')
        process.wait()
        return_code = process.returncode
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
        process.kill()
        return_code = -1
    finally:
        monitoring = False
        monitor_thread.join(timeout=2)

    # ===== 3. Record end time and peak memory =====
    end_time = time.time()
    total_time_seconds = end_time - start_time
    total_time_hours = total_time_seconds / 3600

    print(f"\n[After training]")
    print(f"  Return code: {return_code}")
    print(f"  Total time: {total_time_hours:.2f} hours ({total_time_seconds:.0f} s)")
    print(f"  Peak GPU memory:")
    for gpu_id, peak in enumerate(peak_memory_mb):
        print(f"    GPU {gpu_id}: {peak:.0f} MB")

    # ===== 4. Measure inference latency =====
    print(f"\n[Measuring inference latency]")
    try:
        latency_ms = measure_inference_latency(
            args.config,  # placeholder; update to actual model path if needed
            args.config,
            n_tokens=32,
            n_runs=20,
        )
    except Exception as e:
        print(f"  Failed: {e}")
        latency_ms = None

    # ===== 5. Save results =====
    results = {
        'model_name': args.model_name,
        'config': args.config,
        'reward_type': args.reward_type,
        'num_gpus': args.num_gpus,
        'total_time_hours': total_time_hours,
        'total_time_seconds': total_time_seconds,
        'peak_gpu_memory_mb': peak_memory_mb,
        'max_peak_gpu_memory_mb': max(peak_memory_mb),
        'inference_latency_ms_per_token': latency_ms,
        'return_code': return_code,
        'timestamp': datetime.datetime.now().isoformat(),
    }

    output_path = os.path.join(
        args.output_dir,
        f'cost_{args.model_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[Results saved]")
    print(f"  {output_path}")
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()