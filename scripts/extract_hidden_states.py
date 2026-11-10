"""
Extract hidden states from trained and initialized models for PCA analysis.

Usage:
    # Step 1: Extract baseline (initialized model) hidden states + PCA params
    python scripts/extract_hidden_states.py \
        --model_path /path/to/base/model/ \
        --output_dir ./analysis/hidden_states/init/ \
        --dataset gsm8k \
        --num_samples 200 \
        --baseline

    # Step 2: Extract trained model hidden states (using baseline PCA params)
    python scripts/extract_hidden_states.py \
        --model_path ./outputs/lepo-run/checkpoint-150/ \
        --output_dir ./analysis/hidden_states/lepo/ \
        --dataset gsm8k \
        --num_samples 200 \
        --pca_ref_dir ./analysis/hidden_states/init/
"""

import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.data.datasets import get_dataset
import json
import argparse
import numpy as np


# ===== 支持的 6 个数据集 =====
SUPPORTED_DATASETS = [
    'gsm8k', 'math500', 'minervamath', 'amc23',
    'mmlu_stem', 'arc_challenge'
]


def parse_args():
    parser = argparse.ArgumentParser(description='Extract hidden states for PCA analysis')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model (trained checkpoint or base model)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save hidden states')
    parser.add_argument('--dataset', type=str, default='gsm8k',
                        choices=SUPPORTED_DATASETS,
                        help='Dataset to use')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to extract')
    parser.add_argument('--baseline', action='store_true',
                        help='Use this flag for baseline (untrained) model')
    parser.add_argument('--pca_ref_dir', type=str, default=None,
                        help='Directory with baseline PCA params (required for trained models)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu), auto-detect if not specified')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length for tokenization')
    return parser.parse_args()


def get_hidden_states(model, tokenizer, texts, device='cuda', max_length=128):
    """
    Extract mean-pooled hidden states from all layers.

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        texts: List of text strings
        device: Device to run on
        max_length: Maximum sequence length

    Returns:
        hidden_states: [n_layers, n_samples, hidden_dim]
    """
    all_hidden_states = []
    batch_size = 8

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Extract hidden states from all layers
        hidden_states = outputs.hidden_states  # tuple of [batch, seq, dim]

        # Stack all layers: [n_layers, batch, seq, dim]
        stacked = torch.stack(hidden_states, dim=0)

        # Mean pooling over sequence length (excluding padding)
        attention_mask = inputs['attention_mask']
        expanded_mask = attention_mask.unsqueeze(-1).unsqueeze(0)

        masked = stacked * expanded_mask
        sum_over_seq = masked.sum(dim=2)
        seq_lengths = attention_mask.sum(dim=1).clamp(min=1)
        avg = sum_over_seq / seq_lengths.unsqueeze(0).unsqueeze(-1)

        # Convert to float32 before numpy (avoid BFloat16 error)
        all_hidden_states.append(avg.float().cpu().numpy())

    return np.concatenate(all_hidden_states, axis=1)


def pca_numpy(data, n_components=2):
    """
    Perform PCA using NumPy.

    Args:
        data: [n_samples, n_features]
        n_components: number of components

    Returns:
        projected: [n_samples, n_components] (centered projection, for variance)
        components: [n_components, n_features] (principal directions)
        mean: [n_features] (mean of data)
    """
    # Center the data
    mean = np.mean(data, axis=0, keepdims=True)
    centered = data - mean

    # Compute covariance matrix
    n_samples = data.shape[0]
    cov = centered.T @ centered / (n_samples - 1)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top components
    components = eigenvectors[:, :n_components].T

    # Project centered data (used for explained variance)
    projected = centered @ components.T

    return projected, components, mean


def main():
    args = parse_args()

    # Determine device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto' if device == 'cuda' else None,
    )
    model.eval()
    print("Model loaded.")

    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    dataset = get_dataset(args.dataset, split='train', shot=0)

    # Get samples
    num_samples = min(args.num_samples, len(dataset))
    samples = dataset.select(range(num_samples))
    print(f"Loaded {num_samples} samples.")

    # Extract prompts
    prompts = []
    for sample in samples:
        prompt = sample['prompt']
        if isinstance(prompt, list):
            prompt_text = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = prompt
        prompts.append(prompt_text)

    # Extract hidden states
    print(f"Extracting hidden states from {len(prompts)} samples...")
    hidden_states = get_hidden_states(
        model, tokenizer, prompts, device, args.max_length
    )

    n_layers, n_samples, hidden_dim = hidden_states.shape
    print(f"Hidden states shape: {hidden_states.shape}")

    # Save directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save hidden states as numpy array
    hs_path = os.path.join(args.output_dir, f'{args.dataset}_hidden_states.npy')
    np.save(hs_path, hidden_states)
    print(f"Hidden states saved to {hs_path}")

    # ============================================================
    # Case 1: Baseline model
    # ============================================================
    if args.baseline:
        print("\n" + "=" * 60)
        print("Computing PCA parameters for BASELINE model...")
        print("=" * 60)

        pca_params = {}
        baseline_records = []

        for layer in range(n_layers):
            layer_data = hidden_states[layer]

            # Fit PCA on baseline data
            projected, components, mean = pca_numpy(layer_data, n_components=2)
            comp1 = components[0]
            comp2 = components[1]

            # Project ORIGINAL (non-centered) features onto components
            pc1 = float(layer_data.dot(comp1).mean())
            pc2 = float(layer_data.dot(comp2).mean())

            pca_params[f'layer{layer}_comp1'] = comp1
            pca_params[f'layer{layer}_comp2'] = comp2
            pca_params[f'layer{layer}_mean'] = mean[0]
            pca_params[f'layer{layer}_pc1'] = pc1
            pca_params[f'layer{layer}_pc2'] = pc2

            baseline_records.append({
                "layer": layer,
                "state": "Original",
                "shift": 0.0,
                "principle": pc2
            })

            if layer % 4 == 0 or layer == n_layers - 1:
                print(f"  Layer {layer:2d}: pc1={pc1:+.4f}, pc2={pc2:+.4f}")

        # Save PCA params
        pca_params_path = os.path.join(args.output_dir, f'{args.dataset}_pca_params.npz')
        np.savez(pca_params_path, **pca_params)
        print(f"\nPCA params saved to {pca_params_path}")

        # Save baseline PCA shift records
        output_json = os.path.join(args.output_dir, f'{args.dataset}_pca_shift.json')
        new_entry = {
            "benchmark": f"{args.dataset}_PCA_Shift",
            "data": baseline_records
        }
        with open(output_json, 'w') as f:
            json.dump([new_entry], f, indent=2)
        print(f"Baseline PCA shift data saved to {output_json}")

    # ============================================================
    # Case 2: Trained model
    # ============================================================
    else:
        if args.pca_ref_dir is None:
            print("ERROR: --pca_ref_dir is required for trained models!")
            return

        pca_params_path = os.path.join(args.pca_ref_dir, f'{args.dataset}_pca_params.npz')
        if not os.path.exists(pca_params_path):
            print(f"ERROR: PCA params not found at {pca_params_path}")
            return

        print("\n" + "=" * 60)
        print(f"Computing PCA shift using baseline params from {pca_params_path}")
        print("=" * 60)

        pca_data = np.load(pca_params_path, allow_pickle=True)

        pca_records = []

        for layer in range(n_layers):
            layer_data = hidden_states[layer]

            comp1 = pca_data[f'layer{layer}_comp1']
            comp2 = pca_data[f'layer{layer}_comp2']
            pc1_before = float(pca_data[f'layer{layer}_pc1'])
            pc2_before = float(pca_data[f'layer{layer}_pc2'])

            # Project ORIGINAL (non-centered) features
            pc1_after = float(layer_data.dot(comp1).mean())
            pc2_after = float(layer_data.dot(comp2).mean())

            shift = pc1_after - pc1_before

            pca_records.append({
                "layer": layer,
                "state": "Original",
                "shift": 0.0,
                "principle": pc2_before
            })
            pca_records.append({
                "layer": layer,
                "state": "Updated",
                "shift": shift,
                "principle": pc2_after
            })

            if layer % 4 == 0 or layer == n_layers - 1:
                print(f"  Layer {layer:2d}: shift={shift:+.4f}, "
                      f"pc2_before={pc2_before:+.4f}, pc2_after={pc2_after:+.4f}")

        # Save / append to PCA shift JSON
        output_json = os.path.join(args.output_dir, f'{args.dataset}_pca_shift.json')
        new_entry = {
            "benchmark": f"{args.dataset}_PCA_Shift",
            "data": pca_records
        }

        if os.path.exists(output_json):
            with open(output_json, 'r') as f:
                existing = json.load(f)
            if isinstance(existing, list):
                existing_benchmarks = [e.get('benchmark') for e in existing]
                if new_entry['benchmark'] in existing_benchmarks:
                    existing = [e for e in existing
                                if e.get('benchmark') != new_entry['benchmark']]
                existing.append(new_entry)
            else:
                existing = [existing, new_entry]
        else:
            existing = [new_entry]

        with open(output_json, 'w') as f:
            json.dump(existing, f, indent=2)
        print(f"\nPCA shift data saved to {output_json}")

    # ============================================================
    # Save metadata
    # ============================================================
    metadata = {
        'model_path': args.model_path,
        'dataset': args.dataset,
        'num_samples': n_samples,
        'num_layers': n_layers,
        'hidden_dim': hidden_dim,
        'shape': list(hidden_states.shape),
        'baseline': args.baseline,
        'device': device,
        'dtype': str(hidden_states.dtype),
    }
    metadata_path = os.path.join(args.output_dir, f'{args.dataset}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to {metadata_path}")
    print(f"Done! Shape: {hidden_states.shape}")


if __name__ == "__main__":
    main()