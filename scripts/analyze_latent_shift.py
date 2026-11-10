"""
Analyze latent space shift using PCA (PyTorch implementation).

Usage:
    python scripts/analyze_latent_shift.py \
        --init_dir ./analysis/hidden_states/init/ \
        --trained_dirs ./analysis/hidden_states/lepo/ ./analysis/hidden_states/dlar/ \
        --model_names LEPO DLaR \
        --output_dir ./figures/
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze latent space shift using PCA')
    parser.add_argument('--init_dir', type=str, required=True,
                        help='Directory containing baseline hidden states')
    parser.add_argument('--trained_dirs', type=str, nargs='+', required=True,
                        help='Directories containing trained model hidden states')
    parser.add_argument('--model_names', type=str, nargs='+', required=True,
                        help='Names for trained models')
    parser.add_argument('--output_dir', type=str, default='./figures',
                        help='Output directory for figures')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['gsm8k', 'math500', 'mmlu_stem', 'arc_challenge'],
                        help='Datasets to analyze')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Figure DPI')
    parser.add_argument('--n_components', type=int, default=2,
                        help='Number of PCA components')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    return parser.parse_args()


def load_hidden_states(directory, dataset):
    """Load hidden states from directory."""
    file_path = os.path.join(directory, f'{dataset}_hidden_states.npy')
    if os.path.exists(file_path):
        return np.load(file_path)
    return None


def compute_centroid(hidden_states):
    """
    Compute centroid of hidden states.

    Args:
        hidden_states: [layers, samples, hidden_dim]

    Returns:
        centroid: [layers, hidden_dim]
    """
    return np.mean(hidden_states, axis=1)


def compute_shift_distance(centroid1, centroid2):
    """
    Compute Euclidean distance between two centroids.

    Args:
        centroid1: [layers, hidden_dim]
        centroid2: [layers, hidden_dim]

    Returns:
        distances: [layers]
    """
    return np.linalg.norm(centroid1 - centroid2, axis=1)


def pca_torch(data, n_components=2, device='cpu'):
    """
    Perform PCA using PyTorch.

    Args:
        data: [n_samples, n_features]
        n_components: number of components
        device: device to use

    Returns:
        projected: [n_samples, n_components]
        explained_variance: [n_components]
        explained_variance_ratio: [n_components]
        components: [n_components, n_features]
        mean: [n_features]
    """
    # Convert to torch tensor
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    data = data.to(device)

    # Center the data
    mean = data.mean(dim=0, keepdim=True)
    centered = data - mean

    # Compute covariance matrix
    n_samples = data.shape[0]
    cov = centered.T @ centered / (n_samples - 1)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Sort by eigenvalues in descending order
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top components
    components = eigenvectors[:, :n_components].T  # [n_components, n_features]

    # Project data
    projected = centered @ components.T  # [n_samples, n_components]

    # Compute explained variance
    total_var = eigenvalues.sum()
    explained_variance = eigenvalues[:n_components]
    explained_variance_ratio = explained_variance / total_var

    return (
        projected.cpu().numpy(),
        explained_variance.cpu().numpy(),
        explained_variance_ratio.cpu().numpy(),
        components.cpu().numpy(),
        mean.cpu().numpy(),
    )


def pca_numpy(data, n_components=2):
    """
    Perform PCA using NumPy (no sklearn dependency).

    Args:
        data: [n_samples, n_features]
        n_components: number of components

    Returns:
        projected: [n_samples, n_components]
        explained_variance: [n_components]
        explained_variance_ratio: [n_components]
        components: [n_components, n_features]
        mean: [n_features]
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
    components = eigenvectors[:, :n_components].T  # [n_components, n_features]

    # Project data
    projected = centered @ components.T  # [n_samples, n_components]

    # Compute explained variance
    total_var = eigenvalues.sum()
    explained_variance = eigenvalues[:n_components]
    explained_variance_ratio = explained_variance / total_var

    return projected, explained_variance, explained_variance_ratio, components, mean


def compute_pca_projection(hidden_states, n_components=2, device='cpu'):
    """
    Compute PCA projection of hidden states.

    Args:
        hidden_states: [layers, samples, hidden_dim]
        n_components: number of components
        device: device to use

    Returns:
        projected: [layers, samples, n_components]
        explained_variance: [layers, n_components]
        explained_variance_ratio: [layers, n_components]
    """
    layers, samples, hidden_dim = hidden_states.shape

    # Flatten: [layers * samples, hidden_dim]
    flattened = hidden_states.reshape(-1, hidden_dim)

    # Perform PCA
    projected, explained_var, explained_var_ratio, components, mean = pca_torch(
        flattened, n_components, device
    )

    # Reshape back to [layers, samples, n_components]
    projected = projected.reshape(layers, samples, n_components)
    explained_var = explained_var.reshape(layers, n_components)
    explained_var_ratio = explained_var_ratio.reshape(layers, n_components)

    return projected, explained_var, explained_var_ratio, components, mean


def compute_pca_projection_numpy(hidden_states, n_components=2):
    """
    Compute PCA projection using NumPy (no sklearn).

    Args:
        hidden_states: [layers, samples, hidden_dim]
        n_components: number of components

    Returns:
        projected: [layers, samples, n_components]
        explained_variance: [layers, n_components]
        explained_variance_ratio: [layers, n_components]
    """
    layers, samples, hidden_dim = hidden_states.shape

    # Flatten: [layers * samples, hidden_dim]
    flattened = hidden_states.reshape(-1, hidden_dim)

    # Perform PCA
    projected, explained_var, explained_var_ratio, components, mean = pca_numpy(
        flattened, n_components
    )

    # Reshape back to [layers, samples, n_components]
    projected = projected.reshape(layers, samples, n_components)
    explained_var = explained_var.reshape(layers, n_components)
    explained_var_ratio = explained_var_ratio.reshape(layers, n_components)

    return projected, explained_var, explained_var_ratio, components, mean


def plot_pca_comparison(init_proj, trained_projs, model_names, dataset, output_dir,
                        layers=[0, 4, 8, 12, 16, 20, 24, 28, 31],
                        dpi=300, n_cols=3):
    """
    Plot PCA comparison between models.
    """
    n_layers = len(layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    colors = ['gray', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    alpha_init = 0.2
    alpha_trained = 0.5

    for idx, layer in enumerate(layers):
        if idx >= len(axes):
            break
        ax = axes[idx]

        # Plot baseline (init)
        ax.scatter(init_proj[layer, :, 0], init_proj[layer, :, 1],
                  color='gray', alpha=alpha_init, s=8, label='Init' if idx == 0 else '')

        # Plot trained models
        for i, (proj, name) in enumerate(zip(trained_projs, model_names)):
            ax.scatter(proj[layer, :, 0], proj[layer, :, 1],
                      color=colors[i+1], alpha=alpha_trained, s=8,
                      label=name if idx == 0 else '')

        # Plot centroids
        init_centroid = np.mean(init_proj[layer], axis=0)
        ax.scatter(init_centroid[0], init_centroid[1],
                  color='black', marker='X', s=100, label='Init Centroid' if idx == 0 else '')

        for i, proj in enumerate(trained_projs):
            centroid = np.mean(proj[layer], axis=0)
            ax.scatter(centroid[0], centroid[1],
                      color=colors[i+1], marker='X', s=100,
                      label=f'{model_names[i]} Centroid' if idx == 0 else '')

        ax.set_title(f'Layer {layer}')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'PCA Projection - {dataset}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = os.path.join(output_dir, f'pca_{dataset}_comparison.pdf')
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA comparison to {output_file}")

    # Also save as PNG
    png_file = os.path.join(output_dir, f'pca_{dataset}_comparison.png')
    plt.savefig(png_file, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_layer_shift_distances(results, output_dir, dpi=300):
    """
    Plot layer-wise shift distances.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (model_name, model_results) in enumerate(results.items()):
        # Average across datasets
        all_distances = []
        for dataset, metrics in model_results.items():
            all_distances.append(metrics['layer_distances'])

        if all_distances:
            avg_distances = np.mean(all_distances, axis=0)
            std_distances = np.std(all_distances, axis=0)
            layers = np.arange(len(avg_distances))

            ax.plot(layers, avg_distances, color=colors[i],
                   linewidth=2, label=model_name)
            ax.fill_between(layers, avg_distances - std_distances,
                           avg_distances + std_distances,
                           color=colors[i], alpha=0.15)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Centroid Shift Distance', fontsize=12)
    ax.set_title('Layer-wise Latent Shift', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    output_file = os.path.join(output_dir, 'layer_shift_distances.pdf')
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved layer shift distances to {output_file}")


def generate_latex_table(results, model_names, output_dir):
    """
    Generate LaTeX table from results.
    """
    # Group by Math vs Knowledge
    math_datasets = ['gsm8k', 'math500']
    knowledge_datasets = ['mmlu_stem', 'arc_challenge']

    latex_lines = []
    latex_lines.append("\\begin{table}[!ht]")
    latex_lines.append("\\centering")
    latex_lines.append("\\begin{tabular}{lcc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Model & Math Datasets & Knowledge Datasets \\\\")
    latex_lines.append("\\midrule")

    for model_name in model_names:
        if model_name not in results:
            continue

        model_results = results[model_name]
        math_means = []
        knowledge_means = []

        for dataset in math_datasets:
            if dataset in model_results:
                math_means.append(model_results[dataset]['mean'])

        for dataset in knowledge_datasets:
            if dataset in knowledge_datasets:
                knowledge_means.append(model_results[dataset]['mean'])

        math_avg = np.mean(math_means) if math_means else 0
        knowledge_avg = np.mean(knowledge_means) if knowledge_means else 0

        latex_lines.append(f"{model_name} & {math_avg:.3f} & {knowledge_avg:.3f} \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\caption{PCA shift distances of representation centers.}")
    latex_lines.append("\\label{tab:pca_shift}")
    latex_lines.append("\\end{table}")

    with open(os.path.join(output_dir, 'pca_table.tex'), 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"LaTeX table saved to {os.path.join(output_dir, 'pca_table.tex')}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    print(f"Using device: {device}")

    # Load baseline hidden states
    print("Loading baseline hidden states...")
    init_hidden = {}
    for dataset in args.datasets:
        states = load_hidden_states(args.init_dir, dataset)
        if states is not None:
            init_hidden[dataset] = states
            print(f"  Loaded {dataset}: {states.shape}")
        else:
            print(f"  Warning: {dataset} not found in {args.init_dir}")

    if not init_hidden:
        print("Error: No baseline hidden states loaded!")
        return

    # Load trained hidden states
    print("\nLoading trained hidden states...")
    trained_hidden = {}
    for model_name, dir_path in zip(args.model_names, args.trained_dirs):
        trained_hidden[model_name] = {}
        for dataset in args.datasets:
            states = load_hidden_states(dir_path, dataset)
            if states is not None:
                trained_hidden[model_name][dataset] = states
                print(f"  {model_name} - {dataset}: {states.shape}")
            else:
                print(f"  Warning: {model_name} - {dataset} not found in {dir_path}")

    # Compute centroids
    print("\nComputing centroids...")
    init_centroids = {dataset: compute_centroid(init_hidden[dataset])
                     for dataset in init_hidden}

    trained_centroids = {}
    for model_name in trained_hidden:
        trained_centroids[model_name] = {}
        for dataset in trained_hidden[model_name]:
            trained_centroids[model_name][dataset] = compute_centroid(
                trained_hidden[model_name][dataset]
            )

    # Compute shift distances
    print("\nComputing shift distances...")
    results = {}
    for model_name in trained_centroids:
        results[model_name] = {}
        for dataset in trained_centroids[model_name]:
            centroid1 = init_centroids.get(dataset)
            centroid2 = trained_centroids[model_name][dataset]
            if centroid1 is not None and centroid2 is not None:
                distances = compute_shift_distance(centroid1, centroid2)
                results[model_name][dataset] = {
                    'mean': float(np.mean(distances)),
                    'std': float(np.std(distances)),
                    'max': float(np.max(distances)),
                    'min': float(np.min(distances)),
                    'layer_distances': distances.tolist(),
                }

    # Print results
    print("\n" + "="*70)
    print("LATENT SPACE SHIFT RESULTS")
    print("="*70)
    print(f"{'Model':<15} {'Dataset':<15} {'Mean Shift':<12} {'Std':<10} {'Max':<10}")
    print("-"*70)
    for model_name in results:
        for dataset, metrics in results[model_name].items():
            print(f"{model_name:<15} {dataset:<15} {metrics['mean']:<12.4f} {metrics['std']:<10.4f} {metrics['max']:<10.4f}")
    print("="*70)

    # Compute PCA projections
    print("\nComputing PCA projections...")
    pca_dataset = args.datasets[0]

    if pca_dataset not in init_hidden:
        print(f"Error: {pca_dataset} not found in init_hidden!")
        return

    try:
        init_proj, init_explained_var, init_explained_ratio, _, _ = compute_pca_projection(
            init_hidden[pca_dataset], args.n_components, device
        )
        print(f"  Baseline explained variance: {init_explained_ratio[0, 0]:.3f}, {init_explained_ratio[0, 1]:.3f}")
    except Exception as e:
        print(f"Error computing PCA for baseline: {e}")
        return

    trained_projs = []
    for model_name in trained_hidden:
        if pca_dataset in trained_hidden[model_name]:
            try:
                proj, explained_var, explained_ratio, _, _ = compute_pca_projection(
                    trained_hidden[model_name][pca_dataset], args.n_components, device
                )
                trained_projs.append(proj)
                print(f"  {model_name} explained variance: {explained_ratio[0, 0]:.3f}, {explained_ratio[0, 1]:.3f}")
            except Exception as e:
                print(f"Error computing PCA for {model_name}: {e}")
                trained_projs.append(None)
        else:
            trained_projs.append(None)

    # Remove None
    valid_trained = [(proj, name) for proj, name in zip(trained_projs, args.model_names) if proj is not None]
    if valid_trained:
        trained_projs, valid_names = zip(*valid_trained)
        valid_names = list(valid_names)
    else:
        print("Error: No trained hidden states for PCA!")
        return

    # Plot PCA comparisons
    print("\nGenerating PCA plots...")
    layers_to_plot = [0, 4, 8, 12, 16, 20, 24, 28, 31]
    available_layers = min(init_proj.shape[0], 32)
    layers_to_plot = [l for l in layers_to_plot if l < available_layers]

    plot_pca_comparison(
        init_proj, trained_projs, valid_names,
        pca_dataset, args.output_dir,
        layers=layers_to_plot,
        dpi=args.dpi
    )

    # Plot layer shift distances
    if results:
        plot_layer_shift_distances(results, args.output_dir, args.dpi)

    # Generate LaTeX table
    if results:
        generate_latex_table(results, args.model_names, args.output_dir)

    # Save full results as JSON
    with open(os.path.join(args.output_dir, 'latent_shift_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {os.path.join(args.output_dir, 'latent_shift_results.json')}")

    # Save average results as CSV
    import csv
    csv_file = os.path.join(args.output_dir, 'latent_shift_summary.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Dataset', 'Mean_Shift', 'Std', 'Max', 'Min'])
        for model_name in results:
            for dataset, metrics in results[model_name].items():
                writer.writerow([
                    model_name, dataset,
                    f"{metrics['mean']:.4f}",
                    f"{metrics['std']:.4f}",
                    f"{metrics['max']:.4f}",
                    f"{metrics['min']:.4f}"
                ])
    print(f"CSV summary saved to {csv_file}")

    print("\nAll done!")


if __name__ == "__main__":
    main()