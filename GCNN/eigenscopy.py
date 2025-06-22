import torch
from torch_geometric.datasets import QM9
from torch_geometric.utils import to_dense_adj
import numpy as np
import matplotlib.pyplot as plt
from operators import hub_laplacian, normalized_adjacency, normalized_laplacian, adv_diff, turbohub_laplacian

# Define available operators
OPERATORS = {
    "hub_laplacian": hub_laplacian,
    "normalized_adjacency": normalized_adjacency,
    "normalized_laplacian": normalized_laplacian,
    "adv_diff": adv_diff,
    "turbohub_laplacian": turbohub_laplacian,
}

# User selects two operators to compare
OPERATOR_NAMES = ['normalized_laplacian', 'turbohub_laplacian']  # example selection

# Histogram settings
NUM_BINS = 50


def compute_spectrum(M: torch.Tensor) -> tuple[float, float]:
    """
    Compute spectral gap and range for matrix M.
    """
    M_np = M.cpu().numpy()
    evals = np.linalg.eigvals(M_np)
    lam = np.sort(evals)
    spectral_gap = np.abs(lam[1] - lam[0])
    spectral_range = lam[-1] - lam[0]
    return spectral_gap, spectral_range


def process_graphs(adj_matrices: list[torch.Tensor], operator_fn, alpha: float) -> tuple[list[float], list[float]]:
    """
    Apply operator_fn with alpha to each adjacency, compute spectra.
    """
    gaps, ranges = [], []
    for A in adj_matrices:
        L = operator_fn(A, alpha)
        gap, rng = compute_spectrum(L)
        gaps.append(gap)
        ranges.append(rng)
    return gaps, ranges


def plot_comparison(gaps1, ranges1, gaps2, ranges2, name1, name2):
    plt.figure(figsize=(18, 5))

    # Spectral gap
    plt.subplot(1, 2, 1)
    plt.hist(gaps1, bins=NUM_BINS, alpha=0.7, label=name1)
    plt.hist(gaps2, bins=NUM_BINS, alpha=0.7, label=name2)
    plt.title('Spectral Gap Comparison')
    plt.xlabel('Spectral Gap')
    plt.ylabel('Frequency')
    plt.legend()

    # Spectral range
    plt.subplot(1, 2, 2)
    plt.hist(ranges1, bins=NUM_BINS, alpha=0.7, label=name1)
    plt.hist(ranges2, bins=NUM_BINS, alpha=0.7, label=name2)
    plt.title('Spectral Range Comparison')
    plt.xlabel('Range')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    # Load subset of QM9
    dataset = QM9(root='data/QM9')[:10000]

    # Build dense adjacencies
    adj_matrices = [to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0] for data in dataset]

    # Select operators and alphas
    name1, name2 = OPERATOR_NAMES
    fn1 = OPERATORS[name1]
    fn2 = OPERATORS[name2]
    alpha = 1.0  # example alpha value

    # Compute spectra
    gaps1, ranges1 = process_graphs(adj_matrices, fn1, alpha)
    gaps2, ranges2 = process_graphs(adj_matrices, fn2, alpha)

    # Plot results
    plot_comparison(gaps1, ranges1, gaps2, ranges2, name1, name2)

if __name__ == '__main__':
    main()
