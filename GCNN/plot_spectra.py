from torch_geometric.datasets import QM9
from torch_geometric.utils import to_dense_adj
import numpy as np
import matplotlib.pyplot as plt
from operators import hub_laplacian

def compute_spectrum(data, alpha=0):
    """
    Computes eigenvalues of a graph operator for a molecular graph.

    Parameters:
    - data: PyG data object

    Returns:
    - Sorted eigenvalues as a numpy array
    """

    edge_index = data.edge_index
    num_nodes = data.num_nodes
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    M = hub_laplacian(adj, alpha=alpha)
    
    eigvals = np.linalg.eigvals(M)
    return np.sort(eigvals)

def plot_spectra(dataset, indices, alpha=0):
    """
    Plot spectra for selected graphs in the dataset.

    Parameters:
    - dataset: PyG dataset
    - indices: list of graph indices to plot
    - alpha: parameter for the hub Laplacian operator
    """
    plt.figure(figsize=(10, 5))
    for idx in indices:
        data = dataset[idx]
        eigvals = compute_spectrum(data, alpha=alpha)
        plt.plot(eigvals, marker='o', label=f'Graph {idx} (n={data.num_nodes})')

    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue')
    plt.title(f'Hub Laplacian Spectrum (alpha={alpha}) for Sample Graphs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()   

dataset = QM9(root='data/QM9')
plot_spectra(dataset, indices=[10, 11, 12, 13, 14], alpha=0)