from torch_geometric.datasets import QM9
from torch_geometric.utils import to_dense_adj
import numpy as np
import matplotlib.pyplot as plt
from operators import hub_laplacian, normalized_adjacency, normalized_laplacian,adv_diff
from scipy.special import binom


B = 100  # number of bins for histogram in entropy calculation

def compute_spectrum(M):
    M = M.numpy()
    evals = np.linalg.eigvals(M)
    
    lam = np.sort(evals)
    n = lam.shape[0]

    # Mean pairwise distance 
    #mpd = np.sum(np.abs(np.diff(lam))) / n

    sg = np.abs(lam[1] -lam[0])

    # Spectral range
    size = lam[-1] - lam[0]

    return sg, size


def process_graphs(matrices):
    mpds = []
    sizes = []
    for mat in matrices:
        mpd, size = compute_spectrum(mat)
        mpds.append(mpd)
        sizes.append(size)
    return mpds, sizes


def compare_properties(adjacency_matrices):
    L0_matrices = []
    L1_matrices = []

    for adj in adjacency_matrices:
        L0 = hub_laplacian(adj, alpha=0)
        L1 = hub_laplacian(adj, alpha=1)
        #L1 = adv_diff(adj, alpha= 1, gamma_adv = 0.5, gamma_diff =0.5)


        L0_matrices.append(L0)
        L1_matrices.append(L1)

    mpds_L0, sizes_L0 = process_graphs(L0_matrices)
    mpds_L1, sizes_L1 = process_graphs(L1_matrices)

    plot_histograms(mpds_L0, mpds_L1, sizes_L0, sizes_L1)


def plot_histograms(mpd_L0, mpd_L1, size_L0, size_L1):
    plt.figure(figsize=(18, 5))

    # MPD plot
    plt.subplot(1, 2, 1)
    plt.hist(mpd_L0, bins=100, alpha=0.7, label='L (alpha=0)')
    plt.hist(mpd_L1, bins=100, alpha=0.7, label='L1 (alpha=1)')
    plt.title('Spetral gap')
    plt.xlabel('sg')
    plt.ylabel('Frequency')
    plt.legend()

    # Size plot
    plt.subplot(1, 2, 2)
    plt.hist(size_L0, bins=100, alpha=0.7, label='L (alpha=0)')
    plt.hist(size_L1, bins=100, alpha=0.7, label='L1 (alpha=1)')
    plt.title('Spectral Range')
    plt.xlabel('Size (Range)')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()


# ===========================
# Data Loading and Processing
# ===========================

dataset = QM9(root='data/QM9')
dataset = dataset[:1000]

adjacency_matrices = []

for data in dataset:
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    adjacency_matrices.append(adj)

# Run the comparison and plotting
compare_properties(adjacency_matrices)
