import matplotlib.pyplot as plt
import seaborn as sns
import torch 
from preprocessing import compute_spectral_features, compute_flowmat
import networkx as nx
import numpy as np

# Parameters
sizes = [5, 5]  # two communities with 10 nodes each
p_intra = 0.8     # high probability of edges inside communities
p_inter = 0.2    # low probability of edges between communities
probs = [[p_intra, p_inter], [p_inter, p_intra]]  # SBM probability matrix

# Generate SBM graph
G = nx.stochastic_block_model(sizes, probs, seed=42)

# Convert to numpy adjacency matrix
adj_matrix = torch.tensor(nx.to_numpy_array(G))

# alpha = -1 
lowest, _, _ = compute_spectral_features(adj_matrix, num_nodes=6, k=5, alpha=-1, gamma_diff=0, gamma_adv=1)
F_r, F_norm_edge, F_dig= compute_flowmat(adj_matrix, lowest)

# alpha = 0
lowest, _, _ = compute_spectral_features(adj_matrix, num_nodes=6, k=5, alpha=0, gamma_diff=0, gamma_adv=1)
F_l, F_norm_edge, F_dig= compute_flowmat(adj_matrix, lowest)

# alpha = 1
lowest, _, _ = compute_spectral_features(adj_matrix, num_nodes=6, k=5, alpha=1, gamma_diff=0, gamma_adv=1)
F_a, F_norm_edge, F_dig= compute_flowmat(adj_matrix, lowest)



fig, axs = plt.subplots(1, 3, figsize=(18, 5))  # 1 row, 2 cols

# Plot heatmap 1
sns.heatmap(F_r.T, ax=axs[0], cmap='magma', annot=True)
axs[0].set_title("Repelling 1")

# Plot heatmap 2
sns.heatmap(F_l.T, ax=axs[1], cmap='magma', annot=True)
axs[1].set_title("Laplacian")

# Plot heatmap 2
sns.heatmap(F_a.T, ax=axs[2], cmap='magma', annot=True)
axs[2].set_title("Attracting 2")

plt.tight_layout()
plt.show()