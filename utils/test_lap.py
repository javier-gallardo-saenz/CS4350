import matplotlib.pyplot as plt
import seaborn as sns
import torch 
from utils.operatorsNP import hub_laplacian
from preprocessing import compute_spectral_features, compute_flowmat
import networkx as nx
import numpy as np

# Parameters
sizes = [5, 5]  # two communities of 5 nodes each
p_intra = 0.8
p_inter = 0.2
probs = [[p_intra, p_inter], [p_inter, p_intra]]

# Generate SBM graph and adjacency
G = nx.stochastic_block_model(sizes, probs, seed=42)
adj_matrix = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)

# Alpha values to consider
alphas = [-1, 1, -0.5, 0.5]
titles = [f"Alpha = {a}" for a in alphas]

# Prepare flow matrices container
flow_matrices = []

# Compute flow matrices for each alpha
for alpha in alphas:
    lowest, _, _ = compute_spectral_features(
        adj_matrix,
        num_nodes=adj_matrix.shape[0],
        k=5,
        laplacian_fn=hub_laplacian,
        alpha=alpha
    )
    F, _ = compute_flowmat(adj_matrix, lowest)  # compute_flowmat returns two outputs
    flow_matrices.append(F)

# Compute for alpha=0 (bottom row)
lowest_0, _, _ = compute_spectral_features(
    adj_matrix,
    num_nodes=adj_matrix.shape[0],
    k=5,
    laplacian_fn=hub_laplacian,
    alpha=0  # Correct alpha=0 here
)
F_0, _ = compute_flowmat(adj_matrix, lowest_0)

# Create 2x3 subplot grid
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.flatten()

# Plot SBM graph top-left
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, ax=axs[0], with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
axs[0].set_title("Stochastic Block Model Graph")

# Plot flow matrices heatmaps for alphas in top row (excluding graph slot)
for i, (F, alpha) in enumerate(zip(flow_matrices, alphas)):
    ax = axs[i + 1]  # offset 1 for graph plot
    sns.heatmap(F.T, ax=ax, cmap='magma', annot=True)
    ax.set_title(f"Flow matrix (alpha={alpha})")

# Bottom row center plot: alpha=0 flow matrix, red colormap
sns.heatmap(F_0.T, ax=axs[5], cmap='Reds', annot=True)
axs[5].set_title("Flow matrix (alpha=0)")

# Hide empty subplots (bottom left: axs[3], bottom middle: axs[4]) for cleaner look
axs[3].axis('off')
axs[4].axis('off')

plt.tight_layout()
plt.show()
