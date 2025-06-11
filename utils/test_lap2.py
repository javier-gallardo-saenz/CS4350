import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx
from preprocessing import compute_spectral_features
from utils.operatorsNP import hub_laplacian

# SBM parameters
sizes = [5, 5]
p_intra = 0.8
p_inter = 0.2
probs = [[p_intra, p_inter], [p_inter, p_intra]]

# Generate SBM graph and adjacency matrix
G = nx.stochastic_block_model(sizes, probs, seed=42)
adj_matrix = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)

# Alpha values for first row
alphas = [-1, 1, -0.5, 0.5]
titles = [f"Alpha = {a}" for a in alphas]

# Create 2x3 subplot grid
fig, axs = plt.subplots(2, 3, figsize=(16, 10))
axs = axs.flatten()

# Plot the SBM graph
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, ax=axs[0], with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
axs[0].set_title("Stochastic Block Model Graph")

# Plot spectral bar charts for alphas ≠ 0
for i, alpha in enumerate(alphas):
    _, _, eig_vals = compute_spectral_features(
        adj=adj_matrix,
        num_nodes=adj_matrix.shape[0],
        k=adj_matrix.shape[0],
        laplacian_fn=hub_laplacian,
        alpha=alpha
    )
    print(f"{alpha}: eigenvals {eig_vals}")
    ax = axs[i + 1]  # Offset by 1 due to graph in axs[0]
    ax.bar(np.arange(1, len(eig_vals) + 1), eig_vals.numpy())
    ax.set_title(titles[i])
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue")

# Plot alpha = 0 spectrum in the bottom center
_, _, eig_vals_0 = compute_spectral_features(
    adj=adj_matrix,
    num_nodes=adj_matrix.shape[0],
    k=adj_matrix.shape[0],
    laplacian_fn=hub_laplacian,
    alpha=0.0
)
axs[5].bar(np.arange(len(eig_vals_0)), eig_vals_0.numpy(), color='red')
axs[5].set_title("Alpha = 0 (Laplacian)")
axs[5].set_xlabel("Index")
axs[5].set_ylabel("Eigenvalue")

plt.tight_layout()
plt.show()
