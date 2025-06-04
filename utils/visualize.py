import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.linalg import eig
import matplotlib.pyplot as plt
from operators import hub_laplacian, hub_advection_diffusion

def plot_eigvecs(plotting=True, n_eigs = 10):
    # build graph
    probs = [
        [0.8, 0.2],  # Block 1
        [0.2, 0.8]  # Block 2
    ]
    G = nx.stochastic_block_model([5, 5], probs, seed=42)
    # Compute adjacency and Laplacian matrices
    A = nx.adjacency_matrix(G, weight='weight')
    #L = nx.laplacian_matrix(G, weight='weight')
    Lh = hub_laplacian(A, alpha=1.0)

    # Compute Fiedler vector
    eigvals, eigvecs = eig(Lh)
    eigvals = eigvals.real
    eigvecs = eigvecs.real

    print(eigvals)


    if plotting:
        pos = nx.spring_layout(G, seed=42)
        for i in range(n_eigs):
            eigvec = eigvecs[:, i]
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(1, 1, 1)

            nodes = nx.draw_networkx_nodes(
                G,
                pos=pos,
                node_color=list(eigvec),
                cmap=plt.cm.viridis,
                node_size=500,
                ax=ax
            )

            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
            nx.draw_networkx_labels(G, pos, ax=ax)

            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                       norm=plt.Normalize(vmin=min(eigvec), vmax=max(eigvec)))
            sm.set_array([])

            fig.colorbar(sm, ax=ax, label="Node Heatmap Value")

            ax.set_axis_off()
            plt.title(f"Eigenvector {i}")
            plt.show()

    return

plot_eigvecs()