import numpy as np
import scipy.sparse as sp
import networkx as nx
from netgen.meshing import Mesh as NetgenMesh
# from ngsolve import Mesh
# from netgen.read_gmsh import ReadGmsh
# import os
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

def plot_Laplacian_eigvals(G, plotting=True, n_eigs = 4):
    # build graph

    # Compute adjacency and Laplacian matrices
    A = nx.adjacency_matrix(G, weight='weight')
    L = nx.laplacian_matrix(G, weight='weight')

    # Compute Fiedler vector
    eigvals, eigvecs = eigsh(L, k=3, which='SM')
    print(eigvals)


    if plotting:
        pos = nx.spring_layout(G)
        for i in range(1, n_eigs):
            eigvec = eigvecs[:, i]
            # Draw nodes with heatmap coloring
            plt.figure(figsize=(8, 6))
            nodes = nx.draw_networkx_nodes(
                G,
                pos=pos,
                node_color=list(eigvec),
                cmap=plt.cm.viridis,
                node_size=500
            )

            # Draw edges and labels
            nx.draw_networkx_edges(G, pos, alpha=0.5)
            nx.draw_networkx_labels(G, pos)

            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                       norm=plt.Normalize(vmin=min(eigvec), vmax=max(eigvec)))
            sm.set_array([])
            plt.colorbar(sm, label="Node Heatmap Value")

            plt.axis("off")
            plt.title("NetworkX Graph with Node Heatmap")
            plt.show()

    return G, L