import torch
import scipy as sp
import networkx as nx
import numpy as np

def compute_laplacian(adj, alpha=0):
    """
    Get the Laplacian/normalized Laplacian matrix
    --------------
        adj: np.array
            Symmetric adjacency matrix
            Shape: (N, N)
        
        alpha: int
            0: Laplacian
            1: Hubs-repelling Laplacian
            -1: Hubs-attracting Laplacian
        
    Returns
    -------------
        L: np.array
            Resulting Laplacian matrix
    """

    # degree vector
    degrees = np.sum(adj, axis=1)
    D = np.diag(degrees)

    # Compute the inverse of the degree matrix
    D_inv = np.linalg.inv(D)

    diag_entries_A = np.zeros(len(degrees))
    diag_entries_R = np.zeros(len(degrees))

    for i in range(len(degrees)):
        neighbors = np.where(adj[i] > 0)[0]
        diag_entries_A[i] = np.sum(degrees[i]/degrees[neighbors])
        diag_entries_R[i] = np.sum(degrees[neighbors]/degrees[i])

    # Create diagonal matrix
    Eps_A = np.diag(diag_entries_A)
    Eps_R = np.diag(diag_entries_R)

    # Compute the Laplacian matrix
    if alpha == 0:
        L = D - adj
    elif alpha == 1:
        L = Eps_A - D @ adj @ D_inv
    else:
        L = Eps_R - D_inv @ adj @ D
    return L
    