import torch

def create_filter_list(laplacian, K):
    """
    Generate a list of spectral filter operators up to order K.
    L: Graph Laplacian or normalized adjacency matrix.
    K: Order of the spectral filter (number of hops).
    Returns: List of length K+1 with matrices to apply to features.
    """
    powers = [torch.eye(laplacian.size(0), device=laplacian.device)]  # L^0 = I
    for _ in range(1, K + 1):
        powers.append(torch.matmul(laplacian, powers[-1]))
    return powers