import torch

def create_filter_list(S, K):
    """
    S: Graph Shift Operator 
    Create polynomial filter list: [I, S, S^2, ..., S^K] with self loops
    """
    powers = [torch.eye(S.size(0), device=S.device)]  # L^0 = I
    for _ in range(1, K + 1):
        powers.append(torch.matmul(S, powers[-1]))
    return powers