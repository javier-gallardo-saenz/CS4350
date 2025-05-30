import torch

def hub_laplacian(A: torch.Tensor, alpha: float) -> torch.Tensor:
    assert A.dim() == 2 and A.shape[0] == A.shape[1], "Adjacency matrix must be square"
    A = A.float()
    
    # degree vector
    deg = A.sum(dim=1)

    if (deg == 0).any():
        raise ValueError("Graph has disconnected components (nodes with degree 0)")

    # D^alpha and D^-alpha
    D_alpha = torch.diag(deg.pow(alpha))
    D_neg_alpha = torch.diag(deg.pow(-alpha))

    # Compute Ξ_α: diag( sum_{w in N(v)} (d_w/d_v)^α )
    deg_matrix_ratio = deg.view(1, -1) / deg.view(-1, 1)  # shape: (N, N)
    deg_ratio_pow = deg_matrix_ratio.pow(alpha) * A  # keep only neighbors
    Xi_alpha = torch.diag(deg_ratio_pow.sum(dim=1))

    L_alpha = Xi_alpha - D_neg_alpha @ A @ D_alpha

    return L_alpha

def hub_advection_diffusion(A: torch.Tensor,
                            alpha: float,
                            gamma_diff: float,
                            gamma_adv: float) -> torch.Tensor:
    
    return gamma_adv * hub_laplacian(A, alpha) + gamma_diff * hub_laplacian(A, alpha=0)
