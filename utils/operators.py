import torch
from torch_geometric.utils import get_laplacian

def batched_hub_laplacian(A: torch.Tensor, alpha: float) -> torch.tensor:
    #assume that A is tensor [B, N, N]
    deg = A.sum(dim=2) # [B, N]

    if alpha ==0:
        D = torch.diag_embed(deg)
        return D - A
    
    D_alpha = torch.diag_embed(deg.pow(alpha)) #[B, N, N]
    D_neg_alpha = torch.diag_embed(deg.pow(-alpha))

    # compute Ξ_α: diag( sum_{w in N(v)} (d_w/d_v)^α )
    deg_i = deg.unsqueeze(2)  # [B, N, 1]
    deg_j = deg.unsqueeze(1)  # [B, 1, N]

    ratio = (deg_j / deg_i).pow(alpha)

    # mask out non-edges:
    deg_ratio_pow = ratio * A      # [B,N,N]
    
    # sum over neighbors:
    Xi = torch.diag_embed(deg_ratio_pow.sum(dim=2)) #[B, N, N]

    S = Xi - D_neg_alpha @ A @ D_alpha

    return S

def batched_adv_diff(A: torch.Tensor, 
                    alpha: float,
                    gamma_diff: float,
                    gamma_adv: float) -> torch.Tensor:
    
    gamma_adv * batched_hub_laplacian(A, alpha) + gamma_diff * batched_hub_laplacian(A, alpha=0)
