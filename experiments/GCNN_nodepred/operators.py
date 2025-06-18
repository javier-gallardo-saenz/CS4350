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
    
    L = batched_hub_laplacian(A, alpha=0)
    Lhub = batched_hub_laplacian(A, alpha)
    return gamma_adv * Lhub  + gamma_diff * torch.transpose(L, 1, 2)

############################################
#---------------NOT BATCHED ----------------
############################################

def hub_laplacian(A: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Unbatched hub Laplacian for one graph.
    A: [N, N] adjacency (no padding, N = true node count).
    alpha: parameter.
    """
    A = A.to(torch.float)
    deg = A.sum(dim=1)                 # [N]

    if alpha == 0:
        D = torch.diag(deg)            # [N, N]
        return D - A

    D_alpha     = torch.diag(deg.pow(alpha))    # [N, N]
    D_neg_alpha = torch.diag(deg.pow(-alpha))   # [N, N]

    #  Xi:
    deg_i = deg.view(-1, 1)      # [N, 1]
    deg_j = deg.view(1, -1)      # [1, N]
    ratio = (deg_j / deg_i).pow(alpha) # [N, N]

    # mask out non-edges
    ratio = ratio * A

    Xi = torch.diag(ratio.sum(dim=1))  # [N, N]

    #hub Laplacian
    return Xi - D_neg_alpha @ A @ D_alpha


def adv_diff(A: torch.Tensor,
             alpha: float,
             gamma_diff: float,
             gamma_adv: float) -> torch.Tensor:
    """
    Unbatched adversarial-diffusive operator for one graph.
    A: [N, N] adjacency.
    alpha: hub parameter.
    gamma_diff / gamma_adv: mixing weights.
    """
    # Standard Laplacian (alpha=0)
    L_diff = hub_laplacian(A, alpha=0)

    # Hub Lap (alpha)
    L_hub  = hub_laplacian(A, alpha=alpha)

    # Combine
    return gamma_adv * L_hub + gamma_diff * L_diff.T
