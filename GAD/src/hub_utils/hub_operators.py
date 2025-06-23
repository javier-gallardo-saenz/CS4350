import torch
from typing import Callable
import numpy as np # Used only for type hints if needed, but not for computation within the function


def hub_laplacian_dense_pytorch(A: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Directly computes the Hub Laplacian as a dense PyTorch tensor
    for a given adjacency matrix A and alpha.
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "Adjacency matrix must be square"
    # Ensure A is float and on the correct device if not already
    A_float = A.to(torch.float32) # Or torch.float64 if higher precision is needed
    device = A_float.device
    num_nodes = A_float.shape[0]

    deg = A_float.sum(axis=1)

    if alpha == 0:
        D = torch.diag(deg)
        return D - A_float

    # Check for zero degrees, ensuring it's on the same device as deg
    if torch.any(deg == 0):
        raise ValueError("Graph has disconnected components (nodes with degree 0) which can cause issues with alpha != 0.")

    # Convert degrees to diagonal matrices
    D_alpha = torch.diag(torch.pow(deg, alpha))
    D_neg_alpha = torch.diag(torch.pow(deg, -alpha))

    # For deg_matrix_ratio, we need broadcasting or explicit expansion
    # Method 1: Using unsqueeze and broadcasting for deg.reshape(1, -1) / deg.reshape(-1, 1)
    deg_row = deg.unsqueeze(0) # shape (1, num_nodes)
    deg_col = deg.unsqueeze(1) # shape (num_nodes, 1)
    deg_matrix_ratio = deg_row / deg_col # This performs element-wise division via broadcasting

    # Element-wise power and multiplication with A
    deg_ratio_pow = torch.pow(deg_matrix_ratio, alpha) * A_float
    Xi_alpha = torch.diag(deg_ratio_pow.sum(axis=1))

    # Matrix multiplications
    L_alpha = Xi_alpha - D_neg_alpha @ A_float @ D_alpha
    return L_alpha


def get_hub_laplacian_dense_pytorch(A: torch.Tensor) -> Callable[[float], torch.Tensor]:

    def hub_laplacian_for_A(alpha: float) -> torch.Tensor:
        return hub_laplacian_dense_pytorch(A, alpha)

    return hub_laplacian_for_A


def hub_advection_diffusion_laplacian_dense_pytorch(
        A: torch.Tensor,
        alpha: float,
        gamma_diff: float,
        gamma_adv: float
) -> torch.Tensor:
    """
    Computes the hub advection-diffusion Laplacian directly using
    hub_laplacian_dense_pytorch, returning a PyTorch tensor.

    Args:
        A (torch.Tensor): The dense adjacency matrix (N x N) as a PyTorch tensor.
        alpha (float): The alpha parameter for the Hub Laplacian.
        gamma_diff (float): The diffusion coefficient.
        gamma_adv (float): The advection coefficient.

    Returns:
        torch.Tensor: The computed advection-diffusion Laplacian as a dense PyTorch tensor.
    """
    # Ensure A is a float tensor
    A_float = A.to(torch.float32)

    # Compute the Hub Laplacian for the advection term
    # This calls hub_laplacian_dense_pytorch with the specified alpha
    L_adv = hub_laplacian_dense_pytorch(A_float, alpha)

    # Compute the Hub Laplacian for the diffusion term
    # This calls hub_laplacian_dense_pytorch with alpha=0 (which computes D-A)
    L_diff = hub_laplacian_dense_pytorch(A_float, alpha=0)

    # Combine the terms as per the formula: gamma_adv * L_adv.T + gamma_diff * L_diff
    # All operations are now PyTorch tensor operations
    result_laplacian = gamma_adv * L_adv.T + gamma_diff * L_diff

    return result_laplacian


def get_hub_advection_diffusion_laplacian_dense_pytorch(A: torch.Tensor) -> Callable[[float, float, float], torch.Tensor]:

    def hub_advection_diffusion_laplacian_for_A(alpha: float, gamma_diff: float, gamma_adv: float) -> torch.Tensor:
        return hub_advection_diffusion_laplacian_dense_pytorch(A, alpha,  gamma_diff, gamma_adv)

    return hub_advection_diffusion_laplacian_for_A

