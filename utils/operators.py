import numpy as np
from typing import Callable

def hub_laplacian(A: np.ndarray, alpha: float) -> np.ndarray:
    """
    Directly computes the Hub Laplacian for a given adjacency matrix A and alpha.
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "Adjacency matrix must be square"
    A_float = A.astype(float)

    deg = A_float.sum(axis=1)

    if alpha == 0:
        return np.diag(deg) - A_float

    if np.any(deg == 0):
        raise ValueError("Graph has disconnected components (nodes with degree 0) which can cause issues with alpha != 0.")

    D_alpha = np.diag(np.power(deg, alpha))
    D_neg_alpha = np.diag(np.power(deg, -alpha))

    deg_matrix_ratio = deg.reshape(1, -1) / deg.reshape(-1, 1)
    deg_ratio_pow = np.power(deg_matrix_ratio, alpha) * A_float
    Xi_alpha = np.diag(deg_ratio_pow.sum(axis=1))

    L_alpha = Xi_alpha - D_neg_alpha @ A_float @ D_alpha
    return L_alpha

def get_hub_laplacian(A: np.ndarray) -> Callable[[float], np.ndarray]:
    """
    Returns a specialized get_hub_laplacian function for a given adjacency matrix A.
    The returned function takes only the 'alpha' parameter.
    """
    def hub_laplacian_for_A(alpha: float) -> np.ndarray:
        return hub_laplacian(A, alpha) # Calls the direct function
    return hub_laplacian_for_A

def hub_advection_diffusion_laplacian(
    A: np.ndarray,
    alpha: float,
    gamma_diff: float,
    gamma_adv: float
) -> np.ndarray:
    """
    Directly computes the hub advection-diffusion Laplacian for given A and parameters.
    """
    return gamma_adv * get_hub_laplacian(A, alpha).T + gamma_diff * get_hub_laplacian(A, alpha=0)

def get_hub_advection_diffusion(A: np.ndarray) -> Callable[[float, float, float], np.ndarray]:
    """
    Returns a specialized get_hub_advection_diffusion_laplacian function for a given A.
    The returned function takes alpha, gamma_diff, and gamma_adv.
    """
    def hub_advection_diffusion_for_A(
        alpha: float,
        gamma_diff: float,
        gamma_adv: float
    ) -> np.ndarray:
        return hub_advection_diffusion_laplacian(A, alpha, gamma_diff, gamma_adv)
    return hub_advection_diffusion_for_A