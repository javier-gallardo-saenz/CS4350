import numpy as np

def hub_laplacian(A: np.ndarray, alpha: float) -> np.ndarray:
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "Adjacency matrix must be square"
    A = A.astype(float)

    # degree vector
    deg = A.sum(axis=1)

    if np.any(deg == 0):
        raise ValueError("Graph has disconnected components (nodes with degree 0)")

    # D^alpha and D^-alpha
    D_alpha = np.diag(np.power(deg, alpha))
    D_neg_alpha = np.diag(np.power(deg, -alpha))

    # Compute Ξ_α: diag( sum_{w in N(v)} (d_w/d_v)^α )
    deg_matrix_ratio = deg.reshape(1, -1) / deg.reshape(-1, 1)  # shape: (N, N)
    deg_ratio_pow = np.power(deg_matrix_ratio, alpha) * A  # keep only neighbors
    Xi_alpha = np.diag(deg_ratio_pow.sum(axis=1))

    L_alpha = Xi_alpha - D_neg_alpha @ A @ D_alpha

    return L_alpha


def hub_advection_diffusion(
    A: np.ndarray, alpha: float, gamma_diff: float, gamma_adv: float
) -> np.ndarray:

    return gamma_adv * hub_laplacian(A, alpha).T + gamma_diff * hub_laplacian(
        A, alpha=0
    )
