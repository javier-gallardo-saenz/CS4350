import torch
import numpy as np
import scipy.linalg
from torch import Tensor
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
from typing import List, Tuple, Dict, Any, Callable

def compute_spectral_features(
    adj: torch.Tensor,
    num_nodes: int,
    k: int,
    laplacian_fn: Callable[..., np.ndarray],
    **kwargs: Any
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute spectral features using a provided Laplacian function.
    Selects the first non-trivial eigenvector, and returns top-k eigenvectors and values.
    """
    A = adj.cpu().numpy()
    L = laplacian_fn(A, **kwargs)

    # compute eigen-decomposition
    vals, vecs = scipy.linalg.eig(L)
    vals = np.real(vals)
    vecs = np.real(vecs)

    # sort by eigenvalue magnitude
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]

    # remove (near-)zero eigenvalues (trivial modes)
    tol = 1e-6
    non_trivial_indices = np.where(np.abs(vals) > tol)[0]

    if len(non_trivial_indices) == 0:
        print("Warning: No non-trivial eigenvalues found.")
        i_low = 0
    else:
        i_low = non_trivial_indices[0]

    # select the first non-trivial eigenvector
    lowest = torch.from_numpy(vecs[:, i_low]).float()

    # get top-k eigenvectors and eigenvalues 
    eig_vals = vals[:k]
    eig_vecs = vecs[:, :k]

    eig_vals = torch.from_numpy(eig_vals).float()
    eig_vecs = torch.from_numpy(eig_vecs).float()

    # pad if needed
    if eig_vecs.shape[1] < k:
        pad_dim = k - eig_vecs.shape[1]
        eig_vecs = torch.cat([eig_vecs, torch.zeros(num_nodes, pad_dim)], dim=1)
        eig_vals = torch.cat([eig_vals, torch.zeros(pad_dim)], dim=0)

    return lowest, eig_vecs, eig_vals



def compute_flowmat(adj: Tensor, lowest: Tensor) -> Tuple[Tensor, Tensor]:
    """
    From dense adj and lowest eig vector, build
    directed, normalized flow F_norm_edge & its degree F_dig
    """
    A = adj.cpu().numpy()
    F = np.zeros_like(A, dtype=np.float32)
    n = A.shape[0]

    # construct gradient flow vector field
    for i in range(n):
        for j in range(n):
            if A[i, j] == 1:
                diff = lowest[i] - lowest[j]
                F[i, j] = diff.item() if diff.item() != 0 else 1e-8

    # normalize rows
    row_sums = np.linalg.norm(F, ord=1, axis=1, keepdims=True) + 1e-20
    F_norm = F / row_sums  # matrix: strenth of node i flow to node j

    F_dig = torch.from_numpy(
        F_norm.sum(axis=0)
    )  # vec: total flow that each node receives from all other nodes
    F_norm_edge = dense_to_sparse(torch.from_numpy(F_norm))[1]

    return F_norm_edge, F_dig


def process_single_graph(
    data: Data,
    k: int,
    laplacian_fn: Callable[..., np.ndarray],  # New argument
    **laplacian_kwargs: Any  # New argument for Laplacian function parameters
) -> Dict[str, Any]:
    """
    All processing for one torch_geometric.data.Data graph → dict
    """
    d = data.to_dict()
    n = data.num_nodes

    d["norm_n"] = torch.full((n, 1), 1.0 / n).sqrt()
    adj = to_dense_adj(data.edge_index)[0]

    low, eig_vecs, eig_vals = compute_spectral_features(
        adj, n, k, laplacian_fn, **laplacian_kwargs  # Pass the function and its kwargs
    )
    d.update(k_eig_vec=eig_vecs, k_eig_val=eig_vals)

    F_edge, F_deg = compute_flowmat(adj, low)
    d.update(F_norm_edge=F_edge, F_dig=F_deg)

    return d


def preprocessing_dataset(
    dataset: List[Data],
    num_of_eigenvectors: int,
    laplacian_fn: Callable[..., np.ndarray],  # New argument
    **laplacian_kwargs: Any  # New argument
) -> List[Data]:
    """
    Wrapper: apply to each graph in the input list‐like dataset
    → returns list of torch_geometric.data.Data
    """
    out: List[Data] = []
    for g in dataset:
        d = process_single_graph(
            g, num_of_eigenvectors, laplacian_fn, **laplacian_kwargs
        )
        out.append(Data.from_dict(d))
    return out

