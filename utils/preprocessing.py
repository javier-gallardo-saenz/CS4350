import torch
import numpy as np
import scipy.linalg
from torch import Tensor
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
from typing import List, Tuple, Dict, Any
from operators import hub_advection_diffusion

def compute_spectral_features(
    adj: Tensor,
    num_nodes: int,
    k: int,
    alpha: float,
    gamma_diff: float,
    gamma_adv: float) -> Tuple[Tensor, Tensor, Tensor]:
    """
    1) Build hub Laplacian L
    2) Eigendecompose & sort
    3) Extract 2nd-smallest eigenvector + top-k, with padding
    """
    A = adj.cpu().numpy()
    L = hub_advection_diffusion(
        A, alpha=alpha, gamma_diff=gamma_diff, gamma_adv=gamma_adv
    )
    # compute and sort eigenvals and eigenvs
    vals, vecs = scipy.linalg.eig(L)
    vals[np.abs(vals) < 1e-8] = 0
    vecs[np.abs(vals) < 1e-8] = 0
    idx = np.argsort(vals)
    vals, vecs = vals[idx], vecs[:, idx]

    # pick lowest non trivial eigenvec
    i_low = 1 if len(vals) > 1 else 0
    lowest = torch.from_numpy(vecs[:, i_low]).float()

    # convert to torch and pad if nodes < # eigenvecs
    eig_vecs = torch.from_numpy(vecs[:, :k]).float()
    eig_vals = torch.from_numpy(vals[:k]).float()
    if num_nodes < k:
        pad_v = torch.zeros(num_nodes, k - num_nodes).clamp(min=1e-8)
        pad_l = torch.zeros(k - num_nodes).clamp(min=1e-8)
        eig_vecs = torch.cat([eig_vecs, pad_v], dim=1)
        eig_vals = torch.cat([eig_vals, pad_l], dim=0)

    # return lowest, eigenvectors and eigenvalues
    return lowest, eig_vecs, eig_vals


def compute_flowmat(adj: Tensor, lowest: Tensor) -> Tuple[Tensor, Tensor]:
    """
    From dense adj and lowest eig vector, build
    directed, normalized flow F_norm_edge & its degree F_dig
    """
    A = adj.cpu().numpy()
    F = np.zeros_like(A, dtype=np.float32)
    n = A.shape[0]

    #construct gradient flow vector field
    for i in range(n):
        for j in range(n):
            if A[i, j] == 1:
                diff = lowest[i] - lowest[j]
                F[i, j] = diff.item() if diff.item() != 0 else 1e-8

    # normalize rows
    row_sums = np.linalg.norm(F, ord=1, axis=1, keepdims=True) + 1e-20
    F_norm = F / row_sums #matrix: strenth of node i flow to node j

    F_dig = torch.from_numpy(F_norm.sum(axis=0)) # vec: total flow that each node receives from all other nodes
    F_norm_edge = dense_to_sparse(torch.from_numpy(F_norm))[1] 

    return F_norm_edge, F_dig


def process_single_graph(
    data: Data, k: int, alpha: float, gamma_diff: float, gamma_adv: float
) -> Dict[str, Any]:
    """
    All processing for one torch_geometric.data.Data graph → dict
    """
    d = data.to_dict()
    n = data.num_nodes

    d["norm_n"] = torch.full((n, 1), 1.0 / n).sqrt()
    adj = to_dense_adj(data.edge_index)[0]

    low, eig_vecs, eig_vals = compute_spectral_features(
        adj, n, k, alpha, gamma_diff, gamma_adv
    )
    d.update(k_eig_vec=eig_vecs, k_eig_val=eig_vals)

    F_edge, F_deg = compute_flowmat(adj, low)
    d.update(F_norm_edge=F_edge, F_dig=F_deg)

    return d

def preprocessing_dataset(
    dataset: List[Data],
    num_of_eigenvectors: int,
    alpha: float = 1.0,
    gamma_diff: float = 0.5,
    gamma_adv: float = 0.5,
) -> List[Data]:
    """
    Wrapper: apply to each graph in the input list‐like dataset
    → returns list of torch_geometric.data.Data
    """
    out: List[Data] = []
    for g in dataset:
        d = process_single_graph(g, num_of_eigenvectors, alpha, gamma_diff, gamma_adv)
        out.append(Data.from_dict(d))
    return out
