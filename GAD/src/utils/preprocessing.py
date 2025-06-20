import torch
import numpy as np
import scipy.linalg
from torch import Tensor
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
from typing import List, Tuple, Dict, Any, Callable
from tqdm import tqdm


def compute_spectral_features(
    adj: torch.Tensor,
    num_nodes: int,
    k: int,
    laplacian_fn: Callable[..., torch.Tensor], # laplacian_fn is expected to return torch.Tensor
    **kwargs: Any
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute spectral features using a provided Laplacian function.
    Selects the first non-trivial eigenvector, and returns top-k eigenvectors and values.
    Assumes L is a torch.Tensor and its eigenvalues are real and positive,
    but the matrix itself might not be symmetric, requiring torch.linalg.eig.
    """
    # L is a torch.Tensor, as returned by laplacian_fn
    L = laplacian_fn(adj, **kwargs)

    # Store original device and dtype to return tensors in the same format
    original_device = L.device
    original_dtype = L.dtype

    # Move to CPU and use float64 for eigendecomposition for stability and precision.
    # torch.linalg.eig can also run on GPU, but CPU+float64 is generally more robust
    # for numerical stability in eigendecomposition, especially for larger or tricky matrices.
    L_compute = L.to(torch.float64).cpu()

    # Compute eigen-decomposition using PyTorch's general eigenvalue function.
    # This will return complex tensors for both eigenvalues and eigenvectors.
    vals_complex, vecs_complex = torch.linalg.eig(L_compute)

    # Since we are guaranteed real eigenvalues, take the real part and discard the imaginary.
    # vals_complex will be complex64 or complex128.
    vals = torch.real(vals_complex)
    vecs = torch.real(vecs_complex) # Also take the real part of eigenvectors

    # Since eigenvalues are guaranteed positive, sorting by magnitude is equivalent to sorting by value.
    # Sort eigenvalues in ascending order and get the sorting indices.
    vals_sorted, idx = torch.sort(vals)
    vecs_sorted = vecs[:, idx]

    # Remove (near-)zero eigenvalues (trivial modes).
    # Since eigenvalues are guaranteed positive, we can just check if they are greater than tolerance.
    tol = 1e-6
    non_trivial_indices = torch.where(vals_sorted > tol)[0]  # Check for positive values > tol

    if len(non_trivial_indices) == 0:
        print(f"Warning: No non-trivial (positive and > {tol}) eigenvalues found. "
              "Defaulting to the smallest (often trivial) eigenvector.")
        i_low = 0  # Default to the smallest eigenvalue's eigenvector
    else:
        i_low = non_trivial_indices[0]  # Select the index of the first non-trivial (positive) eigenvalue

    # Select the first non-trivial eigenvector
    lowest = vecs_sorted[:, i_low].to(original_dtype).to(original_device)

    # Get top-k eigenvectors and eigenvalues
    # "Top-k" in spectral features usually refers to the smallest non-trivial eigenvalues
    # (corresponding to low-frequency modes). So, taking the first k from the sorted list
    # (which are the smallest) makes sense.
    effective_k = min(k, vals_sorted.shape[0])

    eig_vals = vals_sorted[:effective_k].to(original_dtype).to(original_device)
    eig_vecs = vecs_sorted[:, :effective_k].to(original_dtype).to(original_device)

    # Pad if needed (using PyTorch operations)
    if eig_vecs.shape[1] < k:
        pad_dim = k - eig_vecs.shape[1]
        # Create zero tensors directly with PyTorch, and place them on the original device
        eig_vecs = torch.cat([eig_vecs, torch.zeros(num_nodes, pad_dim, dtype=original_dtype, device=original_device)], dim=1)
        eig_vals = torch.cat([eig_vals, torch.zeros(pad_dim, dtype=original_dtype, device=original_device)], dim=0)

    return lowest, eig_vecs, eig_vals



def compute_flowmat(adj: Tensor, lowest: Tensor) -> Tuple[Tensor, Tensor]:
    """
    From dense adj and lowest eig vector, build
    directed, normalized flow F_norm_edge & its degree F_dig
    """
    # Initialize F with the same shape as adj, and desired float32 dtype
    F = torch.zeros_like(adj, dtype=torch.float32)
    n = adj.shape[0]

    # construct gradient flow vector field
    # Iterate through indices. This loop is inherently sequential.
    # For very large graphs, consider if this can be vectorized.
    for i in range(n):
        for j in range(n):
            # adj[i, j] directly works with torch.Tensor
            if adj[i, j].item() == 1: # .item() is good here to compare a single tensor element
                diff = lowest[i] - lowest[j]
                # No need for .item() here if diff is already a scalar tensor
                # You can directly assign tensor to tensor element
                F[i, j] = diff if diff != 0 else torch.tensor(1e-8, dtype=torch.float32)

    # Normalize rows using torch.linalg.norm
    # Use torch.linalg.norm for PyTorch tensors
    row_sums = torch.linalg.norm(F, ord=1, dim=1, keepdim=True) + 1e-20
    F_norm = F / row_sums  # matrix: strength of node i flow to node j

    # Use .sum(dim=0) for PyTorch tensor and avoid unnecessary from_numpy
    F_dig = F_norm.sum(dim=0)  # vec: total flow that each node receives from all other nodes

    # dense_to_sparse expects a torch.Tensor, no need for from_numpy
    F_norm_edge = dense_to_sparse(F_norm)[1]

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
    for g in tqdm(dataset, desc="Processing eigenmaps of the dataset graphs"):
        d = process_single_graph(
            g, num_of_eigenvectors, laplacian_fn, **laplacian_kwargs
        )
        out.append(Data.from_dict(d))
    return out


"""
processed_dataset = preprocessing_dataset(
    my_dataset,
    num_of_eigenvectors=k,
    laplacian_fn=hub_laplacian,
    alpha=0.8 # Only 'alpha' is needed for hub_laplacian
)
"""


# Calculate the average node degree in the training data

def average_node_degree(dataset):
    D = []

    for i in tqdm(range(len(dataset)), desc="Processing average_node_degree of the dataset"):
        adj = to_dense_adj(dataset[i].edge_index)[0]

        deg = adj.sum(axis=1, keepdim=True)  # Degree of nodes, shape [N, 1]

        D.append(deg.squeeze())

    D = torch.cat(D, dim=0)

    avg_d = dict(lin=torch.mean(D),
                 exp=torch.mean(torch.exp(torch.div(1, D)) - 1),
                 log=torch.mean(torch.log(D + 1)))

    return D, avg_d