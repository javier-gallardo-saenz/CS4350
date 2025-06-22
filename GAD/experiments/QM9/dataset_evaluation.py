import torch
import torch.optim as opt
import torch.nn as nn

from torch_geometric.datasets import QM9
from torch_geometric.data import DataLoader
import os
import sys

from torch_geometric.data import Data
from typing import List, Tuple, Dict, Any, Callable
from tqdm import tqdm

import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))

from utils.preprocessing import process_single_graph_for_dataset_evaluation
from utils.operator_preparation import get_operator_and_params


#PARAMETERS FOR THE DATASET EVALUATION
N = 1000 #number of graphs to evaluate
gamma_adv = 1
gamma_diff = 1
operators = ['Laplacian', 'Hub_Laplacian', 'Hub_Laplacian', 'Hub_Laplacian', 'Hub_Laplacian',
             'Hub_Advection_Diffusion', 'Hub_Advection_Diffusion', 'Hub_Advection_Diffusion', 'Hub_Advection_Diffusion']
alpha = [0.0, -1.0, -0.5, 0.5, 1.0, -1.0, -0.5, 0.5, 1.0]



DATA_DIR = './data/QM9'

# --- Create the directory if it doesn't exist ---
os.makedirs(DATA_DIR, exist_ok=True)

# check if QM9 already exists locally to avoid redownloading
processed_dir = os.path.join(DATA_DIR, 'processed')
if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
    print(f"Loading QM9 dataset from local storage: {DATA_DIR}")
    # force_reload=False ensures it doesn't redownload/reprocess if already present
    dataset = QM9(root=DATA_DIR, force_reload=False)
else:
    print(f"QM9 dataset not found locally in {DATA_DIR}. Downloading and processing...")
    # This will trigger download and initial processing
    dataset = QM9(root=DATA_DIR, force_reload=True)  # force_reload=True is default for first time
    print("Download and initial processing complete.")

dataset = dataset.shuffle()
dataset = dataset[:N]

if len(operators) != len(alpha):
    raise ValueError("Lengths of operators array and gamma array must match")

num_operators = len(operators)
num_graphs = len(dataset)

# Outer list: per graph. Inner list: per operator.
# all_F_matrices_per_graph[graph_idx][operator_idx] will have (num_nodes, num_nodes) shape
# all_eig_vals_per_graph[graph_idx][operator_idx] will have (num_nodes,) shape
all_F_matrices_per_graph: List[List[torch.Tensor]] = [[] for _ in range(num_graphs)]
all_eig_vals_per_graph: List[List[torch.Tensor]] = [[] for _ in range(num_graphs)]

# New aggregators for average cosine similarities
# Key: (op_idx1, op_idx2) tuple, Value: {'sum': float, 'count': int}
average_cosine_similarities_agg = {}

# New aggregators for eigenvalue metrics
# List of lists, each inner list stores values for one operator across all graphs
all_lambda_max_per_operator: List[List[float]] = [[] for _ in range(num_operators)]
all_lambda_min_per_operator: List[List[float]] = [[] for _ in range(num_operators)]
all_sum_abs_gaps_per_operator: List[List[float]] = [[] for _ in range(num_operators)]

print("\n--- Processing all graphs for all operators (single pass) ---")
for graph_idx, g in enumerate(tqdm(dataset, desc="Processing graphs")):
    # Determine the number of eigenvalues/eigenvectors for the current graph dynamically
    current_k_for_graph = g.num_nodes

    # Process each operator for the current graph
    for op_idx in range(num_operators):
        op_name = operators[op_idx]
        alpha_val = alpha[op_idx]  # alpha is still fixed per operator type

        # Pass the dynamic k_for_graph (g.num_nodes) to the operator function resolver
        operator_fn, params = get_operator_and_params(op_name, alpha_val, gamma_adv, gamma_diff)

        processed_graph_dict, F_dense_matrix, eigenvalues = process_single_graph_for_dataset_evaluation(
            g, current_k_for_graph, operator_fn, **params
        )

        # Store results for the current graph and operator
        # eigenvalues tensor will now have g.num_nodes elements
        all_F_matrices_per_graph[graph_idx].append(F_dense_matrix)
        all_eig_vals_per_graph[graph_idx].append(eigenvalues)

        # Collect eigenvalue metrics for the current graph and operator
        # The number of actual eigenvalues is simply the length of the tensor
        num_actual_eig_vals = eigenvalues.numel()

        if num_actual_eig_vals > 0:
            # Lambda min is the first eigenvalue
            all_lambda_min_per_operator[op_idx].append(eigenvalues[0].item())
            # Lambda max is the last eigenvalue
            all_lambda_max_per_operator[op_idx].append(eigenvalues[-1].item())
        else:
            all_lambda_min_per_operator[op_idx].append(np.nan)
            all_lambda_max_per_operator[op_idx].append(np.nan)

        if num_actual_eig_vals > 1:
            # Calculate sum of absolute differences between consecutive eigenvalues
            # No slicing needed as `eigenvalues` already contains only the actual values
            eig_vals_diffs_meaningful = torch.diff(eigenvalues)
            sum_abs_gaps = torch.sum(torch.abs(eig_vals_diffs_meaningful)).item()
            all_sum_abs_gaps_per_operator[op_idx].append(sum_abs_gaps)
        else:
            all_sum_abs_gaps_per_operator[op_idx].append(np.nan)  # No gaps to compute

print("\n--- Starting Comparisons and Aggregations ---")

# a) Calculate average Cosine similarity between every possible pair of F matrices across all graphs
print("\n--- Average Cosine Similarities between F Matrices Across Dataset ---")
for graph_idx in tqdm(range(num_graphs), desc="Aggregating F matrix similarities"):
    current_graph_F_matrices = all_F_matrices_per_graph[graph_idx]

    # Compute cosine similarity for all unique pairs of operators
    for i in range(num_operators):
        for j in range(i + 1, num_operators):  # Only unique pairs, avoid self-comparison and duplicates
            pair_key = tuple(sorted((i, j)))  # Use a sorted tuple as key for consistent access

            F1 = current_graph_F_matrices[i].flatten()
            F2 = current_graph_F_matrices[j].flatten()

            # Ensure tensors are float and handle potential all-zero vectors
            F1 = F1.to(torch.float32)
            F2 = F2.to(torch.float32)

            similarity = np.nan  # Default to NaN if calculation fails or vectors are zero

            # Check for zero vectors to avoid NaN in cosine similarity
            if F1.norm() != 0 and F2.norm() != 0:
                # F.cosine_similarity expects 2D inputs, so unsqueeze
                similarity = F.cosine_similarity(F1.unsqueeze(0), F2.unsqueeze(0), dim=1).item()

            # Aggregate if similarity is not NaN
            if not np.isnan(similarity):
                average_cosine_similarities_agg.setdefault(pair_key, {'sum': 0.0, 'count': 0})
                average_cosine_similarities_agg[pair_key]['sum'] += similarity
                average_cosine_similarities_agg[pair_key]['count'] += 1

# Print average cosine similarities
print("\nAverage Pairwise F Matrix Cosine Similarities:")
for pair_key, data in average_cosine_similarities_agg.items():
    op1_idx, op2_idx = pair_key
    op1_name = operators[op1_idx]
    op1_alpha = alpha[op1_idx]
    op2_name = operators[op2_idx]
    op2_alpha = alpha[op2_idx]

    if data['count'] > 0:
        avg_similarity = data['sum'] / data['count']
        print(
            f"  Avg Similarity between {op1_name} (alpha={op1_alpha}) and {op2_name} (alpha={op2_alpha}): {avg_similarity:.4f} (over {data['count']} graphs)")
    else:
        print(
            f"  Avg Similarity between {op1_name} (alpha={op1_alpha}) and {op2_name} (alpha={op2_alpha}): No valid similarities to compute (zero vectors or no data)")

# b) Average Eigenvalue Differences for each operator (now fully dynamic and correctly averaged)
print("\n--- Average Eigenvalue Differences ---")

# Now, process and print the average differences for each operator
for op_idx in range(num_operators):
    op_name = operators[op_idx]
    op_alpha = alpha[op_idx]
    print(f"\nOperator: {op_name} (alpha={op_alpha})")

    current_op_all_graph_diffs = []
    for graph_idx in range(num_graphs):
        eig_vals = all_eig_vals_per_graph[graph_idx][op_idx]

        # num_actual_eig_vals is simply the size of the returned tensor
        num_actual_eig_vals = eig_vals.numel()

        if num_actual_eig_vals > 1:
            # Compute differences over all actual eigenvalues for this graph
            eig_vals_diffs = torch.diff(eig_vals).cpu().numpy()
            current_op_all_graph_diffs.append(eig_vals_diffs)
        # If not enough actual eigenvalues, we don't append anything,
        # which correctly excludes this graph from the average for this operator's gaps.

    if current_op_all_graph_diffs:
        # Determine the maximum length of differences array across graphs for padding
        # This will be the largest number of gaps observed in any graph for this operator
        max_len = max((len(d) for d in current_op_all_graph_diffs), default=0)

        if max_len > 0:
            # Pad shorter arrays with NaN to enable stacking and `nanmean`
            padded_diffs = [np.pad(d, (0, max_len - len(d)), 'constant', constant_values=np.nan) for d in
                            current_op_all_graph_diffs]

            stacked_diffs = np.array(padded_diffs)

            # Calculate average differences across all graphs, ignoring NaNs from padding
            mean_diffs = np.nanmean(stacked_diffs, axis=0)

            print(f"  Average eigenvalue differences for first {max_len} gaps across graphs:")
            for diff_idx, avg_diff in enumerate(mean_diffs):
                print(f"    Gap {diff_idx + 1} (λ_{diff_idx + 2} - λ_{diff_idx + 1}): {avg_diff:.6f}")
        else:
            print("  No eigenvalue differences to compute (eigenvalues list too short for any graph).")
    else:
        print("  No eigenvalue differences were collected for this operator (no graphs with enough eigenvalues).")

# c) New Metrics: Average Spectral Range and Average Sum of Absolute Gaps (now fully dynamic)
print("\n--- Additional Eigenvalue Metrics ---")
for op_idx in range(num_operators):
    op_name = operators[op_idx]
    op_alpha = alpha[op_idx]
    print(f"\nOperator: {op_name} (alpha={op_alpha})")

    # np.nanmean will correctly average only the non-NaN values
    avg_lambda_min = np.nanmean(all_lambda_min_per_operator[op_idx])
    avg_lambda_max = np.nanmean(all_lambda_max_per_operator[op_idx])
    avg_sum_abs_gaps = np.nanmean(all_sum_abs_gaps_per_operator[op_idx])

    if not np.isnan(avg_lambda_min) and not np.isnan(avg_lambda_max):
        avg_spectral_range = avg_lambda_max - avg_lambda_min
        print(f"  Average λ_min: {avg_lambda_min:.6f}")
        print(f"  Average λ_max: {avg_lambda_max:.6f}")
        print(f"  Average Spectral Range (Avg λ_max - Avg λ_min): {avg_spectral_range:.6f}")
    else:
        print("  Spectral range could not be computed (not enough valid λ_min/λ_max values).")

    if not np.isnan(avg_sum_abs_gaps):
        print(f"  Average Sum of Absolute Gaps (over relevant gaps): {avg_sum_abs_gaps:.6f}")
    else:
        print("  Average sum of absolute gaps could not be computed (not enough valid gap values).")



