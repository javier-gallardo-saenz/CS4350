import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from tqdm import tqdm
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))

from utils.preprocessing import process_single_graph_for_dataset_evaluation
from utils.operator_preparation import get_operator_and_params

# Dataset and evaluation parameters
N = 1000
k = 25
gamma_adv = 1
gamma_diff = 1
operators = ['Laplacian', 'Hub_Laplacian', 'Hub_Laplacian', 'Hub_Laplacian', 'Hub_Laplacian',
             'Hub_Advection_Diffusion', 'Hub_Advection_Diffusion', 'Hub_Advection_Diffusion', 'Hub_Advection_Diffusion']
alpha = [0.0, -1.0, -0.5, 0.5, 1.0, -1.0, -0.5, 0.5, 1.0]

DATA_DIR = './data/QM9'
os.makedirs(DATA_DIR, exist_ok=True)

dataset = QM9(root=DATA_DIR)
dataset = dataset.shuffle()[:N]

num_operators = len(operators)
num_graphs = len(dataset)

all_F_matrices_per_graph = [[] for _ in range(num_graphs)]
all_eig_vals_per_graph = [[] for _ in range(num_graphs)]

average_cosine_similarities_agg = {}
all_lambda_max_per_operator = [[] for _ in range(num_operators)]
all_lambda_min_per_operator = [[] for _ in range(num_operators)]
all_sum_abs_gaps_per_operator = [[] for _ in range(num_operators)]

print("\n--- Processing graphs ---")
for graph_idx, g in enumerate(tqdm(dataset, desc="Processing graphs")):
    for op_idx in range(num_operators):
        op_name = operators[op_idx]
        alpha_val = alpha[op_idx]
        operator_fn, params = get_operator_and_params(op_name, alpha_val, gamma_adv, gamma_diff)

        processed_graph_dict, F_dense_matrix, eigenvalues = process_single_graph_for_dataset_evaluation(
            g, k, operator_fn, **params
        )

        all_F_matrices_per_graph[graph_idx].append(F_dense_matrix)
        all_eig_vals_per_graph[graph_idx].append(eigenvalues)

        if eigenvalues.numel() > 0:
            all_lambda_min_per_operator[op_idx].append(eigenvalues[0].item())
            all_lambda_max_per_operator[op_idx].append(eigenvalues[min(k - 1, eigenvalues.numel() - 1)].item())

        if eigenvalues.numel() > 1:
            eig_vals_diffs = torch.diff(eigenvalues[:k])
            sum_abs_gaps = torch.sum(torch.abs(eig_vals_diffs)).item()
            all_sum_abs_gaps_per_operator[op_idx].append(sum_abs_gaps)

print("\n--- Computing Cosine Similarities ---")
for graph_idx in tqdm(range(num_graphs), desc="Computing similarities"):
    current_graph_F_matrices = all_F_matrices_per_graph[graph_idx]

    for i in range(num_operators):
        for j in range(i + 1, num_operators):
            pair_key = (i, j)
            F1 = current_graph_F_matrices[i].flatten().to(torch.float32)
            F2 = current_graph_F_matrices[j].flatten().to(torch.float32)

            if F1.norm() != 0 and F2.norm() != 0:
                similarity = F.cosine_similarity(F1.unsqueeze(0), F2.unsqueeze(0), dim=1).item()
                if not np.isnan(similarity):
                    average_cosine_similarities_agg.setdefault(pair_key, {'sum': 0.0, 'count': 0})
                    average_cosine_similarities_agg[pair_key]['sum'] += similarity
                    average_cosine_similarities_agg[pair_key]['count'] += 1

print("\n--- Average Cosine Similarities ---")
for pair_key, data in average_cosine_similarities_agg.items():
    op1_idx, op2_idx = pair_key
    op1_name = operators[op1_idx]
    op2_name = operators[op2_idx]
    avg_similarity = data['sum'] / data['count']
    print(f"Avg Similarity between {op1_name} (α={alpha[op1_idx]}) and {op2_name} (α={alpha[op2_idx]}): {avg_similarity:.4f}")

# ----------- Plotting Section ----------- #

def get_short_label(op_name, alpha_val):
    name_map = {
        'Laplacian': 'Lap',
        'Hub_Laplacian': 'HubLap',
        'Hub_Advection_Diffusion': 'HubAD'
    }
    short_name = name_map.get(op_name, op_name)
    return f'{short_name} (α={alpha_val})'


def plot_histograms(metric_lists, metric_name, xlabel, operators, alpha_values):
    plt.figure(figsize=(12, 8))
    for op_idx, values in enumerate(metric_lists):
        clean_values = [v for v in values if not np.isnan(v)]
        plt.hist(clean_values, bins=50, alpha=0.5,
         label=get_short_label(operators[op_idx], alpha_values[op_idx]))

    plt.title(f'{metric_name} Distribution Across Operators')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_cosine_histograms(similarity_dict):
    plt.figure(figsize=(16, 8))
    for pair_key, data in similarity_dict.items():
        op1_idx, op2_idx = pair_key
        op1_name = operators[op1_idx]
        op2_name = operators[op2_idx]

        avg_sim = data['sum'] / data['count'] if data['count'] > 0 else np.nan
        label = f"{get_short_label(operators[op1_idx], alpha[op1_idx])} \nvs\n {get_short_label(operators[op2_idx], alpha[op2_idx])}"
        plt.bar(label, avg_sim)


    plt.title('Average Cosine Similarities Between F Matrices')
    plt.ylabel('Cosine Similarity')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# Plot Spectral Range
spectral_ranges = []
for op_idx in range(num_operators):
    ranges = np.array(all_lambda_max_per_operator[op_idx]) - np.array(all_lambda_min_per_operator[op_idx])
    spectral_ranges.append(ranges)

plot_histograms(spectral_ranges, 'Spectral Range', 'Spectral Range', operators, alpha)

# Plot Sum of Absolute Gaps
plot_histograms(all_sum_abs_gaps_per_operator, 'Sum of Absolute Eigenvalue Gaps', 'Sum of Gaps', operators, alpha)

# Plot Cosine Similarities
plot_cosine_histograms(average_cosine_similarities_agg)


