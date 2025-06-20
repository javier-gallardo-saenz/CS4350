import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

def plot_similarity_heatmap(sim_matrix, title="Cosine Similarity Matrix"):
    """
    Plots a heatmap of a cosine similarity matrix.

    Args:
        sim_matrix (Tensor): Tensor [N x N] representing cosine similarities.
        title (str): Title for the heatmap.
    """
    sim_np = sim_matrix.detach().cpu().numpy()
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(sim_np, cmap="viridis", square=True, cbar=True)
    plt.title(title)
    plt.xlabel("Node Index")
    plt.ylabel("Node Index")
    plt.tight_layout()
    plt.show()

def plot_avg_similarity(avg_cos_sims, title=None):
    """
    Plots average cosine similarity per layer.

    Args:
        avg_cos_sims (List[float]): Average cosine similarity per layer.
    """
    layers = list(range(len(avg_cos_sims)))
    plt.figure(figsize=(8, 5))
    plt.plot(layers, avg_cos_sims, marker='o', linestyle='-')
    plt.xlabel("Layer")
    plt.ylabel("Average Cosine Similarity")
    if title:
        plt.title(title)
    else:
        plt.title("Average Pairwise Cosine Similarity per Layer")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def eval_cos_sim_per_layer(embeddings_per_layer, labels):
    """
    Computes the average pairwise cosine similarity for each layer of node embeddings.

    Args:
        embeddings_per_layer (List[Tensor]): List of tensors [N x D] from each GNN layer.
        labels (Tensor, optional): Tensor [N] with node labels (either predicted or true).

    Returns:
        List[Tensor]: List of cosine similarity matrices for each layer.
        List[float]: Average overall cosine similarity per layer.
        List[float]: Average intra-class cosine similarity per layer.
        List[float]: Average inter-class cosine similarity per layer.

    """
    avg_cos_sims = []
    sim_matrices = []
    avg_inter_class_sims = []
    avg_intra_class_sims = []

    for emb in embeddings_per_layer:

        # Normalize embeddings
        emb = F.normalize(emb, p=2, dim=1)  # [N x D]

        # Compute cosine similarity matrix: sim(i,j) = emb[i] · emb[j]
        sim_matrix = emb @ emb.T  # [N x N]

        # Get upper triangle (excluding diagonal) to avoid redundancy
        num_nodes = emb.size(0)
        if num_nodes < 2:
            avg_cos_sims.append(float('nan'))
            continue

        triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
        pairwise_sims = sim_matrix[triu_indices[0], triu_indices[1]]

        avg_cos = pairwise_sims.mean().item()
        avg_cos_sims.append(avg_cos)
        sim_matrices.append(sim_matrix)

        # Masks for class-based comparisons
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # Shape: [N, N]
        triu_mask = torch.triu(torch.ones(num_nodes, num_nodes, dtype=torch.bool), diagonal=1)
        intra_mask = label_equal & ~torch.eye(num_nodes, dtype=torch.bool) & triu_mask
        inter_mask = ~label_equal & triu_mask

        intra_similarity = sim_matrix[intra_mask].mean().item()
        inter_similarity = sim_matrix[inter_mask].mean().item()
        avg_inter_class_sims.append(inter_similarity)
        avg_intra_class_sims.append(intra_similarity)

    return sim_matrices, avg_cos_sims, avg_intra_class_sims, avg_inter_class_sims
