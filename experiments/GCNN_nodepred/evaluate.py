import torch
import torch.nn.functional as F

def eval_cos_sim_per_layer(embeddings_per_layer, mask=None):
    """
    Computes the average pairwise cosine similarity for each layer of node embeddings.

    Args:
        embeddings_per_layer (List[Tensor]): List of tensors [N x D] from each GNN layer.
        mask (Tensor, optional): Boolean tensor [N] indicating which nodes to include 
                                 (e.g., train/test/val mask). If None, all nodes are used.

    Returns:
        List[float]: Average cosine similarity per layer.
    """
    avg_cos_sims = []

    for emb in embeddings_per_layer:
        if mask is not None:
            emb = emb[mask]  # Select only the masked nodes

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

    return sim_matrix, avg_cos_sims 
