from torch_geometric.datasets import KarateClub
from torch_geometric.data import Data
import torch

def get_karateclub_data():
    dataset = KarateClub()
    data = dataset[0]  # only one graph in the dataset
    train_mask = data.train_mask # one training node per class

    num_nodes = data.num_nodes
    
    torch.manual_seed(42) 
    unlabeled_mask = ~data.train_mask
    unlabeled_indices = torch.where(unlabeled_mask)[0]

    shuffled_indices_permutation = torch.randperm(len(unlabeled_indices))
    shuffled_unlabeled_indices = unlabeled_indices[shuffled_indices_permutation]

    num_unlabeled = len(shuffled_unlabeled_indices)
    split_point = num_unlabeled // 2  

    validation_indices = shuffled_unlabeled_indices[:split_point]
    test_indices = shuffled_unlabeled_indices[split_point:]

    val_mask = torch.full((num_nodes,), False, dtype=torch.bool)
    test_mask = torch.full((num_nodes,), False, dtype=torch.bool)

    val_mask[validation_indices] = True
    test_mask[test_indices] = True


    # Return the full graph and the masks
    return data, train_mask, val_mask, test_mask
