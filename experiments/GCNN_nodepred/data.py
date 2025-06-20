from torch_geometric.datasets import KarateClub
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit # Import the transformation
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

def get_karateclub_data_custom(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Loads the Karate Club dataset and creates custom train/val/test masks.

    Args:
        train_ratio (float): Proportion of nodes to use for training.
        val_ratio (float): Proportion of nodes to use for validation.
        test_ratio (float): Proportion of nodes to use for testing.
        seed (int): Random seed for reproducibility.
    Returns:
        data (Data): PyG Data object with updated masks.
        train_mask (Tensor): Boolean mask for training nodes.
        val_mask (Tensor): Boolean mask for validation nodes.
        test_mask (Tensor): Boolean mask for test nodes.
    """
    # Load the original Karate Club dataset
    dataset = KarateClub()
    data = dataset[0] # The KarateClub dataset is a single graph

    # Ensure the sum of ratios is 1 (or close to it due to floating point arithmetic)
    if not torch.isclose(torch.tensor(train_ratio + val_ratio + test_ratio), torch.tensor(1.0)):
        raise ValueError("Train, val, and test ratios must sum to 1.")

    torch.manual_seed(42) 
    # Create the RandomNodeSplit transform
    # We specify the number of validation and test nodes directly.
    # The remaining nodes will be used for training.
    num_nodes = data.num_nodes
    num_val = int(val_ratio * num_nodes)
    num_test = int(test_ratio * num_nodes)

    # Note: RandomNodeSplit can also take num_train_per_class, but for a general random split,
    # just specifying num_val and num_test is usually simpler.
    # We use num_splits=1 to get a single set of masks.
    splitter = RandomNodeSplit(
        num_val=num_val,
        num_test=num_test,
    )

    # Apply the split to the data object
    # This will add data.train_mask, data.val_mask, data.test_mask
    data = splitter(data)

    # For convenience, return the masks explicitly
    return data, data.train_mask, data.val_mask, data.test_mask
