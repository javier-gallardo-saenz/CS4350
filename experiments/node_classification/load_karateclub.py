import torch
from torch_geometric.datasets import KarateClub

data = KarateClub()[0]
num_nodes = data.num_nodes

# Initialize all nodes to False
data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

# Use label 0 and 1 to choose one node each for training
class0 = (data.y == 0).nonzero(as_tuple=True)[0]
class1 = (data.y == 1).nonzero(as_tuple=True)[0]

# Choose first node of each class for training
data.train_mask[class0[0]] = True
data.train_mask[class1[0]] = True

# Optional: use a few more for validation
data.val_mask[class0[1]] = True
data.val_mask[class1[1]] = True

# Remaining nodes go to test
data.test_mask = ~(data.train_mask | data.val_mask)