from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import torch
import random

# Load QM9 dataset
path = './qm9'
dataset = QM9(path)

# Shuffle dataset indices with a fixed seed for reproducibility
num_total = len(dataset)
indices = list(range(num_total))
random.seed(42)
random.shuffle(indices)

# Select 10k train, 1k val, 1k test samples
train_indices = indices[:10_000]
val_indices = indices[10_000:11_000]
test_indices = indices[11_000:12_000]

# Create subsets
train_dataset = dataset[train_indices]
val_dataset = dataset[val_indices]
test_dataset = dataset[test_indices]

#create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# Example: Iterating and accessing the first 7 targets
for batch in train_loader:
    y_first_seven = batch.y[:, :7]  # Shape: (batch_size, 7)
    #print(y_first_seven.shape)
