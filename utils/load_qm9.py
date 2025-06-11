import torch
from torch_geometric.datasets import QM9
from torch.utils.data import DataLoader

def get_data_loaders(batch_size, target_idx=0, seed=42):
    """
    Returns train, validation, and test DataLoaders for QM9.
    """
    dataset = QM9(root='data/QM9')
    # Select one target property
    for data in dataset:
        data.y = data.y[:, target_idx]
    
    torch.manual_seed(seed)
    dataset = dataset.shuffle()
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val   = int(0.1 * n_total)
    train_dataset = dataset[:n_train]
    val_dataset   = dataset[n_train:n_train + n_val]
    test_dataset  = dataset[n_train + n_val:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size)
    return train_loader, val_loader, test_loader
