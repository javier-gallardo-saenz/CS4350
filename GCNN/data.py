# data.py
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

# Modify function signature to accept 'experiment_targets' instead of a default
def get_data_loaders(N: int, targets: list,batch_size=64, split: tuple=(0.7, 0.15, 0.15), root='data/QM9'): 
    dataset = QM9(root=root)
    dataset = dataset[:N]
    dataset = dataset.shuffle()

    
    dataset.data.y = dataset.data.y[:, targets] 
    f_train, _, f_test = split
    n1, n2 = int(N* f_train), int(N*(1-f_test))

    train, val, test = dataset[:n1], dataset[n1:n2], dataset[n2:]
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val,   batch_size=batch_size, shuffle=False),
        DataLoader(test,  batch_size=batch_size, shuffle=False),
    )