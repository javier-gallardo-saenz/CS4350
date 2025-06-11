import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from gcnn import GCNNalpha

def get_data_loaders(batch_size=64):
    dataset = QM9(root='data/QM9')
    dataset = dataset.shuffle()

    split1 = int(0.8 * len(dataset))
    split2 = int(0.9 * len(dataset))
    train_dataset = dataset[:split1]
    val_dataset = dataset[split1:split2]
    test_dataset = dataset[split2:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def build_model(params):
    model = GCNNalpha(
        dims=params["dims"],
        degrees=params["degrees"],
        activations=params["act_fns"],
        gso_generator=params["gso_generator"],
        readout_dims=params.get("readout_dims", [128, 1])
    )
    return model

def train_epoch(model, data_loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0

    for data in tqdm(data_loader):
        optimizer.zero_grad()

        x = data.x.to(device)
        batch = data.batch.to(device)
        edge_index = data.edge_index.to(device)
        y = data.y.to(device)

        out = model(x, batch, edge_index)
        loss = loss_fn(out, y[:, 0])  # Assuming prop_idx = 0

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)

def eval_epoch(model, data_loader, loss_fn, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for data in tqdm(data_loader):
            x = data.x.to(device)
            batch = data.batch.to(device)
            edge_index = data.edge_index.to(device)
            y = data.y.to(device)

            out = model(x, batch, edge_index)
            loss = loss_fn(out, y[:, 0])

            epoch_loss += loss.item()

    return epoch_loss / len(data_loader)

def train(params, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = get_data_loaders(batch_size)

    model = build_model(params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5, verbose=True)
    loss_fn = nn.L1Loss()

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in tqdm(range(params["num_epochs"])):

        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)

        scheduler.step(val_loss)

        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    # Load best model
    model.load_state_dict(best_model_state)

    test_loss = eval_epoch(model, test_loader, loss_fn, device)
    print(f'Final Test Loss: {test_loss:.4f}')

    return model, best_val_loss, test_loss
