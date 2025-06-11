# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from gcnn import GCNNalpha
from data import get_data_loaders

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total, count = 0.0, 0
    for batch in tqdm(loader, desc='Train'):
        batch = batch.to(device)
        optimizer.zero_grad()
        out   = model(batch.x, batch.batch, batch.edge_index)
        loss  = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total += loss.item() * batch.num_graphs
        count += batch.num_graphs
    return total / count

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device, tag='Val'):
    model.eval()
    total, count = 0.0, 0
    for batch in tqdm(loader, desc=tag):
        batch = batch.to(device)
        out   = model(batch.x, batch.batch, batch.edge_index)
        loss  = loss_fn(out, batch.y)
        total += loss.item() * batch.num_graphs
        count += batch.num_graphs
    return total / count

def run_experiment(params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ntargets = len(params["targets"])
    # data
    train_loader, val_loader, test_loader = get_data_loaders(params["N"], params["batch_size"])

    # model
    model = GCNNalpha(
        dims=params["dims"],
        output_dim= ntargets,
        degrees=params["degrees"],
        activations=params["act_fns"],
        gso_generator=params["gso_generator"],
        alpha=params["alpha"],
        pooling_fn=params.get("pooling_fn", None),
        readout_dims=params.get("readout_dims", None),
        apply_pooling=params.get("apply_pooling", True),
        apply_readout=params.get("apply_readout", True),
    ).to(device)

    # ptimizer, scheduler, loss
    optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     patience=10, factor=0.5, verbose=True)
    loss_fn = nn.L1Loss()

    best_val = float('inf')
    best_state = None

    for epoch in range(1, params["num_epochs"] + 1):
        train_mae = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_mae   = eval_epoch(model, val_loader, loss_fn, device, tag='Val ')
        scheduler.step(val_mae)

        if val_mae < best_val:
            best_val   = val_mae
            best_state = model.state_dict()

        if epoch % 100 == 0 or epoch == params["num_epochs"]:
            print(f"Epoch {epoch}/{params['num_epochs']} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")

    # test
    model.load_state_dict(best_state)
    test_mae = eval_epoch(model, test_loader, loss_fn, device, tag='Test')
    print(f"Final Test MAE: {test_mae:.4f}")

    return best_val, test_mae