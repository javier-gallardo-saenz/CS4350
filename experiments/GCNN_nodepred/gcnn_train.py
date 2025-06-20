# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from gcnn import GCNNalpha
from data import get_karateclub_data, get_karateclub_data_custom

def train_epoch(model, data, optimizer, loss_fn, mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)  # Forward pass
    loss = loss_fn(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    pred = out.argmax(dim=1)
    correct = pred[mask] == data.y[mask]
    acc = int(correct.sum()) / int(mask.sum())
    return loss.item(), acc

@torch.no_grad()
def eval_epoch(model, data, loss_fn, mask, eval_embeddings=False):
    model.eval()
    with torch.no_grad():
        if eval_embeddings:
            out, embeddings = model(data.x, data.edge_index, eval_embeddings=True)
            loss = loss_fn(out[mask], data.y[mask])
            pred = out.argmax(dim=1)
            correct = pred[mask] == data.y[mask]
            acc = int(correct.sum()) / int(mask.sum())
            return loss.item(), acc, embeddings, pred
        else:
            out = model(data.x, data.edge_index)
            loss = loss_fn(out[mask], data.y[mask])
            pred = out.argmax(dim=1)
            correct = pred[mask] == data.y[mask]
            acc = int(correct.sum()) / int(mask.sum())
        return loss.item(), acc


def run_experiment(params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    data, train_mask, val_mask, test_mask = get_karateclub_data_custom()
    data = data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # model
    model = GCNNalpha(
        dims=params["dims"],
        output_dim= 4,
        degrees=params["degrees"],
        activations=params["act_fns"],
        gso_generator=params["gso_generator"],
        alpha=params["alpha"],
        readout_dims=params.get("readout_dims", None),
        apply_readout=params.get("apply_readout", True),
        eval_embeddings=params.get("eval_embeddings", False) 
    ).to(device)

    # optimizer, scheduler, loss
    optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
    loss_fn = nn.CrossEntropyLoss()

    best_val_ce = float('inf')
    best_state = None

    train_ce_history = []
    val_ce_history = []

    for epoch in tqdm(range(1, params["num_epochs"] + 1)):
        train_ce, train_acc = train_epoch(model, data, optimizer, loss_fn, train_mask)
        val_ce, val_acc= eval_epoch(model, data, loss_fn, val_mask)

        scheduler.step(val_ce)

        train_ce_history.append(train_ce)
        val_ce_history.append(val_ce)

        if val_ce < best_val_ce:
            best_val_ce = val_ce
            best_state = model.state_dict()

        if epoch % 10 == 0 or epoch == params["num_epochs"]:
            print(f"Epoch {epoch}/{params['num_epochs']} | Train CE: {train_ce:.4f} | Val CE : {val_ce:.4f}")
            print(f"Validation Accuracy: {val_acc:.4f}")
            #print(f"Alpha: {model.alpha.item()}")
            
    # test on the best model
    model.load_state_dict(best_state)

    if params.get("eval_embeddings", False):    # returns embeddings if eval_embeddings is True
        _, test_acc, embeddings, pred = eval_epoch(model, data, loss_fn, test_mask, eval_embeddings=True)
    else:
        _, test_acc = eval_epoch(model, data, loss_fn, test_mask, eval_embeddings=False)

    if params.get("eval_embeddings", False):    # returns embeddings if eval_embeddings is True
        return best_val_ce, test_acc, val_ce_history, embeddings, pred
    else:
        return best_val_ce, test_acc, val_ce_history