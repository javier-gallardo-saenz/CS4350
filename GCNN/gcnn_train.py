# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from gcnn import GCNNalpha
import os 
from data import get_data_loaders
from gs_utils import save_experiment_results, generate_run_id


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total, count = 0.0, 0
    for batch in loader:
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
def eval_epoch(model, data_loader, params, device, tag='Val'):
    model.eval()
    total_errors = None
    total_samples = 0

    with torch.no_grad():
        for data in data_loader:
            x = data.x.to(device)
            batch = data.batch.to(device)
            edge_index = data.edge_index.to(device)
            y = data.y[:, params["targets"]].to(device)  # select relevant targets

            out = model(x, batch, edge_index)

            abs_errors = torch.abs(out - y)  # Per-sample, per-target absolute errors

            if total_errors is None:
                total_errors = abs_errors.sum(dim=0)
            else:
                total_errors += abs_errors.sum(dim=0)

            total_samples += y.size(0)

    per_target_mae = (total_errors / total_samples).cpu().numpy()

    #print(f"{tag} MAE per target: {per_target_mae}")
    return per_target_mae


def run_experiment(params):
    # Create a unique run ID using a timestamp
    run_id = generate_run_id(length=4)
    
    # Define save directory
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

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
        learn_alpha= params.get("learn_alpha", True),
        reduction= params.get("pooling"),
        readout_dims=params.get("readout_dims", None),
        apply_readout=params.get("apply_readout", True),
    ).to(device)

    # optimizer, scheduler, loss
    optimizer = optim.Adam([
        # Parameters excluding 'alpha'
        {'params': [p for name, p in model.named_parameters() if 'alpha' not in name],
         'lr': params['lr'],
         'weight_decay': params['weight_decay']},

        # 'alpha' parameter with its own learning rate
        {'params': model.alpha,
         'lr': params["alpha_lr"],
         'weight_decay': 0.0} # Typically no weight decay for a single scalar parameter like alpha
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     patience=10, factor=0.5, verbose=True)
    loss_fn = nn.L1Loss()

    best_val_mean_mae = float('inf')
    best_state = None

    train_mae_history = []
    val_mean_mae_history = []
    val_per_target_mae_history = [] # New list to store per-target MAE
    alphas_history = []

    for epoch in tqdm(range(1, params["num_epochs"] + 1)):
        train_mae = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_per_target_mae = eval_epoch(model, val_loader, params, device, tag='Val') # Get per-target MAE
        val_mean_mae = val_per_target_mae.mean() # Calculate mean for scheduling

        scheduler.step(val_mean_mae)

        train_mae_history.append(train_mae)
        val_mean_mae_history.append(val_mean_mae)
        val_per_target_mae_history.append(val_per_target_mae.tolist()) # Store the per-target MAE as list

        if isinstance(model.alpha, torch.Tensor) and model.alpha.numel() == 1:
            alpha = model.alpha.item()
        else:
            alpha = model.alpha

        alphas_history.append(alpha)

        if val_mean_mae < best_val_mean_mae:
            best_val_mean_mae = val_mean_mae
            best_state = model.state_dict() # Capture the state dict of the best model

        if epoch % 10 == 0 or epoch == params["num_epochs"]:
            print(f"Epoch {epoch}/{params['num_epochs']} | Train MAE: {train_mae:.4f} | Val MAE (per target): {val_per_target_mae} | alpha: {alpha}")
            
    # test
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("Warning: No best model state found. Using the last trained model for testing.")
    test_per_target_mae = eval_epoch(model, test_loader, params, device, tag='Test') # Pass params here
    print(f"Final Test MAE (mean): {test_per_target_mae.mean():.4f}")
    print(f"Final Test MAE (per target): {test_per_target_mae}")

    # Call the new function to save all results

    return best_val_mean_mae, test_per_target_mae, val_per_target_mae_history, alphas_history