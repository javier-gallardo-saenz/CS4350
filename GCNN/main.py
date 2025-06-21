# main.py
import torch
from torch.nn import ReLU
from operators import hub_laplacian, adv_diff
from gcnn_train import run_experiment  # this should return model, histories, etc.
import itertools
import os
import pandas as pd
from gs_utils import generate_run_id, plot_val_mae_per_target, plot_alphas_history
import csv

# -----------------------
# 1. Default + Grid Params
# -----------------------
DEFAULT_PARAMS = {
    "N": 1200,
    "targets": [0],
    "batch_size": 64,
    "lr": 1e-4,
    "alpha_lr": 1e-2,
    "weight_decay": 1e-5,
    "alpha": 0.5,
    "num_epochs": 250,
    "dims": [11, 64, 64],
    "hops": 2,
    "act_fn": ReLU(),
    "readout_hidden_dims": [64, 32],
    "pooling": "sum",
    "apply_readout": True,
    "learn_alpha": False,
    "gso_generator": hub_laplacian,
    "use_bn": True,
    "dropout_p": 0.2,
    "patience": 50
}

GRID_PARAMS = {
    "targets" : [[0], [1], [2]],
    "learn_alpha": [True, False],
    "alpha": [-0.5, 0, 0.5, 1.0],
    "pooling": ["sum", "max", "mean"]
}

# Build the grid
keys, values = zip(*GRID_PARAMS.items())
grid = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

# -----------------------
# 2. Prepare Result Paths
# -----------------------
TOP_RESULTS_DIR = "GCNN/results_t0"
os.makedirs(TOP_RESULTS_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(TOP_RESULTS_DIR, "summary.csv")
csv_exists = os.path.isfile(SUMMARY_CSV)

# We'll write header once
fieldnames = None

# -----------------------
# 3. Run Experiments
# -----------------------
for idx, grid_params in enumerate(grid, start=1):
    # Merge defaults + grid
    config = DEFAULT_PARAMS.copy()
    config.update(grid_params)

    # Unique run ID
    run_id = generate_run_id(4)
    print(f"\n--- Run {idx}/{len(grid)} — ID: {run_id} ---")
    for k, v in grid_params.items():
        name = v.__name__ if callable(v) else v
        print(f"  {k}: {name}")

    # Create run‐specific folder
    run_dir = os.path.join(TOP_RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    model, best_val_mean_mae, test_per_target_mae, val_per_target_mae_history, alphas_history = run_experiment(config)

    # --- Save model weights ---
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    # --- Save plots into run_dir ---
    plot_val_mae_per_target(val_per_target_mae_history, run_id, save_dir=run_dir)
    plot_alphas_history(alphas_history, run_id, save_dir=run_dir)

    # --- Prepare summary row ---
    row = {
        "run_id": run_id,
        "best_val_mean_mae": best_val_mean_mae,
        "mean_test_mae": test_per_target_mae.mean().item(),
    }
    # per-target test MAEs
    for t_i, mae in enumerate(test_per_target_mae):
        row[f"test_target_{t_i}_mae"] = mae.item()
    # record all hyperparameters
    for param_name, param_value in config.items():
        if isinstance(param_value, torch.nn.Module):
            row[param_name] = param_value.__class__.__name__
        elif callable(param_value):
            row[param_name] = param_value.__name__
        else:
            row[param_name] = param_value

    # initialize fieldnames on first run
    if fieldnames is None:
        fieldnames = list(row.keys())

    # --- Append to summary CSV ---
    with open(SUMMARY_CSV, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
            csv_exists = True
        writer.writerow(row)

    print(f"✅ Run {run_id} complete. Artifacts saved to {run_dir}")

# -----------------------
# 4. Final Summary
# -----------------------
print(f"\nAll runs complete. Summary written to {SUMMARY_CSV}")

# Optionally print best overall
df = pd.read_csv(SUMMARY_CSV)
best = df.loc[df["mean_test_mae"].idxmin()]
print("\n--- Best Run (lowest mean_test_mae) ---")
print(best.to_string())
