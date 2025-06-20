# main.py
import torch
from torch.nn import Tanh, ReLU
from operators import hub_laplacian, adv_diff  # ensure these are defined
from gcnn_train import run_experiment
import itertools
import os
import pandas as pd
from gs_utils import generate_run_id, plot_val_mae_per_target, plot_alphas_history

# --------------------
# 1. DEFAULT PARAMS
# --------------------
DEFAULT_PARAMS = {
    "N": 1000,                        # number of molecules to use
    "targets": [0, 1, 2],             # which targets to predict
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "alpha": 0.0,
    "num_epochs": 300,                 
    "dims": [150, 64, 64, 64],             # input + hidden dims only
    "degrees": [1]*3,               # poly degree per conv layer
    "act_fns": [Tanh()]* 3,
    "readout_dims": [64, 3],          # last must match len(targets)
    "pooling": "sum",
    "apply_readout": True,
    "learn_alpha": False,
    "gso_generator": hub_laplacian,   # default graph shift op
}

# --------------------
# 2. GRID SEARCH SPACE
# --------------------
GRID_PARAMS = {
    "num_epochs": [500, 1000],
    "act_fns": [[Tanh()]*3 , [ReLU()]* 3],
    "pooling": ["sum", "max", "mean"],
    "alpha": [0, 0.5, -0.5]
}


keys, values = zip(*GRID_PARAMS.items())
grid = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

# --------------------
# 3. OUTPUT SETUP
# --------------------
RESULTS_DIR = "GCNN/grid_search_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "grid_search_summary.csv")

all_grid_results = []

# --------------------
# 4. RUN GRID SEARCH
# --------------------
for idx, grid_params in enumerate(grid, start=1):
    # merge defaults + grid overrides
    config = DEFAULT_PARAMS.copy()
    config.update(grid_params)

    # generate a short run ID
    run_id = generate_run_id(length=4)
    print(f"\n--- Run {idx}/{len(grid)} — ID: {run_id} ---")
    for k, v in grid_params.items():
        name = v.__name__ if callable(v) else v
        print(f"  {k}: {name}")

    # execute
    best_val_mean_mae, final_test_per_target_mae, val_per_target_mae_history, alphas_history = run_experiment(config)
    plot_val_mae_per_target(val_per_target_mae_history, run_id)
    plot_alphas_history(alphas_history, run_id)
    
    # collect results
    row = {
        "run_id": run_id,
        "best_val_mean_mae": best_val_mean_mae,
        "mean_test_mae": final_test_per_target_mae.mean().item(),
    }

    # per-target test MAEs
    for t_i, mae in enumerate(final_test_per_target_mae):
        row[f"test_target_{t_i}_mae"] = mae.item()

    # log all hyperparameters
    for param_name, param_value in config.items():
        if isinstance(param_value, torch.nn.Module):
            row[param_name] = param_value.__class__.__name__
        elif callable(param_value):
            row[param_name] = param_value.__name__
        else:
            row[param_name] = param_value

    all_grid_results.append(row)

# --------------------
# 5. SAVE TO CSV
# --------------------
df = pd.DataFrame(all_grid_results)
df.to_csv(CSV_PATH, index=False)
print(f"\n✔️  Grid search summary saved to {CSV_PATH}")

# print best run
if not df.empty:
    best = df.loc[df["best_val_mean_mae"].idxmin()]
    print("\n--- Best Run (lowest validation MAE) ---")
    print(best.to_string())
