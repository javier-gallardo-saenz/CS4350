# main.py
import torch
from torch.nn import Tanh, ReLU
from operators import hub_laplacian, adv_diff # Assuming these are correctly defined
from gcnn_train import run_experiment
import itertools
import os
import pandas as pd
from datetime import datetime

# --- Grid Search Parameters ---
# Define the parameters you want to loop over and their possible values
GRID_PARAMS = {
    "lr": [1e-3, 5e-4, 1e-4],
    "weight_decay": [1e-5, 1e-6],
    "alpha": [0.1, 0.5, 0.9],
    # Example for `dims` if you wanted to try different architectures,
    # but be careful as this changes the model structure
    # "dims": [[11, 32, 32], [11, 64, 64]],
    # Example for 'gso_generator'
    # "gso_generator": [hub_laplacian, adv_diff],
    # "learn_alpha": [True, False],
}

BASE_PARAMS = {
    "N": 1000, #number of molecules to use
    "targets" : [0, 1, 2],
    "batch_size":   64,
    "num_epochs":   10, # Reduced for faster grid search demonstration
    "dims":         [11, 64, 64],        #  THIS DOES NOT INCLUDE OUTPUT. OUTPUT SIZE SET BY len(targets)
    "degrees":      [1]* 2,                 # must be = to number of hidden layer
    "act_fns":      [Tanh()]* 2,            # must be = to number of hidden layer
    "readout_dims": [64, 3],               # first dimension must match last of the hidden layer
    "reduction" : 'sum',
    "apply_readout": True,
    "learn_alpha" : True,
    "gso_generator": hub_laplacian,
}

if __name__ == "__main__":
    grid_search_results_dir = "grid_search_results"
    os.makedirs(grid_search_results_dir, exist_ok=True)

    all_grid_results = []
    param_names = GRID_PARAMS.keys()
    param_values = GRID_PARAMS.values()

    # Generate all combinations of parameters
    for i, combo in enumerate(itertools.product(*param_values)):
        current_params = BASE_PARAMS.copy() # Start with base parameters for each run

        # Apply the current combination of grid parameters
        combo_dict = dict(zip(param_names, combo))
        current_params.update(combo_dict) # Overwrite base params with current grid params

        print(f"\n--- Running Experiment {i+1} with Parameters: ---")
        for k, v in combo_dict.items():
            # For functions/classes, print their name
            if callable(v):
                print(f"  {k}: {v.__name__}")
            else:
                print(f"  {k}: {v}")

        # Run the experiment with the current set of parameters
        best_val_mean_mae, final_test_per_target_mae, val_per_target_mae_history = run_experiment(current_params)

        # Store the results for this combination
        run_summary = {
            "run_id": f"grid_run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}", # More precise timestamp
            "best_val_mean_mae": best_val_mean_mae,
            "mean_test_mae": final_test_per_target_mae.mean().item(),
        }

        # Add per-target test MAE to the summary
        for j, mae_val in enumerate(final_test_per_target_mae):
            run_summary[f"test_target_{j}_mae"] = mae_val.item()

        # Add the parameters used for this run
        for k, v in current_params.items():
            if isinstance(v, type): # e.g., Tanh, hub_laplacian
                run_summary[f"param_{k}"] = v.__name__
            elif isinstance(v, (list, tuple)): # e.g., dims, degrees, act_fns
                run_summary[f"param_{k}"] = str(v)
            elif isinstance(v, torch.nn.Module): # For instantiated modules like Tanh()
                run_summary[f"param_{k}"] = v.__class__.__name__
            else:
                run_summary[f"param_{k}"] = v
        
        all_grid_results.append(run_summary)

    grid_summary_df = pd.DataFrame(all_grid_results)
    grid_summary_csv_path = os.path.join(grid_search_results_dir, "grid_search_summary.csv")
    grid_summary_df.to_csv(grid_summary_csv_path, index=False)
    print(f"\nGrid Search Summary saved to {grid_summary_csv_path}")

    print("\n--- Grid Search Complete ---")
    print("Best performing run (based on Best Validation Mean MAE):")
    if not grid_summary_df.empty:
        best_run_row = grid_summary_df.loc[grid_summary_df['best_val_mean_mae'].idxmin()]
        print(best_run_row)