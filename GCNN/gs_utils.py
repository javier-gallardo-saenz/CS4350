import pandas as pd
import os
import torch
import matplotlib.pyplot as plt 
import random 
import string

def generate_run_id(length=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


def save_experiment_results(run_id, save_dir, params, best_val_mean_mae, 
                            test_per_target_mae, val_per_target_mae_history, 
                            best_model_state=None, ntargets=None):
    """
    Saves all experiment results, parameters, and the best model into a single CSV.

    Args:
        run_id (str): Unique identifier for the experiment run.
        save_dir (str): Directory to save the results.
        params (dict): Dictionary of experiment parameters.
        best_val_mean_mae (float): Best mean validation MAE achieved.
        test_per_target_mae (torch.Tensor): Per-target MAE on the test set.
        val_per_target_mae_history (list): List of per-target MAEs for each validation epoch.
        best_model_state (dict, optional): The state_dict of the best performing model.
        ntargets (int, optional): Number of targets. Inferred from test_per_target_mae if not provided.
    """
    os.makedirs(save_dir, exist_ok=True)

    if ntargets is None and test_per_target_mae is not None:
        ntargets = len(test_per_target_mae)
    elif ntargets is None:
        print("Warning: ntargets could not be determined. Some saving features might be limited.")
        ntargets = 0 # Default to 0 if no targets can be inferred

    # Save best model state if provided
    if best_model_state is not None:
        model_save_path = os.path.join(save_dir, f"{run_id}_best_model.pth")
        torch.save(best_model_state, model_save_path)
        print(f"Best model saved to {model_save_path}")

    # Prepare data for the single results CSV
    results_data = {
        "run_id": run_id,
        "best_val_mean_mae": best_val_mean_mae,
        "mean_test_mae": test_per_target_mae.mean().item(),
    }
    
    # Add per-target test MAE
    for i in range(ntargets):
        results_data[f"test_target_{i}_mae"] = test_per_target_mae[i].item()

    # Add all parameters
    for key, value in params.items():
        if isinstance(value, type): # For callable objects like Tanh or hub_laplacian
            results_data[f"param_{key}"] = value.__name__
        elif isinstance(value, (list, dict)):
            results_data[f"param_{key}"] = str(value) # Convert lists/dicts to string for CSV
        elif isinstance(value, (torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau)):
             results_data[f"param_{key}"] = str(value) # Convert complex objects to string representation
        else:
            results_data[f"param_{key}"] = value
    
    # Create a DataFrame from the single row of results and parameters
    results_df = pd.DataFrame([results_data])

    results_csv_path = os.path.join(save_dir, f"{run_id}_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results and parameters saved to {results_csv_path}")

    # Save validation per-target MAE history (still separate for detailed epoch-wise data)
    if val_per_target_mae_history: # Only save if history is not empty
        val_history_df = pd.DataFrame(val_per_target_mae_history, 
                                      columns=[f"val_epoch_target_{i}_mae" for i in range(len(val_per_target_mae_history[0]))])
        val_history_csv_path = os.path.join(save_dir, f"{run_id}_val_per_target_mae_history.csv")
        val_history_df.to_csv(val_history_csv_path, index=False)
        print(f"Validation per-target MAE history saved to {val_history_csv_path}")

def plot_val_mae_per_target(val_per_target_mae_history, run_id, save_dir="grid_search_results"):
    """
    Plots and saves per-target validation MAE curves.

    Args:
        val_per_target_mae_history (list of lists): Each entry is a list of per-target MAEs for that epoch.
        run_id (str): Unique identifier for the run.
        save_dir (str): Directory to save the plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = list(range(1, len(val_per_target_mae_history) + 1))
    val_per_target_mae_array = list(map(list, zip(*val_per_target_mae_history)))  # Transpose: targets x epochs

    plt.figure(figsize=(10, 6))

    # Plot each target’s validation MAE
    for idx, target_mae in enumerate(val_per_target_mae_array):
        plt.plot(epochs, target_mae, label=f"Val MAE - Target {idx}")

    plt.xlabel("Epoch")
    plt.ylabel("Validation MAE")
    plt.title(f"Validation MAE per Target (Run {run_id})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, f"loss_{run_id}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_alphas_history(alphas_history, run_id, save_dir="grid_search_results"):
    """
    Plots and saves per-target validation MAE curves.

    Args:
        val_per_target_mae_history (list of lists): Each entry is a list of per-target MAEs for that epoch.
        run_id (str): Unique identifier for the run.
        save_dir (str): Directory to save the plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = list(range(1, len(alphas_history) + 1))

    plt.figure(figsize=(10, 6))

    # Plot each target’s validation MAE
    plt.plot(epochs, alphas_history)

    plt.xlabel("Epoch")
    plt.ylabel("alpha")
    plt.title(f"alpha (Run {run_id})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, f"alpha_{run_id}.png")
    plt.savefig(plot_path)
    plt.close()