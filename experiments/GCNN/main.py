#main.py
from torch.nn import Tanh, ReLU
from operators import hub_laplacian, adv_diff
from gcnn_train import run_experiment
from torch_geometric.nn import global_mean_pool

if __name__ == "__main__":
    PARAMS = {
        "N": 1000, #number of molecules to use
        "targets" : [0, 1, 2],
        "batch_size":   64,
        "lr":           1e-3,
        "weight_decay": 1e-5,
        "num_epochs":   200,
        "dims":         [11, 64, 64],        #  THIS DOES NOT INCLUDE OUTPUT. OUTPUT SIZE SET BY len(targets)

        "degrees":      [1]* 2,                 # must be = to number of hidden layer
        "act_fns":      [ReLU()]* 2,            # must be = to number of hidden layer
        "alpha":        0.5,                    # alpha to initialize GSO
        "readout_dims": [64, 3],               # first dimension must match last of the hidden layer
        "apply_pooling":  True,
        "apply_readout": True,
        "gso_generator": hub_laplacian,
        "pooling_fn": global_mean_pool,
    }
    best_val_mean_mae, final_test_per_target_mae, val_per_target_mae_history = run_experiment(PARAMS)
    print("\nExperiment Summary:")
    print(f"Best Validation Mean MAE: {best_val_mean_mae:.4f}")
    print(f"Final Test MAE (per target): {final_test_per_target_mae}")