#main.py
import torch 
from torch.nn import Tanh, ReLU
from utils.operators import hub_laplacian, adv_diff
from gcnn_train import run_experiment
from torch_geometric.nn import global_mean_pool

if __name__ == "__main__":
    # Example params for a single run
    PARAMS = {
        "N": 10000, #number of molecules to use
        "targets" : [0, 1, 2], 
        "batch_size":   64,
        "lr":           1e-3,
        "weight_decay": 1e-5,
        "num_epochs":   200,
        "dims":         [11, 64, 64],        #  THIS DOES NOT INCLUDE OUTPUT. OUTPUT SIZE SET BY len(targets)

        "degrees":      [1]* 2,                 # must be = to number of hidden layer
        "act_fns":      [ReLU()]* 2,            # must be = to number of hidden layer
        "alpha":        0.5,                    # alpha to initialize GSO
        "readout_dims": [128, 3],               # first dimension must match last of the hidden layer
        "apply_pooling":  True,                 
        "apply_readout": True,                  # if you set false make 
        "gso_generator": adv_diff,
        "pooling_fn": global_mean_pool,
    }
    run_experiment(PARAMS)