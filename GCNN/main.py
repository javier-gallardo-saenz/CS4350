#main.py
import torch 
from torch.nn import Tanh
from utils.operators import batched_hub_laplacian, batched_adv_diff
from gcnn_train import train

PARAMS ={
    "lr": 1e-4,
    "min_lr": 1e-5,
    "num_epochs": 1000,
    "weight_decay" : 0., 
    "dims" : [64, 64, 64, 1 ] , #first dimension is the batch size 
    "degrees": [1]* 2,
    "act_fns": [Tanh]* 2,
    "weight_decay": 0.4,
}


if __name__ == "__main__":

    PARAMS = {
        "lr": 1e-3,
        "min_lr": 1e-5,
        "num_epochs": 300,
        "weight_decay": 1e-5,
        "dims": [11, 64, 64, 1],  # Assuming 11 input features in QM9
        "degrees": [1, 1],  # polynomial degrees per layer
        "act_fns": [torch.nn.Tanh(), torch.nn.Tanh()],
        "readout_dims": [128, 64, 1],  # readout MLP layers
    }

    # GSO generator function (batched version)
    gso_generator = lambda A: batched_hub_laplacian(A, alpha=0.5)

    # Train the model
    model, best_val_loss, test_loss = train(PARAMS, gso_generator)

    # Optionally, save the trained model
    torch.save(model.state_dict(), "best_model.pth")
    print(f"Training finished! Best Val Loss: {best_val_loss:.4f}, Test Loss: {test_loss:.4f}")
