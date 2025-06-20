import torch
import torch.optim as opt
import torch.nn as nn
from tqdm import tqdm

from train_eval_QM9 import train_epoch, evaluate_network



def train_QM9(model, optimizer, train_loader, val_loader, prop_idx, factor, device, num_epochs, min_lr,
              diffusion_operator, early_stopping_patience=20, path_to_save_model="model.pth"):

    loss_fn = nn.L1Loss() # Mean Absolute Error is good for QM9 properties

    scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                   factor=0.5,
                                                   patience=15,  # This patience is for LR reduction, not early stopping
                                                   threshold=0.004,
                                                   verbose=True)

    epoch_train_MAEs, epoch_val_MAEs = [], []

    # --- Early Stopping Variables ---
    best_val_mae = float('inf') # Initialize with a very large number for MAE
    epochs_no_improve = 0       # Counter for epochs without improvement
    # ----------------------------------

    print("Start training")

    for epoch in tqdm(range(num_epochs)):

        # Check for minimum learning rate
        if optimizer.param_groups[0]['lr'] < min_lr:
            print(f"Learning rate ({optimizer.param_groups[0]['lr']:.6f}) below minimum ({min_lr:.6f}): stopping training.")
            break

        # --- Training Epoch ---
        model.train()  # Ensure model is in training mode
        epoch_train_mae, optimizer = train_epoch(model, train_loader, optimizer, prop_idx, factor, device, loss_fn,
                                                 diffusion_operator)

        # --- Validation Epoch ---
        model.eval()  # Ensure model is in evaluation mode
        epoch_val_mae = evaluate_network(model, val_loader, prop_idx, factor, device, diffusion_operator)

        epoch_train_MAEs.append(epoch_train_mae)
        epoch_val_MAEs.append(epoch_val_mae)

        # Learning Rate Scheduler Step (uses its own patience)
        scheduler.step(epoch_val_mae)

        print("")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train MAE: {epoch_train_mae:.6f}")
        print(f"  Val MAE:   {epoch_val_mae:.6f}")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}") # Show current LR

        # --- Early Stopping Logic ---
        if epoch_val_mae < best_val_mae:
            best_val_mae = epoch_val_mae
            epochs_no_improve = 0 # Reset counter as performance improved
            torch.save(model, path_to_save_model)
            print(f"  Validation MAE improved. Saving model to '{path_to_save_model}'. Best MAE: {best_val_mae:.6f}")
        else:
            epochs_no_improve += 1 # Increment counter
            print(f"  Validation MAE did not improve. Patience count: {epochs_no_improve}/{early_stopping_patience}")

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}! No improvement for {early_stopping_patience} epochs.")
            break # Exit the training loop


        # To checkpoint every epoch, consider saving state_dict.
        # torch.save(model.state_dict(), 'model_running_state_dict.pth') # Example

        torch.cuda.empty_cache() # Good practice

    print("Finish training")