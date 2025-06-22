import torch
import torch.optim as opt
import torch.nn as nn
from tqdm import tqdm

from train_eval_QM9 import train_epoch, evaluate_network



def train_QM9(model, optimizer, train_loader, val_loader, prop_idx, factor, device, num_epochs, min_lr,
              diffusion_operator, early_stopping_patience=20, path_to_save_model="model.pth"):

    loss_fn = nn.L1Loss('mean') # Mean Absolute Error is good for QM9 properties

    scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                   factor=0.5,
                                                   patience=15,  # This patience is for LR reduction, not early stopping
                                                   threshold=0.004,
                                                   verbose=True)

    epoch_train_MAEs, epoch_val_MAEs = [], []
    # You might want to store individual MAEs for plotting later, e.g., as a list of lists/tensors
    # individual_epoch_val_MAEs_history = []

    # --- Early Stopping Variables ---
    best_val_mae = float('inf') # Initialize with a very large number for MAE
    epochs_no_improve = 0       # Counter for epochs without improvement
    # ----------------------------------

    print("Start training")

    for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):

        # Check for minimum learning rate
        if optimizer.param_groups[0]['lr'] < min_lr:
            tqdm.write(f"Learning rate ({optimizer.param_groups[0]['lr']:.6f}) below minimum ({min_lr:.6f}): stopping training.")
            break

        # --- Training Epoch ---
        model.train()
        # train_epoch now returns combined_train_mae and optimizer
        combined_train_mae, optimizer = train_epoch(model, train_loader, optimizer, prop_idx, factor, device, loss_fn,
                                                 diffusion_operator)

        # --- Validation Epoch ---
        model.eval()
        # evaluate_network now returns combined_val_mae and individual_val_maes_tensor
        combined_val_mae, individual_val_maes_tensor = evaluate_network(model, val_loader, prop_idx, factor, device, diffusion_operator)

        epoch_train_MAEs.append(combined_train_mae) # Store combined train MAE
        epoch_val_MAEs.append(combined_val_mae)   # Store combined val MAE
        # individual_epoch_val_MAEs_history.append(individual_val_maes_tensor.cpu()) # Store individual val MAEs if needed later

        alpha_value_layer0 = model.layers[0].alpha.item()
        gamma_adv_value_layer0 = model.layers[0].gamma_adv.item()
        gamma_diff_value_layer0 = model.layers[0].gamma_diff.item()
        print(f"Epoch {epoch + 1}: Alpha Parameter Value (Layer 0): {alpha_value_layer0:.4f}")
        print(f"Epoch {epoch + 1}: Gamma_adv Parameter Value (Layer 0): {gamma_adv_value_layer0:.4f}")
        print(f"Epoch {epoch + 1}: Gamma_diff Parameter Value (Layer 0): {gamma_diff_value_layer0:.4f}")

        # Learning Rate Scheduler Step (uses its own patience)
        scheduler.step(combined_val_mae)

        tqdm.write("") # Newline for cleaner output with tqdm
        tqdm.write(f"Epoch {epoch+1}/{num_epochs}")
        tqdm.write(f"  Train MAE (Combined): {combined_train_mae:.6f}")
        tqdm.write(f"  Val MAE (Combined):   {combined_val_mae:.6f}")
        tqdm.write(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}") # Show current LR

        # --- Print Individual Validation MAEs ---
        if isinstance(prop_idx, list):
            tqdm.write("  Validation Individual MAEs:")
            # Use prop_idx to get the original index names if desired, or just enumerate
            for i, p_idx_val in enumerate(prop_idx):
                tqdm.write(f"    Property {p_idx_val}: {individual_val_maes_tensor[i].item():.6f}")
        else:
            # If prop_idx is not a list, individual_val_maes_tensor will have 1 element
            tqdm.write(f"  Validation Individual MAE: {individual_val_maes_tensor.item():.6f}")


        # --- Early Stopping Logic ---
        if combined_val_mae < best_val_mae: # Check against combined validation MAE
            best_val_mae = combined_val_mae
            epochs_no_improve = 0 # Reset counter as performance improved
            torch.save(model, path_to_save_model) # Save model's state_dict
            tqdm.write(f"  Validation MAE (Combined) improved. Saving model to '{path_to_save_model}'. Best MAE: {best_val_mae:.6f}")
        else:
            epochs_no_improve += 1 # Increment counter
            tqdm.write(f"  Validation MAE (Combined) did not improve. Patience count: {epochs_no_improve}/{early_stopping_patience}")

        if epochs_no_improve >= early_stopping_patience:
            tqdm.write(f"Early stopping triggered at epoch {epoch+1}! No improvement for {early_stopping_patience} epochs.")
            break # Exit the training loop

        torch.cuda.empty_cache() # Good practice

    tqdm.write("Finish training")
    # return model, epoch_train_MAEs, epoch_val_MAEs # Returning model for potential post-training use