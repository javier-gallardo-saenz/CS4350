import torch
import torch.optim as opt
import torch.nn as nn
from tqdm import tqdm

from train_eval_QM9 import train_epoch, evaluate_network


def train_QM9(model, optimizer, train_loader, val_loader, prop_idx, factor, device, num_epochs, min_lr,
              diffusion_operator):
    loss_fn = nn.L1Loss()

    scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                   factor=0.5,
                                                   patience=15,
                                                   threshold=0.004,
                                                   verbose=True)

    epoch_train_MAEs, epoch_val_MAEs = [], []

    Best_val_mae = 1000

    print("Start training")

    for epoch in tqdm(range(num_epochs)):

        if optimizer.param_groups[0]['lr'] < min_lr:
            print("lr equal to min_lr: exist")
            break

        epoch_train_mae, optimizer = train_epoch(model, train_loader, optimizer, prop_idx, factor, device, loss_fn,
                                                 diffusion_operator)
        epoch_val_mae = evaluate_network(model, val_loader, prop_idx, factor, device, diffusion_operator)

        epoch_train_MAEs.append(epoch_train_mae)
        epoch_val_MAEs.append(epoch_val_mae)

        scheduler.step(epoch_val_mae)
        if (epoch_val_mae < Best_val_mae):
            Best_val_mae = epoch_val_mae
            torch.save(model, 'model.pth')

        torch.save(model, 'model_running.pth')

        print("")
        print("epoch_idx", epoch)
        print("epoch_train_MAE", epoch_train_mae)
        print("epoch_val_MAE", epoch_val_mae)

        torch.cuda.empty_cache()

    print("Finish training")