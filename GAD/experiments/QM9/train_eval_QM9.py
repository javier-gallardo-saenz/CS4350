import torch
import torch.nn as nn
import gc

from tqdm import tqdm

from torch_geometric.utils import to_dense_adj, get_laplacian


def evaluate_network(model, data_loader, prop_idx, factor, device, diffusion_operator):
    model.eval()

    # Convert factor to a tensor here, will be used for both target and output scaling
    if isinstance(prop_idx, list):
        num_properties = len(prop_idx)
        factors_tensor = torch.tensor(factor, dtype=torch.float32).to(device)
        individual_epoch_test_maes = torch.zeros(num_properties).to(device)
    else:
        num_properties = 1
        factors_tensor = torch.tensor([factor], dtype=torch.float32).to(device)
        individual_epoch_test_maes = torch.zeros(1).to(device)

    epoch_test_mae = 0

    # For evaluation, we still use reduction='none' to get per-property MAEs before final averaging
    loss_fn_eval = nn.L1Loss(reduction='none')

    with torch.no_grad():
        for idx, batched_graph in tqdm(enumerate(data_loader)):
            num_nodes = batched_graph.num_nodes
            adj = to_dense_adj(batched_graph.edge_index, max_num_nodes=num_nodes)[0].to(device)
            node_deg_vec = adj.sum(axis=1, keepdim=True).to(device)
            node_deg_mat = torch.diag(node_deg_vec[:, 0]).to(device)
            lap_mat = diffusion_operator(adj)
            F_norm_edge = batched_graph.F_norm_edge.to(device)
            F_dig = batched_graph.F_dig.to(device)
            edge_index = batched_graph.edge_index.to(device)
            batch_idx = batched_graph.batch.to(device)
            node_fts = batched_graph.x.to(device).squeeze()
            automic_num = batched_graph.z.to(device)
            edge_fts = batched_graph.edge_attr.to(device)
            norm_n = batched_graph.norm_n.to(device)
            k_eig_val = batched_graph.k_eig_val.to(device)
            k_eig_vec = batched_graph.k_eig_vec.to(device)

            out_model = model(node_fts, automic_num, edge_fts, edge_index, F_norm_edge, F_dig, node_deg_vec,
                              node_deg_mat, lap_mat, k_eig_val, k_eig_vec, num_nodes, norm_n, batch_idx)

            if isinstance(prop_idx, list):
                target_properties = batched_graph.y[:, prop_idx].to(device)
            else:
                target_properties = batched_graph.y[:, prop_idx].unsqueeze(1).to(device)

            # --- Apply factors to both model output and target properties ---
            scaled_out_model = out_model * factors_tensor
            scaled_target_properties = target_properties * factors_tensor

            # Calculate loss based on scaled values
            per_sample_per_property_loss = loss_fn_eval(scaled_out_model, scaled_target_properties)

            # Combined MAE is the mean of all scaled errors
            combined_loss = per_sample_per_property_loss.mean()
            epoch_test_mae += combined_loss.item()

            # Individual MAEs are the mean of scaled errors per property
            individual_batch_maes = per_sample_per_property_loss.mean(dim=0)
            individual_epoch_test_maes += individual_batch_maes

            # Memory management
            del out_model
            del batched_graph
            del adj
            del node_deg_vec
            del lap_mat
            del node_deg_mat
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        epoch_test_mae /= (idx + 1)
        individual_epoch_test_maes /= (idx + 1)

    return epoch_test_mae, individual_epoch_test_maes


def train_epoch(model, data_loader, optimizer, prop_idx, factor, device, loss_fn, diffusion_operator):
    epoch_train_mae = 0

    # Convert factor to a tensor here, will be used for both target and output scaling
    if isinstance(prop_idx, list):
        num_properties = len(prop_idx)
        factors_tensor = torch.tensor(factor, dtype=torch.float32).to(device)
    else:
        num_properties = 1
        factors_tensor = torch.tensor([factor], dtype=torch.float32).to(device)

    model.train()

    for idx, batched_graph in tqdm(enumerate(data_loader)):
        optimizer.zero_grad()

        num_nodes = batched_graph.num_nodes
        adj = to_dense_adj(batched_graph.edge_index, max_num_nodes=num_nodes)[0].to(device)
        node_deg_vec = adj.sum(axis=1, keepdim=True).to(device)
        node_deg_mat = torch.diag(node_deg_vec[:, 0]).to(device)
        lap_mat = diffusion_operator(adj)
        F_norm_edge = batched_graph.F_norm_edge.to(device)
        F_dig = batched_graph.F_dig.to(device)
        edge_index = batched_graph.edge_index.to(device)
        batch_idx = batched_graph.batch.to(device)
        node_fts = batched_graph.x.to(device).squeeze()
        automic_num = batched_graph.z.to(device)
        edge_fts = batched_graph.edge_attr.to(device)
        norm_n = batched_graph.norm_n.to(device)
        k_eig_val = batched_graph.k_eig_val.to(device)
        k_eig_vec = batched_graph.k_eig_vec.to(device)

        out_model = model(node_fts, automic_num, edge_fts, edge_index, F_norm_edge, F_dig, node_deg_vec, node_deg_mat,
                          lap_mat, k_eig_val, k_eig_vec, num_nodes, norm_n, batch_idx)

        if isinstance(prop_idx, list):
            target_properties = batched_graph.y[:, prop_idx].to(device)
        else:
            target_properties = batched_graph.y[:, prop_idx].unsqueeze(1).to(device)

        # --- Apply factors to both model output and target properties ---
        scaled_out_model = out_model * factors_tensor
        scaled_target_properties = target_properties * factors_tensor

        # Pass the scaled outputs and targets to the loss function
        # Assuming loss_fn is nn.L1Loss(reduction='mean')
        loss = loss_fn(scaled_out_model, scaled_target_properties)

        loss.backward()
        optimizer.step()

        epoch_train_mae += loss.item()

        # Memory management
        del out_model
        del loss
        del batched_graph
        del adj
        del node_deg_vec
        del lap_mat
        del node_deg_mat
        gc.collect()  # Python's garbage collector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # PyTorch's CUDA memory cache

    epoch_train_mae /= (idx + 1)
    return epoch_train_mae, optimizer
