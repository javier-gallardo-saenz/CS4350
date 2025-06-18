import torch
import torch.nn as nn

from tqdm import tqdm

from torch_geometric.utils import to_dense_adj, get_laplacian

def evaluate_network(model, data_loader, device):

    model.eval()
    
    epoch_test_mae = 0


    loss_fn = nn.L1Loss()

    with torch.no_grad():

        for idx, batched_graph in tqdm(enumerate(data_loader)):

            
            
            num_nodes = batched_graph.num_nodes
            adj = to_dense_adj(batched_graph.edge_index, max_num_nodes = num_nodes)[0].to(device)
            node_deg_vec = adj.sum(axis=1, keepdim=True).to(device)
            
            node_deg_mat = torch.diag(node_deg_vec[:, 0]).to(device)

            lap_mat_sparse = get_laplacian(batched_graph.edge_index)
            lap_mat = to_dense_adj(edge_index = lap_mat_sparse[0], edge_attr = lap_mat_sparse[1], max_num_nodes = num_nodes)[0].to(device)
   
            F_norm_edge = batched_graph.F_norm_edge.to(device)
            F_dig = batched_graph.F_dig.to(device)
            
            edge_index = batched_graph.edge_index.to(device)
            batch_idx = batched_graph.batch.to(device)

            node_fts = batched_graph.x.to(device)
            node_fts = node_fts.squeeze()
            
            edge_fts = batched_graph.edge_attr.to(device)
            norm_n = batched_graph.norm_n.to(device)
            
            k_eig_val = batched_graph.k_eig_val.to(device)
            k_eig_vec = batched_graph.k_eig_vec.to(device)
     
            out_model = model(node_fts, edge_fts, edge_index, F_norm_edge, F_dig, node_deg_vec, node_deg_mat, lap_mat, k_eig_val, k_eig_vec, num_nodes, norm_n, batch_idx)

            loss = loss_fn(out_model, batched_graph.y.to(device))

            epoch_test_mae += loss.item()  # MAE: mean absolute error


        epoch_test_mae /= (idx + 1)
        
    return epoch_test_mae


def train_epoch(model, data_loader, optimizer, device, loss_fn,
                diffusion_operator="Laplacian"):
    
        epoch_train_mae = 0

        model.train() 

        for idx, batched_graph in tqdm(enumerate(data_loader)):

            optimizer.zero_grad()
        
            num_nodes = batched_graph.num_nodes
            
            adj = to_dense_adj(batched_graph.edge_index, max_num_nodes = num_nodes)[0].to(device)
            node_deg_vec = adj.sum(axis=1, keepdim=True).to(device)
            
            node_deg_mat = torch.diag(node_deg_vec[:, 0]).to(device)

            if model.diffusion_operator['id'] == "Laplacian":
                lap_mat_sparse = get_laplacian(batched_graph.edge_index)
                lap_mat = to_dense_adj(edge_index = lap_mat_sparse[0], edge_attr = lap_mat_sparse[1],
                                       max_num_nodes = num_nodes)[0].to(device)

            elif model.diffusion_operator['id'] == "HubOperator":
                gsos = [] #This will store the hub operator for each graph
                num_nodes_per_graph = []  # This will store the actual number of nodes for each graph

                # 2. Iterate through each graph in the batch
                for i in range(adj.size(0)):  # adj.size(0) is the batch size (B)
                    A_i = adj[i]  # Get the i-th padded adjacency matrix from the batch, shape (N_max, N_max)

                    # Calculate the actual number of nodes for the current graph
                    # (Summing rows and checking for non-zero sum identifies non-padded nodes)
                    num_nodes_i = (A_i.sum(dim=1) != 0).sum().item()
                    num_nodes_per_graph.append(num_nodes_i)

                    # Crop to the actual (non-padded) adjacency matrix
                    A_i_unpadded = A_i[:num_nodes_i, :num_nodes_i]  # Shape (num_nodes_i, num_nodes_i)

                    # L_i will have shape (num_nodes_i, num_nodes_i)
                    L_i = operator(A_i_unpadded, model.hub_params)

                    # 3. Re-pad the result (L_i) to N_max x N_max
                    L_i_padded = torch.zeros(num_nodes, num_nodes, dtype=L_i.dtype, device=device)
                    L_i_padded[:num_nodes_i, :num_nodes_i] = L_i
                    gsos.append(L_i_padded)

                # 4. Stack the padded results vertically to get the desired (B, N_max, N_max) shape
                lap_mat = torch.stack(gsos, dim=0)

        
            F_norm_edge = batched_graph.F_norm_edge.to(device)
            F_dig = batched_graph.F_dig.to(device)
     
            edge_index = batched_graph.edge_index.to(device)
            batch_idx = batched_graph.batch.to(device)

            node_fts = batched_graph.x.to(device)
            node_fts = node_fts.squeeze()
            
            edge_fts = batched_graph.edge_attr.to(device)
            
            norm_n = batched_graph.norm_n.to(device)
            
            k_eig_val = batched_graph.k_eig_val.to(device)
            k_eig_vec = batched_graph.k_eig_vec.to(device)

            out_model = model(node_fts, edge_fts, edge_index, F_norm_edge, F_dig, node_deg_vec, node_deg_mat, lap_mat, k_eig_val, k_eig_vec, num_nodes, norm_n, batch_idx)

            loss = loss_fn(out_model, batched_graph.y.to(device))
            
            loss.backward()
            optimizer.step()

            epoch_train_mae += loss.item() 

        epoch_train_mae /= (idx + 1)
        return epoch_train_mae, optimizer

