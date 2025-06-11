from filters import create_filter_list
from torch_geometric.utils import to_dense_adj 
import torch.nn as nn
from conv_layer import ConvLayer
import os
import torch 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'GAD', 'src')))
from mlp import MLP

class GeneralPolyGNN(nn.Module):
    def __init__(self, dims: list, degrees: list, activations: list,
                  gso_generator: nn.Module, readout_dims=None):
        super().__init__()

        self.gso_generator = gso_generator 

        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            degree = degrees[i]

            # The GSO will be generated on the fly in forward pass
            conv_layer = ConvLayer(in_dim, out_dim, None)
            self.layers.append((conv_layer, degree, activations[i]))

        if readout_dims is not None:
            self.readout = MLP([dims[-1]] + readout_dims)
        else:
            self.readout = None

    def forward(self, X, batch, edge_index):
        """
        base_operator: the initial graph structure (could be Laplacian, adjacency, etc.)
        """
        adj = to_dense_adj(edge_index, batch)
        gsos = []
        num_nodes_per_graph = []

        for i in range(adj.size(0)):
            A_i = adj[i]
            num_nodes_i = (A_i.sum(dim=1) != 0).sum().item()
            num_nodes_per_graph.append(num_nodes_i)
            A_i = A_i[:num_nodes_i, :num_nodes_i]  # crop to real adj since torch pads
            S_i = self.gso_generator(A_i) 
            gsos.append(S_i)
        
        # sparse block diagonal matrix
        S = torch.block_diag(*gsos)
        x = X

        for conv_layer, degree, activation in self.layers:
            filter_list = create_filter_list(S, degree)  # Create polynomial filters based on current S
            conv_layer.filters = filter_list  # Dynamically set filters
            x = conv_layer(x)
            if activation is not None:
                x = activation(x)

        x = self.pooling(x, batch)

        if self.readout is not None:
            return self.readout(x).squeeze(-1)
        return x
    
    def pooling(self, x, batch):
        from torch_geometric.nn import global_mean_pool
        return global_mean_pool(x, batch)
