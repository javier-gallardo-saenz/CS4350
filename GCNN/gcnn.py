from filters import create_filter_list
from mlp import MLP

from torch_geometric.utils import to_dense_adj 
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, filter_list, activation=None):
        super().__init__()

        self.K = len(filter_list) - 1
        self.filters = filter_list
        self.activation = activation

        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(in_channels, out_channels) * (2 / (in_channels + out_channels))**0.5)
            for _ in range(self.K + 1)
        ])

    def forward(self, X):
        N, _ = X.shape
        out = torch.zeros(N, self.weights[0].size(1), device=X.device)

        for k, Lk in enumerate(self.filters):
            out += Lk @ X @ self.weights[k]

        if self.activation is not None:
            out = self.activation(out)

        return out

class GCNNalpha(nn.Module):
    def __init__(self, dims: list, degrees: list, activations: list,
                  gso_generator: callable, alpha: int= 0.5, pooling_fn=None, readout_dims=None):
        super().__init__()

        self.pooling_fn = pooling_fn
        self.gso_generator = gso_generator 
        self.alpha = nn.Parameter(torch.tensor(alpha))
        
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
            S_i = self.gso_generator(A_i, self.alpha) 
            gsos.append(S_i)
        
        # block diagonal matrix (we should sparify later)
        S = torch.block_diag(*gsos)
        x = X

        for conv_layer, degree, activation in self.layers:
            conv_layer.filters = create_filter_list(S, degree)  # create poly filters based on current S
            x = conv_layer(x)
            if activation is not None:
                x = activation(x)

        if self.pooling_fn is not None:
            x = self.pooling_fn(x, batch)
        else:
            # fallback default pooling
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, batch)

        if self.readout is not None:
            return self.readout(x).squeeze(-1)
        return x