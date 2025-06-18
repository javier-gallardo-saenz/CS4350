from filters import create_filter_list
from mlp import MLP
from torch_geometric.utils import to_dense_adj
import torch
import torch.nn as nn
from torch.nn import ReLU # Assuming ReLU is the activation used in params

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, filter_list, activation=None):
        super().__init__()

        self.K = len(filter_list) - 1
        self.filters = filter_list # This will be updated in forward pass of GCNNalpha
        self.activation = activation

        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(in_channels, out_channels))
            for _ in range(self.K + 1)
        ])

        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
        nn.init.zeros_(self.bias)

    def forward(self, X):
        N, _ = X.shape
        out = torch.zeros(N, self.weights[0].size(1), device=X.device)

        for k, Lk in enumerate(self.filters):
            out += Lk @ X @ self.weights[k]

        out += self.bias

        if self.activation is not None:
            out = self.activation(out)

        return out

class GCNNalpha(nn.Module):
    def __init__(
        self,
        dims: list,
        output_dim: int,
        degrees: list,
        activations: list, # Changed to list to hold activation functions for each layer
        gso_generator: callable,
        alpha: float = 0.5,
        readout_dims=None,
        apply_readout: bool = True
    ):
        super().__init__()

        self.apply_readout = apply_readout

        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.gso_generator = gso_generator

        # layers
        self.layers = nn.ModuleList()
        # Store degrees and activations separately since they are not nn.Module
        self.layer_degrees = []

        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            degree = degrees[i]
            activation_fn = activations[i] # Get the specific activation for this layer

            # Pass activation to ConvLayer so it handles it internally
            conv_layer = ConvLayer(in_dim, out_dim, [torch.eye(1)] * (degree + 1), activation=activation_fn)
            self.layers.append(conv_layer)
            self.layer_degrees.append(degree)

        # readout MLP
        if readout_dims is not None:
            self.readout = MLP([dims[-1]] + readout_dims)
        else:
            self.readout = None

        # default linear output mapping
        self.output_lin = nn.Linear(dims[-1], output_dim, bias=True)

    def forward(self, X, edge_index):
        A = to_dense_adj(edge_index).squeeze(0)      # Convert edge_index to dense adjacency matrix
        num_nodes = (A.sum(dim=1) != 0).sum().item()
        A = A[:num_nodes, :num_nodes]
        S = self.gso_generator(A, self.alpha)

        x = X
        for i, conv_layer in enumerate(self.layers):
            degree = self.layer_degrees[i]
            filters = create_filter_list(S, degree)
            conv_layer.filters = filters
            x = conv_layer(x)

        # apply readout or default linear mapping to output_dim
        if self.apply_readout:
            if self.readout is not None:
                x = self.readout(x)
            else:
                x = self.output_lin(x)
            return x.squeeze(-1) if x.dim() > 1 and x.size(-1) == 1 else x
        return x