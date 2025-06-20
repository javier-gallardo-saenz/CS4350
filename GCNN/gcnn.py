from filters import create_filter_list
from mlp import MLP
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
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
        activations: list,
        gso_generator: callable,
        alpha: float = 0.5,
        learn_alpha: bool = True,  # Add this parameter
        reduction: str = 'max',
        readout_dims=None,
        apply_readout: bool = True
    ):
        super().__init__()

        self.reduction = reduction
        self.apply_readout = apply_readout

        self.learn_alpha = learn_alpha
        if learn_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))  # Non-trainable

        self.gso_generator = gso_generator

        self.layers = nn.ModuleList()
        self.layer_degrees = []

        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            degree = degrees[i]
            activation_fn = activations[i]

            conv_layer = ConvLayer(in_dim, out_dim, [torch.eye(1)] * (degree + 1), activation=activation_fn)
            self.layers.append(conv_layer)
            self.layer_degrees.append(degree)

        if readout_dims is not None:
            self.readout = MLP([dims[-1]] + readout_dims)
        else:
            self.readout = None

        self.output_lin = nn.Linear(dims[-1], output_dim, bias=True)


    def forward(self, X, batch, edge_index):
        adj = to_dense_adj(edge_index, batch)
        gsos = []
        for i in range(adj.size(0)):
            A_i = adj[i]
            num_nodes_i = (A_i.sum(dim=1) != 0).sum().item()
            A_i = A_i[:num_nodes_i, :num_nodes_i]
            S_i = self.gso_generator(A_i, self.alpha)
            gsos.append(S_i)

        S = torch.block_diag(*gsos)

        x = X
        for i, conv_layer in enumerate(self.layers):
            degree = self.layer_degrees[i]
            filters = create_filter_list(S, degree)
            conv_layer.filters = filters
            x = conv_layer(x)

        if self.reduction == 'sum':
            x = global_add_pool(x, batch)
        elif self.reduction == 'mean':
            x = global_mean_pool(x, batch)
        elif self.reduction == 'max':
            x = global_max_pool(x, batch)

        # apply readout or default linear mapping to output_dim
        if self.apply_readout:
            if self.readout is not None:
                x = self.readout(x)
            else:
                x = self.output_lin(x)
            return x.squeeze(-1) if x.dim() > 1 and x.size(-1) == 1 else x
        return x