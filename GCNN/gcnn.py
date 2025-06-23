from filters import create_filter_list
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import to_dense_adj
import torch
import torch.nn as nn
from torch.nn import ReLU # Assuming ReLU is the activation used in params

class MLP(nn.Module):
    def __init__(self, dims: list, activation=None, bias=True):
        """
        dims: list of layer dimensions [input, hidden1, ..., output]
        activation: single activation function (applied after each hidden layer)
        bias: whether to include biases in linear layers
        """
        super().__init__()
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            if i < len(dims) - 2 and activation is not None:
                layers.append(activation)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, filter_list,
                 activation=None, use_bn=False, dropout_p=0.0):
        super().__init__()
        self.K = len(filter_list) - 1
        self.filters = filter_list
        self.activation = activation

        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(in_channels, out_channels))
            for _ in range(self.K + 1)
        ])
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.zeros_(self.bias)
        for w in self.weights:
            nn.init.xavier_uniform_(w)

        self.use_bn = use_bn
        if use_bn:
            # BatchNorm over the feature dimension
            self.bn = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else None

    def forward(self, X):
        # X: [N, in_channels]
        out = torch.zeros(X.size(0), self.bias.size(0), device=X.device)
        for k, Lk in enumerate(self.filters):
            out += Lk @ X @ self.weights[k]
        out += self.bias

        if self.activation is not None:
            out = self.activation(out)
        if self.use_bn:
            out = self.bn(out)         # normalize each feature channel
        if self.dropout is not None:
            out = self.dropout(out)    # randomly zero some features

        return out


class GCNNalpha(nn.Module):
    def __init__(
        self,
        dims: list,
        output_dim: int,
        hops: int,
        activation,
        gso_generator: callable,
        alpha: float = 0.5,
        learn_alpha: bool = True,
        pooling: str = 'max',
        readout_hidden_dims: list = None,   # renamed
        apply_readout: bool = True,
        use_bn=False,
        dropout_p=0.0
    ):
        super().__init__()

        self.reduction = pooling
        self.apply_readout = apply_readout
        self.hops = hops 

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

            conv_layer = ConvLayer(in_dim,
                out_dim,
                [torch.eye(1)] * (hops + 1),
                activation=activation,
                use_bn= use_bn,
                dropout_p = dropout_p)
            
            self.layers.append(conv_layer)

        if readout_hidden_dims is not None:
            self.readout = MLP([dims[-1]] + readout_hidden_dims,
                               activation=activation)
            last_readout_size = readout_hidden_dims[-1]
        else:
            self.readout = None
            last_readout_size = dims[-1]

        self.output_lin = nn.Linear(last_readout_size, output_dim, bias=True)

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

        filters = create_filter_list(S, self.hops)

        for i, conv_layer in enumerate(self.layers):
            conv_layer.filters = filters
            x = conv_layer(x)

        if self.reduction == 'sum':
            x = global_add_pool(x, batch)
        elif self.reduction == 'mean':
            x = global_mean_pool(x, batch)
        elif self.reduction == 'max':
            x = global_max_pool(x, batch)

        # apply readout or default linear mapping to output_dim
        if self.apply_readout and self.readout is not None:
            x = self.readout(x)

        # always apply the final linear to get the correct output_dim
        if self.apply_readout:
            x = self.output_lin(x)
            # if it ends up with last‐dim 1, squeeze
            if x.dim() > 1 and x.size(-1) == 1:
                x = x.squeeze(-1)
        return x