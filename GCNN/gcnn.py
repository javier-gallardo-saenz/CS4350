from filters import create_filter_list
import torch 
import torch.nn as nn
from conv_layer import ConvLayer
import os
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

    def forward(self, X, batch, A):
        x = X
        for layer in self.layers:
            x = layer(x)

        x = self.pooling(x, batch)

        if self.readout:
            x = self.readout(x).squeeze(-1)
        return x
