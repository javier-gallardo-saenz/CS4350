import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from conv_layer import ConvLayer
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'GAD', 'src')))
from mlp import MLP

class GCNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, readout_dims, filters):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, hidden_channels, filters)
        self.conv2 = ConvLayer(hidden_channels, hidden_channels, filters)
        self.readout = MLP([hidden_channels] + readout_dims)

    
    def forward(self, X, batch):
        x = torch.relu(self.conv1(X))       # can I call it like this?
        x = torch.relu(self.conv2(X))
        x = global_mean_pool(x, batch)  # Aggregate node embeddings to graph embedding

        return self.readout(x).squeeze(-1)