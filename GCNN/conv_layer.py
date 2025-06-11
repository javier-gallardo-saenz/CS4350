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
