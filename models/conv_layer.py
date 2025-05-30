import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, filter_list):
        super().__init__()
        
        self.K = len(filter_list) - 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = filter_list

        self.weights = self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(in_channels, out_channels))
            for _ in range(self.K + 1)
        ])
        # initialize weights?    

    def forward(self, X):
        # the operator is a sparse matrix (e.g., hubs-laplacian)

        N, _ = X.shape
        out = torch.zeros(N, self.weights.size(2), device=X.device)  # Accumulate output

        for k, Lk in enumerate(self.filters):
            out += torch.matmul(Lk @ X, self.weights[k])

        out = F.relu(out)

        return out
