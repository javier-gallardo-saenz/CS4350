from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F

class GCNBaseline(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, activation=None):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))
        self.activation = activation if activation is not None else F.leaky_relu

    def forward(self, x, edge_index, eval_oversmoothing=False):
        embeddings_per_layer = [x] if eval_oversmoothing else None

        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.activation(x)
            if eval_oversmoothing:
                embeddings_per_layer.append(x)

        if eval_oversmoothing:
            return x, embeddings_per_layer  # Final output and all intermediate layers
        else:
            return x  # Final output only