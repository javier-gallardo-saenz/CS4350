import torch
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
from itertools import combinations
import torch.nn.functional as F
import torch.nn as nn
from operators import hub_laplacian
import matplotlib.pyplot as plt

# Dataset
dataset = KarateClub()
data = dataset[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# ------------------ Model Definitions ------------------ #

class GCNbaseline(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        torch.manual_seed(12345)
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(dataset.num_features, 4))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(4, 4))
        self.layers.append(GCNConv(4, 2))
        self.classifier = nn.Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        embeddings = []
        h = x
        for conv in self.layers:
            h = conv(h, edge_index)
            h = h.tanh()
            embeddings.append(h)
        out = self.classifier(h)
        return out, embeddings


class HubConv(nn.Module):
    def __init__(self, in_channels, out_channels, S: torch.Tensor):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=True)
        self.register_buffer('S', S)

    def forward(self, x):
        h = torch.matmul(self.S, x)
        return self.lin(h)


class HubGCN(nn.Module):
    def __init__(self, dataset, S: torch.Tensor, num_layers):
        super().__init__()
        torch.manual_seed(12345)
        self.layers = nn.ModuleList()
        self.layers.append(HubConv(dataset.num_features, 4, S))
        for _ in range(num_layers - 2):
            self.layers.append(HubConv(4, 4, S))
        self.layers.append(HubConv(4, 2, S))
        self.classifier = nn.Linear(2, dataset.num_classes)

    def forward(self, x):
        embeddings = []
        h = x
        for conv in self.layers:
            h = conv(h)
            h = h.tanh()
            embeddings.append(h)
        out = self.classifier(h)
        return out, embeddings

# ------------------ Cosine Similarity Function ------------------ #

def cos_similarity(node_features):
    N = node_features.shape[0]
    indices = list(combinations(range(N), 2))  # All unique node pairs
    similarities = []
    for i, j in indices:
        sim = F.cosine_similarity(node_features[i].unsqueeze(0), node_features[j].unsqueeze(0)).item()
        similarities.append(sim)
    avg_similarity = sum(similarities) / len(similarities)
    return avg_similarity

# ------------------ Training ------------------ #

def train_one_epoch(model, optimizer, use_edge_index=True):
    model.train()
    optimizer.zero_grad()
    if use_edge_index:
        out, _ = model(data.x, data.edge_index)
    else:
        out, _ = model(data.x)
    loss = crit(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, use_edge_index=True):
    model.eval()
    with torch.no_grad():
        if use_edge_index:
            out, _ = model(data.x, data.edge_index)
        else:
            out, _ = model(data.x)

        # Get predicted class by taking argmax
        pred = out.argmax(dim=1)

        # Calculate accuracy on all nodes (you can also restrict to test_mask if you have one)
        correct = (pred == data.y).sum().item()
        acc = correct / data.num_nodes

    return pred.cpu(), acc


# ------------------ Run Experiment ------------------ #
for alpha in [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]:
    print("Alpha: ", alpha)
    # Build Graph Shift Operator
    N = data.num_nodes
    adj = to_dense_adj(data.edge_index, max_num_nodes=N)[0]
    S = hub_laplacian(adj, alpha=alpha)

    num_layers = 2 
    baseline = GCNbaseline(num_layers=num_layers).to(device)
    hub = HubGCN(dataset, S, num_layers=num_layers).to(device)

    opt_base = torch.optim.Adam(baseline.parameters(), lr=0.01)
    opt_hub = torch.optim.Adam(hub.parameters(), lr=0.01)
    crit = torch.nn.CrossEntropyLoss()

    epochs = 100
    lhub = []
    lbase = []
    for epoch in range(epochs):
        l1 = train_one_epoch(baseline, opt_base, use_edge_index=True)
        l2 = train_one_epoch(hub, opt_hub, use_edge_index=False)
        lhub.append(l2)
        lbase.append(l1)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | GCN Loss: {l1:.4f} | Hub Loss: {l2:.4f}")

    """
    if alpha == 0.5:
        plt.plot(range(1, len(lbase) + 1), lbase, marker='o')
        plt.title('Training Loss Curve baseline')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

        plt.plot(range(1, len(lhub) + 1), lhub, marker='o')
        plt.title('Training Loss Curve hub')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
        """

    # ------------------ Cosine Similarity Evaluation ------------------ #

    baseline.eval()
    hub.eval()

    pred_base, acc_base = evaluate(baseline, use_edge_index=True)
    print(f"\nGCN Baseline Accuracy: {acc_base:.4f}")
    print(f"GCN Predictions: {pred_base.numpy()}")

    # Evaluate Hub GCN
    pred_hub, acc_hub = evaluate(hub, use_edge_index=False)
    print(f"\nHub GCN Accuracy: {acc_hub:.4f}")
    print(f"Hub GCN Predictions: {pred_hub.numpy()}")

    _, baseline_embeddings = baseline(data.x, data.edge_index)
    _, hub_embeddings = hub(data.x)

    for layer_idx, (h_base, h_hub) in enumerate(zip(baseline_embeddings, hub_embeddings), start=1):
        cos_base = cos_similarity(h_base)
        cos_hub = cos_similarity(h_hub)
        print(f"Layer {layer_idx}:")
        print(f"  GCN Mean Cosine Similarity: {cos_base:.4f}")
        print(f"  Hub Mean Cosine Similarity: {cos_hub:.4f}")
