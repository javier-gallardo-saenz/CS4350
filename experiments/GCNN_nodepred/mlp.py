import torch.nn as nn

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
