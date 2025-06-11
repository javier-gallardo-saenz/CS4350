import torch
import torch.nn as nn

class LearnableGSO(nn.Module):
    def __init__(self, make_operator: callable, alpha_init=0.5):
        """
        A: Adjacency matrix as a torch tensor.
        make_operator: Function to generate the GSO
        alpha_init: Initial value of the alpha parameter.
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))  # Learnable parameter
        self.make_operator = make_operator

    def forward(self, A):
        """
        Builds the GSO using the current learnable alpha.
        """
        return self.make_operator(A, self.alpha)

