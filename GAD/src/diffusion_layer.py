
import torch
import torch.nn as nn
import inspect

# Helper function

def get_mask(k, batch_num_nodes, num_nodes, device):
    mask = torch.zeros(num_nodes, k*len(batch_num_nodes)).to(device)
    partial_n = 0
    partial_k = 0
    for n in batch_num_nodes:
        mask[partial_n: partial_n + n, partial_k: partial_k + k] = 1
        partial_n = partial_n + n
        partial_k = partial_k + k
    return mask

#-----------------------------------------------
# Diffusion Layer from DAG, d/dt X = D^{1}LX
#-----------------------------------------------

class Diffusion_layer(nn.Module):

    def __init__(self, width, method, k, device):
        super().__init__()

        self.width = width
        self.method = method
        self.k = k
        self.device = device
        self.relu = nn.LeakyReLU()

        self.diffusion_time = nn.Parameter(torch.Tensor(self.width)) # num_channels

        nn.init.constant_(self.diffusion_time, 0.0)

    def forward(self, node_fts, node_deg_vec, node_deg_mat, operator, k_eig_val, k_eig_vec, num_nodes, batch_idx):


        with torch.no_grad():

            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        if self.method == 'spectral':

            _, indices_of_each_graph = torch.unique(batch_idx, return_counts=True)
            
            indices_of_each_graph = indices_of_each_graph.to(self.device)

            batch_size = indices_of_each_graph.shape[0]

            mask = get_mask(self.k, indices_of_each_graph.tolist(), num_nodes, self.device)
            
            k_eig_vec_ = k_eig_vec.repeat(1, batch_size)

            k_eig_vec_ = k_eig_vec_ * mask

            basis_T = k_eig_vec_.transpose(-2, -1)

            x_spec = torch.matmul(basis_T, node_fts * node_deg_vec)

            time = self.diffusion_time

            diffusion_coefs = torch.exp(-k_eig_val.unsqueeze(-1) * time.unsqueeze(0))

            x_diffuse_spec = diffusion_coefs * x_spec

            x_diffuse      = torch.matmul(k_eig_vec_, x_diffuse_spec)


        elif self.method == 'implicit':
                

            mat_ = operator.unsqueeze(0).expand(self.width, num_nodes, num_nodes).clone()

            mat_ *= self.diffusion_time.unsqueeze(-1).unsqueeze(-1)

            mat_ += node_deg_mat.unsqueeze(0)


            cholesky_factors = torch.linalg.cholesky(mat_)

            cholesky_decomp = node_fts * node_deg_vec

            cholesky_decomp_T = torch.transpose(cholesky_decomp, 0, 1).unsqueeze(-1)


            final_sol = torch.cholesky_solve(cholesky_decomp_T, cholesky_factors)

            x_diffuse = torch.transpose(final_sol.squeeze(-1), 0, 1)


        x_diffuse = self.relu(x_diffuse)

        return x_diffuse



#----------------------------------------------------------------------------------------------------
# Diffusion Layer for Hub Laplacian (d/dt X = L_{\alhpa} X)
#                     /Adv-Diff Operator (d/dt X = (\gamma_{a} L^{*}_{\alpha} + \gamma_{d} L) X )
# with fixed \gamma_{d}, \gamma_{a}, \alpha
#----------------------------------------------------------------------------------------------------

class Diffusion_layer_DegOperators(nn.Module):

    def __init__(self, width, method, k, device):
        super().__init__()

        self.width = width
        self.method = method
        self.k = k
        self.device = device
        self.relu = nn.LeakyReLU()

        self.diffusion_time = nn.Parameter(torch.Tensor(self.width))  # num_channels

        nn.init.constant_(self.diffusion_time, 0.0)

    def forward(self, node_fts, node_deg_vec, node_deg_mat, operator, k_eig_val, k_eig_vec, num_nodes, batch_idx):

        with torch.no_grad():

            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        mat_ = operator.unsqueeze(0).expand(self.width, num_nodes, num_nodes).clone()

        mat_ *= self.diffusion_time.unsqueeze(-1).unsqueeze(-1)

        cholesky_factors = torch.linalg.cholesky(mat_)

        cholesky_decomp = node_fts

        cholesky_decomp_T = torch.transpose(cholesky_decomp, 0, 1).unsqueeze(-1)

        final_sol = torch.cholesky_solve(cholesky_decomp_T, cholesky_factors)

        x_diffuse = torch.transpose(final_sol.squeeze(-1), 0, 1)

        x_diffuse = self.relu(x_diffuse)

        return x_diffuse


#----------------------------------------------------------------------------------------------------
# Diffusion Layer for Hub Laplacian (d/dt X = L_{\alhpa} X)
#                     /Adv-Diff Operator (d/dt X = (\gamma_{a} L^{*}_{\alpha} + \gamma_{d} L) X )
# with LEARNABLE \gamma_{d}, \gamma_{a}, \alpha
#----------------------------------------------------------------------------------------------------

class Diffusion_layer_LearnableDegOperators(nn.Module):

    def __init__(self, width, method, k, device):
        super().__init__()

        self.width = width
        self.method = method
        self.k = k
        self.device = device
        self.relu = nn.LeakyReLU()

        self.diffusion_time = nn.Parameter(torch.Tensor(self.width))  # num_channels
        self.gamma_adv = nn.Parameter(torch.Tensor(1))
        self.gamma_diff = nn.Parameter(torch.Tensor(1))
        self.alpha = nn.Parameter(torch.Tensor(1))

        nn.init.constant_(self.diffusion_time, 0.0)
        nn.init.constant_(self.gamma_adv, 0.5)
        nn.init.constant_(self.gamma_diff, 0.5)
        nn.init.constant_(self.alpha, 0.0)

    def forward(self, node_fts, node_deg_vec, node_deg_mat, operator, k_eig_val, k_eig_vec, num_nodes, batch_idx):
        #make sure operator is a function
        sig = inspect.signature(operator)
        if len(sig.parameters) != 3:
            raise ValueError("The operator function must take exactly 3 arguments: gamma_adv, gamma_diff, alpha.")

        with torch.no_grad():

            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        mat_ = operator(self.gamma_adv, self.gamma_diff, self.alpha).unsqueeze(0).expand(self.width, num_nodes, num_nodes).clone()

        mat_ *= self.diffusion_time.unsqueeze(-1).unsqueeze(-1)

        cholesky_factors = torch.linalg.cholesky(mat_)

        cholesky_decomp = node_fts

        cholesky_decomp_T = torch.transpose(cholesky_decomp, 0, 1).unsqueeze(-1)

        final_sol = torch.cholesky_solve(cholesky_decomp_T, cholesky_factors)

        x_diffuse = torch.transpose(final_sol.squeeze(-1), 0, 1)

        x_diffuse = self.relu(x_diffuse)

        return x_diffuse