import torch
import torch.nn as nn
import inspect


# Helper function

def get_mask(k, batch_num_nodes, num_nodes, device):
    mask = torch.zeros(num_nodes, k * len(batch_num_nodes)).to(device)
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

        self.diffusion_time = nn.Parameter(torch.Tensor(self.width))  # num_channels

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

            x_diffuse = torch.matmul(k_eig_vec_, x_diffuse_spec)


        elif self.method == 'implicit':

            sig = inspect.signature(operator)
            num_params = len(sig.parameters)

            if num_params == 1:
                # If the operator takes one argument, it must be alpha
                mat_ = operator(0.0)
            else:
                raise ValueError(
                    "The Laplacian must take 1 argument (alpha)")

            mat_ = mat_.unsqueeze(0).expand(self.width, num_nodes, num_nodes).clone()

            mat_ *= self.diffusion_time.unsqueeze(-1).unsqueeze(-1)

            mat_ += node_deg_mat.unsqueeze(0)

            cholesky_factors = torch.linalg.cholesky(mat_)

            cholesky_decomp = node_fts * node_deg_vec

            cholesky_decomp_T = torch.transpose(cholesky_decomp, 0, 1).unsqueeze(-1)

            final_sol = torch.cholesky_solve(cholesky_decomp_T, cholesky_factors)

            x_diffuse = torch.transpose(final_sol.squeeze(-1), 0, 1)

        else:
            raise ValueError(f"Unknown diffusion method: {self.method}. Expected 'spectral' or 'implicit'.")

        x_diffuse = self.relu(x_diffuse)

        return x_diffuse


#----------------------------------------------------------------------------------------------------
# Diffusion Layer for Hub Laplacian (d/dt X = L_{\alhpa} X)
#                     /Adv-Diff Operator (d/dt X = (\gamma_{a} L^{*}_{\alpha} + \gamma_{d} L) X )
# with fixed \gamma_{d}, \gamma_{a}, \alpha
#----------------------------------------------------------------------------------------------------

class Diffusion_layer_DegOperators(nn.Module):

    def __init__(self, width, method, k, device, alpha_0, gamma_diff_0, gamma_adv_0):
        super().__init__()

        self.width = width
        self.method = method
        self.k = k
        self.device = device
        self.relu = nn.LeakyReLU()
        self.alpha = alpha_0
        self.gamma_diff = gamma_diff_0
        self.gamma_adv = gamma_adv_0

        self.diffusion_time = nn.Parameter(torch.Tensor(self.width))  # num_channels

        nn.init.constant_(self.diffusion_time, 0.0)

    def forward(self, node_fts, node_deg_vec, node_deg_mat, operator, k_eig_val, k_eig_vec, num_nodes, batch_idx):

        with torch.no_grad():

            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        sig = inspect.signature(operator)
        num_params = len(sig.parameters)

        if num_params == 1:
            L_star = operator(self.alpha)
        elif num_params == 3:
            L_star = operator(self.gamma_adv, self.gamma_diff, self.alpha)
        else:
            raise ValueError(
                "The operator function must take either 1 argument (alpha) or "
                "3 arguments (gamma_adv, gamma_diff, alpha).")

        node_deg_vec_1d = node_deg_vec.squeeze(-1)

        # L_star is now (num_nodes, num_nodes)

        # --- Step 2: Compute D_inv (inverse of degree matrix) ---
        # node_deg_vec_1d is a 1D tensor (num_nodes,) with degrees >= 1.
        node_deg_inv_vec = 1.0 / node_deg_vec_1d  # Element-wise inverse, no need for epsilon given >=1 constraint
        D_inv = torch.diag_embed(node_deg_inv_vec)  # Create diagonal matrix from inverse degrees
        # D_inv is (num_nodes, num_nodes)

        # --- Step 3: Compute D_inv @ L_star ---
        # Expand D_inv and L_star for `self.width` (features/channels)
        # D_inv is typically considered universal across features, so we expand it.
        D_inv_expanded = D_inv.unsqueeze(0).expand(self.width, num_nodes, num_nodes)  # (W, N, N)
        L_star_expanded = L_star.unsqueeze(0).expand(self.width, num_nodes, num_nodes)  # (W, N, N)

        # Perform batch matrix multiplication for D_inv @ L_star
        D_inv_L_star = torch.bmm(D_inv_expanded, L_star_expanded)  # (self.width, num_nodes, num_nodes)

        # --- Step 4: Scale by self.diffusion_time (dt * D_inv @ L_star) ---
        dt_D_inv_L_star = D_inv_L_star * self.diffusion_time.unsqueeze(-1).unsqueeze(-1)
        # dt_D_inv_L_star is (self.width, num_nodes, num_nodes)

        # --- Step 5: Construct the matrix A = (I - dt * D_inv @ L_star) ---
        # Create a 2D identity matrix of size (num_nodes, num_nodes)
        identity_2d = torch.eye(num_nodes, dtype=L_star.dtype, device=L_star.device)

        # Expand the 2D identity matrix to match the first dimension (width)
        identity_matrix_expanded = identity_2d.unsqueeze(0).expand(self.width, num_nodes, num_nodes)
        # identity_matrix_expanded is (self.width, num_nodes, num_nodes)

        # A is (I - dt * D_inv @ L_star)
        A = identity_matrix_expanded - dt_D_inv_L_star
        # A is (self.width, num_nodes, num_nodes)

        # --- Step 6: Solve the linear system ---
        node_fts_for_solve = node_fts.transpose(0, 1).unsqueeze(-1)
        Solved_x = torch.linalg.solve(A, node_fts_for_solve)

        # --- Step 9: Reshape back and apply ReLU ---
        x_diffuse_squeezed = Solved_x.squeeze(-1)  # (self.width, num_nodes)
        # Transpose to get (num_nodes, self.width)
        x_diffuse = self.relu(x_diffuse_squeezed.transpose(0, 1))

        return x_diffuse


#----------------------------------------------------------------------------------------------------
# Diffusion Layer for Hub Laplacian (d/dt X = L_{\alhpa} X)
#                     /Adv-Diff Operator (d/dt X = (\gamma_{a} L^{*}_{\alpha} + \gamma_{d} L) X )
# with LEARNABLE \gamma_{d}, \gamma_{a}, \alpha
#----------------------------------------------------------------------------------------------------

class Diffusion_layer_LearnableDegOperators(nn.Module):

    def __init__(self, width, method, k, device, alpha_0, gamma_diff_0, gamma_adv_0):
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
        nn.init.constant_(self.gamma_adv, gamma_diff_0)
        nn.init.constant_(self.gamma_diff, gamma_adv_0)
        nn.init.constant_(self.alpha, alpha_0)

    def forward(self, node_fts, node_deg_vec, node_deg_mat, operator, k_eig_val, k_eig_vec, num_nodes, batch_idx):
        with torch.no_grad():

            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        sig = inspect.signature(operator)
        num_params = len(sig.parameters)

        if num_params == 1:
            L_star = operator(self.alpha)
        elif num_params == 3:
            L_star = operator(self.gamma_adv, self.gamma_diff, self.alpha)
        else:
            raise ValueError(
                "The operator function must take either 1 argument (alpha) or "
                "3 arguments (gamma_adv, gamma_diff, alpha).")

        node_deg_vec_1d = node_deg_vec.squeeze(-1)

        # L_star is now (num_nodes, num_nodes)

        # --- Step 2: Compute D_inv (inverse of degree matrix) ---
        # node_deg_vec_1d is a 1D tensor (num_nodes,) with degrees >= 1.
        node_deg_inv_vec = 1.0 / node_deg_vec_1d # Element-wise inverse, no need for epsilon given >=1 constraint
        D_inv = torch.diag_embed(node_deg_inv_vec)  # Create diagonal matrix from inverse degrees
        # D_inv is (num_nodes, num_nodes)

        # --- Step 3: Compute D_inv @ L_star ---
        # Expand D_inv and L_star for `self.width` (features/channels)
        # D_inv is typically considered universal across features, so we expand it.
        D_inv_expanded = D_inv.unsqueeze(0).expand(self.width, num_nodes, num_nodes)  # (W, N, N)
        L_star_expanded = L_star.unsqueeze(0).expand(self.width, num_nodes, num_nodes)  # (W, N, N)

        # Perform batch matrix multiplication for D_inv @ L_star
        D_inv_L_star = torch.bmm(D_inv_expanded, L_star_expanded)  # (self.width, num_nodes, num_nodes)

        # --- Step 4: Scale by self.diffusion_time (dt * D_inv @ L_star) ---
        dt_D_inv_L_star = D_inv_L_star * self.diffusion_time.unsqueeze(-1).unsqueeze(-1)
        # dt_D_inv_L_star is (self.width, num_nodes, num_nodes)

        # --- Step 5: Construct the matrix A = (I - dt * D_inv @ L_star) ---
        # Create a 2D identity matrix of size (num_nodes, num_nodes)
        identity_2d = torch.eye(num_nodes, dtype=L_star.dtype, device=L_star.device)

        # Expand the 2D identity matrix to match the first dimension (width)
        identity_matrix_expanded = identity_2d.unsqueeze(0).expand(self.width, num_nodes, num_nodes)
        # identity_matrix_expanded is (self.width, num_nodes, num_nodes)

        # A is (I - dt * D_inv @ L_star)
        A = identity_matrix_expanded - dt_D_inv_L_star
        # A is (self.width, num_nodes, num_nodes)

        # --- Step 6: Solve the linear system ---
        node_fts_for_solve = node_fts.transpose(0, 1).unsqueeze(-1)
        Solved_x = torch.linalg.solve(A, node_fts_for_solve)

        # --- Step 9: Reshape back and apply ReLU ---
        x_diffuse_squeezed = Solved_x.squeeze(-1)  # (self.width, num_nodes)
        # Transpose to get (num_nodes, self.width)
        x_diffuse = self.relu(x_diffuse_squeezed.transpose(0, 1))

        return x_diffuse
