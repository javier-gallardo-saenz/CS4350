############ GAD layer #####################
import torch
import torch.nn as nn

from aggregators import AGGREGATORS
from mlp import MLP
from scalers import SCALERS
from dgn_layer import DGN_layer_Simple, DGN_Tower, DGN_layer_Tower
from diffusion_layer import Diffusion_layer, Diffusion_layer_DegOperators, Diffusion_layer_LearnableDegOperators


class GAD_layer(nn.Module):
    def __init__(self, aux_hid_dim, graph_norm, batch_norm, dropout, aggregators, scalers, edge_fts, avg_d, D,
                 device, towers, type_net, residual,
                 use_diffusion, diffusion_type, diffusion_method, diffusion_param, k):
        super().__init__()

        #aggregators = [aggr for aggr in aggregators.split()]
        #scalers = [SCALERS[scale] for scale in scalers.split()]
        scalers = [SCALERS[scale] for scale in scalers]

        if type_net == 'simple':
            self.DGN_layer = DGN_layer_Simple(aux_hid_dim=aux_hid_dim, graph_norm=graph_norm, batch_norm=batch_norm,
                                              aggregators=aggregators, scalers=scalers, edge_fts=edge_fts,
                                              avg_d=avg_d, D=D, device=device)
        elif type_net == 'tower':
            self.DGN_layer = DGN_layer_Tower(aux_hid_dim=aux_hid_dim, graph_norm=graph_norm, batch_norm=batch_norm,
                                             aggregators=aggregators, scalers=scalers, edge_fts=edge_fts,
                                             avg_d=avg_d, D=D, device=device, towers=towers)
            
        self.dropout = dropout  
        self.use_diffusion = use_diffusion
        if self.use_diffusion:
            if diffusion_type == "standard":
                self.diffusion_layer = Diffusion_layer(aux_hid_dim, method=diffusion_method, k=k, device=device)
            elif diffusion_type == "degree_operators":
                self.diffusion_layer = Diffusion_layer_DegOperators(aux_hid_dim, method=diffusion_method, k=k,
                                                                    device=device,
                                                                    alpha_0=diffusion_param['alpha'],
                                                                    gamma_diff_0=diffusion_param['gamma_diff'],
                                                                    gamma_adv_0=diffusion_param['gamma_adv_0'])
            elif diffusion_type == "learnable_degree_operators":
                self.diffusion_layer = Diffusion_layer_LearnableDegOperators(aux_hid_dim, method=diffusion_method, k=k,
                                                                             device=device,
                                                                             alpha_0=diffusion_param['alpha'],
                                                                             gamma_diff_0=diffusion_param['gamma_diff'],
                                                                             gamma_adv_0=diffusion_param['gamma_adv_0'])

            self.MLP_last = MLP([2*aux_hid_dim, aux_hid_dim], dropout=False)
            
        self.residual = residual


    def forward(self, node_fts, edge_fts, edge_index, F_norm_edge, F_dig, node_deg_vec, node_deg_mat, lap_mat,
                k_eig_val, k_eig_vec, num_nodes, norm_n, batch_idx):

        if self.use_diffusion:
            diffusion_out   = self.diffusion_layer(node_fts, node_deg_vec, node_deg_mat, lap_mat, k_eig_val, k_eig_vec,
                                                   num_nodes, batch_idx)
            dgn_out         = self.DGN_layer(diffusion_out, edge_fts, edge_index, F_norm_edge, F_dig, node_deg_vec,
                                             norm_n)
            output          = torch.cat((diffusion_out, dgn_out), dim=1)
            output          = self.MLP_last(output)
        else:
            output          = self.DGN_layer(node_fts, edge_fts, edge_index, F_norm_edge, F_dig, node_deg_vec, norm_n)
            

        if self.residual:
            output   = node_fts + output
            
#         output = nn.functional.dropout(output, self.dropout, training=self.training)

        return output
