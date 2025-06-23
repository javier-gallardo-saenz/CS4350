import torch
import torch.nn as nn

from aggregators import AGGREGATORS
from mlp import MLP
from scalers import SCALERS


########## DGN Layer Simple ############# 

class DGN_layer_Simple(nn.Module):
    def __init__(self, aux_hid_dim, graph_norm, batch_norm, aggregators, scalers, edge_fts, avg_d, D, device):
        super().__init__()
        self.aux_hid_dim = aux_hid_dim
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.scalers = scalers
        self.edge_fts = edge_fts
        self.device = device
        self.avg_d = avg_d
        self.D = D
        
        self.aggregators = []
        for agg in aggregators:
            agg_ = AGGREGATORS[agg](self.edge_fts, self.aux_hid_dim, self.device)
            self.aggregators.append(agg_)
            
        self.MLP_last = MLP([((len(self.aggregators)*len(self.scalers)+1))*self.aux_hid_dim, self.aux_hid_dim])

        self.relu = nn.ReLU()
        self.batchnorm_layer = nn.BatchNorm1d(self.aux_hid_dim)

    def forward(self, node_fts, edge_fts, edge_index, F_norm_edge, F_dig, node_deg_vec, norm_n):
                
        # aggregators and scalers
        output = torch.cat([aggregate(node_fts, edge_fts, edge_index, F_norm_edge, F_dig) for aggregate in self.aggregators], dim=1)
        
        if len(self.scalers) > 1:
            output = torch.cat([scale(output, D= node_deg_vec, avg_d=self.avg_d, device = self.device) for scale in self.scalers], dim=1)
        output = torch.cat([node_fts, output], dim = 1)


        output = self.MLP_last(output)
        
        if self.graph_norm:
            output = output*norm_n
            
        if self.batch_norm:
            output = self.batchnorm_layer(output)

        output = self.relu(output)
        

        return output
      
      
      
########## DGN Tower ############# 

class DGN_Tower(nn.Module):
    def __init__(self, aux_hid_dim, graph_norm, batch_norm, aggregators, scalers, edge_fts, avg_d, D, device, towers):
        super().__init__()
        self.aux_hid_dim = aux_hid_dim
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.scalers = scalers
        self.edge_fts = edge_fts
        self.device = device
        self.avg_d = avg_d
        self.D = D

        self.aggregators = []
        for agg in aggregators:
            agg_ = AGGREGATORS[agg](self.edge_fts, aux_hid_dim//towers, self.device)
            self.aggregators.append(agg_)

            
        self.MLP_last = MLP([((len(self.aggregators)*len(self.scalers)+1))* aux_hid_dim//towers, aux_hid_dim//towers])
        
        self.batchnorm_layer = nn.BatchNorm1d(aux_hid_dim//towers)

    def forward(self, node_fts, edge_fts, edge_index, F_norm_edge, F_dig, node_deg_vec, norm_n):
        
        
        # aggregators and scalers

        output = torch.cat([aggregate(node_fts, edge_fts, edge_index, F_norm_edge, F_dig)
                            for aggregate in self.aggregators], dim=1)
    
        if len(self.scalers) > 1:
            output = torch.cat([scale(output, D=node_deg_vec, avg_d=self.avg_d, device=self.device)
                                for scale in self.scalers], dim=1)

 
        output = torch.cat([node_fts, output], dim=1)

        output = self.MLP_last(output)
        
        if self.graph_norm:
            output = output*norm_n
            
        if self.batch_norm:
            output = self.batchnorm_layer(output)


        return output
      

########## DGN Layer Tower ############# 

class DGN_layer_Tower(nn.Module):
    def __init__(self, aux_hid_dim, graph_norm, batch_norm, aggregators, scalers, edge_fts, avg_d, D, device, towers):
        super().__init__()
        self.aux_hid_dim = aux_hid_dim
        
        self.input_tower = aux_hid_dim // towers
        self.output_tower = aux_hid_dim // towers
        
        self.towers = nn.ModuleList()
        
        for _ in range(towers):
            self.towers.append(DGN_Tower(aux_hid_dim=aux_hid_dim, graph_norm=graph_norm, batch_norm=batch_norm,
                                         aggregators=aggregators, scalers=scalers, edge_fts=edge_fts,
                                         avg_d=avg_d, D=D, device=device, towers=towers))
    
        self.aggregators = aggregators
            
        self.MLP_last = MLP([aux_hid_dim, aux_hid_dim])
        
        self.relu = nn.LeakyReLU()

    def forward(self, node_fts, edge_fts, edge_index, F_norm_edge, F_dig, node_deg_vec, norm_n):
                
     
        output = torch.cat([tower(node_fts[:, n_tower * self.input_tower: (n_tower + 1) * self.input_tower],
                                  edge_fts, edge_index, F_norm_edge, F_dig, node_deg_vec, norm_n)
                               for n_tower, tower in enumerate(self.towers)], dim=1)
        
        output = self.MLP_last(output)
        output = self.relu(output)
        


        return output
