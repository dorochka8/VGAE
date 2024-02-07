import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from config import config


class GVAE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder1 = GCNConv(in_channels=in_channels,      out_channels=config['hidden'])
        self.encoder2 = GCNConv(in_channels=config['hidden'], out_channels=config['latent'])

        self.mu       = GCNConv(in_channels=config['latent'], out_channels=out_channels)
        self.log_std  = GCNConv(in_channels=config['latent'], out_channels=out_channels)


    def reparametrization(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mu + eps * std


    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    
    def forward(self, x, edge_index):
        x = F.relu(self.encoder1(x, edge_index))
        x = self.encoder2(x, edge_index)

        mu = self.mu(x, edge_index)
        log_std = self.log_std(x, edge_index)

        z = self.reparametrization(mu, log_std)

        return z, mu, log_std
