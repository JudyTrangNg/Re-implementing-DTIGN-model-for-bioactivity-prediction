# dtign_model.py ===== Ver2 ====
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool


class Embedding(torch.nn.Module):
    def __init__(self, num_rbf=64, dmin=1.0, dmax=6.0, powers=[1, -2, -6]):
        super().__init__()
        self.centers = {}
        self.widths = {}
        self.powers = powers
        for p in powers:
            centers = torch.linspace(dmin ** p, dmax ** p, num_rbf)
            width = (centers[1] - centers[0]) ** 2
            self.register_buffer(f'centers_{p}', centers)
            self.widths[p] = width

    def forward(self, d, power):
        centers = getattr(self, f'centers_{power}').to(d.device)
        width = self.widths[power]
        d_power = d ** power
        return torch.exp(-((d_power.unsqueeze(-1) - centers)** 2) / width)

class CovalentMessagePassing(MessagePassing):
    def __init__(self, in_channels, rbf_dim):
        super().__init__(aggr='add')
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + in_channels + rbf_dim, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

class NonCovalentMessagePassing(MessagePassing):
    def __init__(self, in_channels, rbf_dim):
        super().__init__(aggr='add')
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + 2 * rbf_dim, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

class DTIGNLayer(torch.nn.Module):
    def __init__(self, in_channels, bond_dim, rbf_embed, rbf_dim=64):
        super().__init__()
        self.rbf_embed = rbf_embed
        self.bond_proj = nn.Linear(bond_dim, in_channels)

        self.cov_mp = CovalentMessagePassing(in_channels, rbf_dim)
        self.ncov_mp = NonCovalentMessagePassing(in_channels, rbf_dim)

        self.update_cov = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU()
        )
        self.update_ncov = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU()
        )

    def forward(self, x, pos,
                edge_index_cov, edge_attr_cov,
                edge_index_ncov):
        # Covalent
        row_cov, col_cov = edge_index_cov
        dist_cov = (pos[row_cov] - pos[col_cov]).norm(dim=1)
        rbf_cov = self.rbf_embed(dist_cov, power=1)
        bond_feat = self.bond_proj(edge_attr_cov)
        edge_feat_cov = torch.cat([bond_feat, rbf_cov], dim=-1)
        m_cov = self.cov_mp(x, edge_index_cov, edge_feat_cov)

        # Non-covalent
        row_ncov, col_ncov = edge_index_ncov
        dist_ncov = (pos[row_ncov] - pos[col_ncov]).norm(dim=1)
        rbf2 = self.rbf_embed(dist_ncov, power=-2)
        rbf6 = self.rbf_embed(dist_ncov, power=-6)
        rbf_ncov = torch.cat([rbf2, rbf6], dim=-1)
        m_ncov = self.ncov_mp(x, edge_index_ncov, rbf_ncov)

        h_cov = self.update_cov(x + m_cov)
        h_ncov = self.update_ncov(x + m_ncov)
        return h_cov + h_ncov

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_output = None

    def forward(self, x):
        B, N, D = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, N, self.n_heads, 3 * self.d_k)
        qkv = qkv.permute(2, 0, 1, 3)

        q, k, v = torch.chunk(qkv, 3, dim=-1)

        scores = torch.matmul(q, k.transpose(-2,-1))/(self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)

        context = torch.matmul(attn, v).permute(1,2,0,3).reshape(B,N,D)

        self.attn_output = attn.mean(dim=0).mean(dim=1)
        return self.out_proj(context)

class DTIGN(torch.nn.Module):
    def __init__(self, in_dim=35, bond_dim=10, hidden_dim = 128, n_layers=3, n_heads=4):
        super().__init__()
        self.atom_proj = nn.Linear(in_dim, hidden_dim)
        self.rbf_embed = Embedding()
        self.layers = nn.ModuleList([
            DTIGNLayer(hidden_dim, bond_dim, self.rbf_embed)
            for _ in range(n_layers)
        ])
        self.attn = MultiHeadAttention(hidden_dim, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim,1)
        )

    def forward(self, data):
        x = self.atom_proj(data.x)
        for layer in self.layers:
            x = layer(
                x, data.pos,
                data.edge_index_intra, data.x_bond,
                data.edge_index_inter
            )

        if hasattr(data, "batch"):
            reps = global_mean_pool(x, data.batch)
            reps = reps.unsqueeze(1)
        else:
            reps = x.sum(dim=0, keepdim=True).unsqueeze(0)

        reps = self.attn(reps)
        self.attn_output = self.attn.attn_output
        final_rep = reps.mean(dim=1)
        return self.mlp(final_rep).view(-1)
        # return 7.0*torch.sigmoid(self.mlp(final_rep)).view(-1)
