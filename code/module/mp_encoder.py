

import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.spmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)

class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax(dim=0)  # 将 dim 设置为 0
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.stack(beta, dim=0)  # 将 beta 堆叠成一个张量
        beta = self.softmax(beta)

        z_mp = torch.sum(torch.stack(embeds, dim=0) * beta.unsqueeze(-1), dim=0)
        return z_mp

class Mp_encoder(nn.Module):
    def __init__(self, P, hidden_dim, attn_drop):
        super(Mp_encoder, self).__init__()
        self.P = P
        self.node_level = nn.ModuleList([GCN(hidden_dim, hidden_dim) for _ in range(P)])
        self.att = Attention(hidden_dim, attn_drop)

    def uniform_random_mask(self, x, mask_ratio):
        N = x.size(0)
        mask_num = int(N * mask_ratio)
        mask_idx = torch.randperm(N)[:mask_num]
        mask = torch.zeros(N, dtype=torch.bool, device=x.device)
        mask[mask_idx] = True
        return mask

    def apply_mask(self, x, mask):
        return x * (~mask).float().unsqueeze(-1)

    def encode(self, h, mps, mask_ratio):
        mask = self.uniform_random_mask(h, mask_ratio)
        h_masked = self.apply_mask(h, mask)

        embeds = []
        for i in range(self.P):
            embeds.append(self.node_level[i](h_masked, mps[0]))  # 使用 mps[0] 而不是 mps[i]

        z_mp = self.att(embeds)
        return z_mp, mask

    def decode(self, z_mp, mps):
        embeds = []
        for i in range(self.P):
            embeds.append(self.node_level[i](z_mp, mps[0]))  # 使用 mps[0] 而不是 mps[i]

        x_recon = self.att(embeds)
        return x_recon

    def forward(self, h, mps, mask_ratio1, mask_ratio2):
        z_mp, mask1 = self.encode(h, mps, mask_ratio1)
        z_mp2, mask1 = self.encode(h, mps, 0)
        mask2 = self.uniform_random_mask(z_mp, mask_ratio2)
        z_mp_masked = self.apply_mask(z_mp, mask2)
        x_recon = self.decode(z_mp_masked, mps)
        return z_mp,z_mp2, x_recon


