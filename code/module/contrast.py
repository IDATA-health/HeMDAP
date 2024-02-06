import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def gclloss(self, z_mp, z_sc, pos):
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()

        matrix_mp2sc = matrix_mp2sc / (torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        lori_mp = -torch.log(matrix_mp2sc.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2mp.mul(pos.to_dense()).sum(dim=-1)).mean()
        return self.lam * lori_mp + (1 - self.lam) * lori_sc

    def forward(self, z_mp1, z_sc1, pos1, z_mp2, z_sc2, pos2, md_matrix):

        one_index = []
        zero_index = []
        for i in range(md_matrix.shape[0]):
            for j in range(md_matrix.shape[1]):
                if md_matrix[i][j] >= 1:
                    one_index.append([i, j])
                else:
                    zero_index.append([i, j])
        random.seed(4)
        random.shuffle(one_index)
        random.shuffle(zero_index)
        num_samples = len(one_index)
        zero_index = zero_index[: num_samples]

        first_elements1 = [item[0] for item in one_index]
        first_elements2 = [item[1] for item in one_index]

        two_elements1 = [item[0] for item in zero_index]
        two_elements2 = [item[1] for item in zero_index]

        one_features_mp1 = z_mp1[first_elements1]  # Get positive sample features from z_mp1
        one_features_sc1 = z_sc1[first_elements1]  # Get positive sample features from z_sc1
        one_features_mp2 = z_mp2[first_elements2]  # Get negative sample features from z_sc2
        one_features_sc2 = z_sc2[first_elements2]

        positive_features_m1 = one_features_mp1 + one_features_sc1
        positive_features_d1 = one_features_mp2 + one_features_sc2

        zero_features_mp1 = z_mp1[two_elements1]  # Get positive sample features from z_mp1
        zero_features_sc1 = z_sc1[two_elements1]  # Get positive sample features from z_sc1

        inner_products1 = []
        inner_products2 = []

        for i in range(len(positive_features_m1)):
            inner_product = torch.dot(positive_features_m1[i], positive_features_d1[i])
            inner_products1.append(inner_product)

        zero_features_mp2 = z_mp2[two_elements2]  # Get negative sample features from z_sc2
        zero_features_sc2 = z_sc2[two_elements2]

        neg_features_m2 = zero_features_mp1 + zero_features_sc1
        neg_features_d2 = zero_features_mp2 + zero_features_sc2

        for i in range(len(neg_features_m2)):
            inner_product = torch.dot(neg_features_m2[i], neg_features_d2[i])
            inner_products2.append(inner_product)

        tensor1 = torch.tensor(inner_products1, requires_grad=True)
        tensor2 = -torch.tensor(inner_products2, requires_grad=True)

        logsigmoid_tensor1 = torch.mean(F.logsigmoid(tensor1))
        logsigmoid_tensor2 = torch.mean(F.logsigmoid(tensor2))

        loss_main = -logsigmoid_tensor1.sum() + logsigmoid_tensor2.sum()

        loss1 = self.gclloss(z_mp1, z_sc1, pos1)
        loss2 = self.gclloss(z_mp2, z_sc2, pos2)

        combined_loss = loss_main + loss1 + loss2

        return combined_loss

