import torch.nn as nn
import torch.nn.functional as F
from .mp_encoder import Mp_encoder
from .sc_encoder import *

from .contrast import Contrast
import torch
import random


class Hemdap(nn.Module):
    def __init__(self, hidden_dim, feats_dim_list1, feats_dim_list2, feat_drop, attn_drop, P1, P2, sample_rate,sample_rate1, nei_num, tau, lam,gamma):
        super(Hemdap, self).__init__()
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.fc_list2 = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                       for feats_dim in feats_dim_list2])
        # print(self.fc_list2)

        self.fc_list1 = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                       for feats_dim in feats_dim_list1])
        # print(self.fc_list1)
        for fc in self.fc_list1:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        for fc in self.fc_list2:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

        self.mp1 = Mp_encoder(P1, hidden_dim, attn_drop)
        self.sc1 = Sc_encoder(hidden_dim, sample_rate, nei_num, attn_drop)

        self.mp2 = Mp_encoder(P2, hidden_dim, attn_drop)
        self.sc2 = Sc_encoder(hidden_dim, sample_rate1, nei_num, attn_drop)

        self.contrast = Contrast(hidden_dim, tau, lam)

    def forward(self, feats1, feats2, mps1, mps2, nei_index1, nei_index2):

        h_all1 = []
        h_all2 = []
        for i in range(len(feats1)):
            h_all1.append(F.elu(self.feat_drop(self.fc_list1[i](feats1[i]))))
        for i in range(len(feats2)):
            h_all2.append(F.elu(self.feat_drop(self.fc_list2[i](feats2[i]))))

        z_mp1 = self.mp1(h_all1[0], mps1)
        z_sc1 = self.sc1(h_all1, nei_index1)
        z_mp2 = self.mp2(h_all2[0], mps2)
        z_sc2 = self.sc2(h_all2, nei_index2)

        return z_mp1, z_sc1, z_mp2, z_sc2

    def get_embeds(self, feats1, feats2, mps1, mps2, nei_index1, nei_index2):

        h_all1 = []
        h_all2 = []
        for i in range(len(feats1)):
            h_all1.append(F.elu(self.feat_drop(self.fc_list1[i](feats1[i]))))
        for i in range(len(feats2)):
            h_all2.append(F.elu(self.feat_drop(self.fc_list2[i](feats2[i]))))

        z_mp1 = self.mp1(h_all1[0], mps1)
        z_sc1 = self.sc1(h_all1, nei_index1)

        z_mp1 = (1-self.gamma)*z_mp1.detach()
        z_sc1 = self.gamma*z_sc1.detach()
        z1 =  torch.add(z_mp1, z_sc1)

        z_mp2 = self.mp2(h_all2[0], mps2)
        z_sc2 = self.sc2(h_all2, nei_index2)

        z_mp2 = (1-self.gamma)*z_mp2.detach()
        z_sc2 = self.gamma*z_sc2.detach()
        z2 =  torch.add(z_mp2, z_sc2)

        return z1, z2
