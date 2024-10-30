import torch.nn as nn
import torch.nn.functional as F
from .mp_encoder import Mp_encoder
from .sc_encoder import *

from .contrast import Contrast
import torch
import random


class Hemdap(nn.Module):
    def __init__(self, hidden_dim, feats_dim_list1, feats_dim_list2, feat_drop, attn_drop,
                 P1, P2, sample_rate, sample_rate1, nei_num, tau, lam, gamma, mask_ratio1=0.1, mask_ratio2=0.1):
        super(Hemdap, self).__init__()
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.P1 = P1
        self.P2 = P2
        self.mask_ratio1 = 0.1
        self.mask_ratio2 = 0.1
        # 定义 W_r 映射矩阵
        self.W_r = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 特征转换层
        self.fc_list1 = nn.ModuleList([
            nn.Linear(feats_dim, hidden_dim, bias=True)
            for feats_dim in feats_dim_list1
        ])
        self.fc_list2 = nn.ModuleList([
            nn.Linear(feats_dim, hidden_dim, bias=True)
            for feats_dim in feats_dim_list2
        ])
        self.fc_list3 = nn.ModuleList([
            nn.Linear(feats_dim, hidden_dim, bias=True)
            for feats_dim in feats_dim_list1
        ])

        # 分别初始化 fc_list1 和 fc_list2 的权重
        for fc in self.fc_list1:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        for fc in self.fc_list2:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else lambda x: x

        # 更新的Mp_encoder
        self.mp1 = Mp_encoder(P1, hidden_dim, attn_drop)
        self.mp2 = Mp_encoder(P2, hidden_dim, attn_drop)

        # 保留原有的结构编码器
        self.sc1 = Sc_encoder(hidden_dim, sample_rate, nei_num, attn_drop)
        self.sc2 = Sc_encoder(hidden_dim, sample_rate1, nei_num, attn_drop)

        # 链接预测MLP
        self.link_pred_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.contrastive = ContrastiveLearning(hidden_dim)

        # 添加语义注意力层
        # 使用新的SemanticAttention类

        self.semantic_attention1 = SemanticAttention(hidden_dim, attn_drop)
        self.semantic_attention2 = SemanticAttention(hidden_dim, attn_drop)

    def forward(self, feats1, feats2, mps1, mps2, nei_index1, nei_index2, md_matrix):
        # 特征转换
        h_all1 = [F.elu(self.feat_drop(fc(feat))) for fc, feat in zip(self.fc_list1, feats1)]
        h_all2 = [F.elu(self.feat_drop(fc(feat))) for fc, feat in zip(self.fc_list2, feats2)]
        h_all3 = [F.elu(self.feat_drop(fc(feat))) for fc, feat in zip(self.fc_list3, feats1)]
        # 元路径编码和重构
        z_mp1_list = []
        recon_mp1_list = []
        for i in range(self.P1):
            z_mp1, z_mp3, recon_mp1 = self.mp1(h_all1[0], [mps1[i]], self.mask_ratio1, self.mask_ratio2)
            z_mp1_list.append(recon_mp1 + z_mp3)
            recon_mp1_list.append(recon_mp1)

        z_mp2_list = []
        recon_mp2_list = []
        for i in range(self.P2):
            z_mp2, z_mp4, recon_mp2 = self.mp2(h_all2[0], [mps2[i]], self.mask_ratio1, self.mask_ratio2)
            z_mp2_list.append(recon_mp2 + z_mp4)
            recon_mp2_list.append(recon_mp2)

        # 语义聚合
        z_mp1_agg = self.semantic_attention1(z_mp1_list)
        z_mp2_agg = self.semantic_attention2(z_mp2_list)

        # 计算对比损失
        contrast_loss1 = self.contrastive(h_all1[0], recon_mp1_list)
        contrast_loss2 = self.contrastive(h_all2[0], recon_mp2_list)
        contrast_loss = contrast_loss1 + contrast_loss2

        # 结构编码
        z_sc1 = self.sc1(h_all1, nei_index1)
        z_sc2 = self.sc2(h_all2, nei_index2)

        # 调用知识感知编码器，生成增强表示
        new_h_mirna, new_h_disease, knowledge_loss = self.knowledge_aware_encoder(h_all1[0], h_all2[0], z_mp1_agg,
                                                                                  md_matrix)
        new_h_disease1, new_h_mirna1, knowledge_loss1 = self.knowledge_aware_encoder(h_all2[0], h_all1[0], z_mp1_agg,
                                                                                     md_matrix.T)
        # 总损失
        total_loss = contrast_loss + knowledge_loss + knowledge_loss1

        return z_mp1_agg + new_h_mirna + new_h_mirna1 + h_all1[0], z_sc1, z_mp2_agg + new_h_disease1 + new_h_disease + \
               h_all2[0], z_sc2, total_loss

    def get_link_prediction(self, embedding_u, embedding_v):
        x = torch.cat([embedding_u, embedding_v], dim=1)
        prob = self.link_pred_mlp(x)
        return prob

    def knowledge_aware_encoder(self, h_mirna, h_disease, z_mp_agg, md_matrix):
        """
        知识感知编码器，生成增强的miRNA和疾病表示，并计算知识感知损失。

        参数:
        - h_mirna: miRNA节点的特征向量，形状 [num_mirna, hidden_dim]
        - h_disease: 疾病节点的特征向量，形状 [num_disease, hidden_dim]
        - z_mp_agg: 聚合后的元路径表示，形状 [num_mirna, meta_path_dim]
        - md_matrix: miRNA和疾病之间的关联矩阵，形状 [num_mirna, num_disease]

        返回:
        - new_h_mirna: 增强后的miRNA表示
        - new_h_disease: 增强后的疾病表示
        - knowledge_loss: 知识感知损失
        """
        # Step 1: 将 numpy.ndarray 转换为 torch.Tensor
        if isinstance(md_matrix, np.ndarray):
            md_matrix = torch.tensor(md_matrix, dtype=torch.float32).to(h_mirna.device)  # 确保它与输入数据在同一设备上

        # Step 2: 获取正三元组（等于1的位置为正样本）
        positive_indices = torch.nonzero(md_matrix == 1)  # 正样本 (miRNA, disease) 对
        u_indices = positive_indices[:, 0]  # miRNA索引
        j_indices = positive_indices[:, 1]  # 疾病索引

        # Step 3: 获取负三元组（等于0的位置为负样本）
        negative_indices = torch.nonzero(md_matrix == 0)  # 负样本 (miRNA, disease) 对

        # 随机选择与正样本数量相同的负样本
        num_positive_samples = len(positive_indices)
        negative_indices = negative_indices[torch.randperm(len(negative_indices))[:num_positive_samples]]

        u_neg = negative_indices[:, 0]  # 负样本的miRNA索引
        j_neg = negative_indices[:, 1]  # 负样本的疾病索引

        # Step 4: 获取节点和元路径的表示
        h_u = h_mirna[u_indices]  # 正样本的miRNA节点表示
        h_j = h_disease[j_indices]  # 正样本的疾病节点表示
        h_P = z_mp_agg[u_indices]  # 对应的元路径表示

        # 获取负样本的表示
        h_u_neg = h_mirna[u_neg]  # 负样本的miRNA节点表示
        h_j_neg = h_disease[j_neg]  # 负样本的疾病节点表示
        h_P_neg = z_mp_agg[u_neg]  # 负样本的元路径表示

        # Step 5: 计算正负样本的得分
        s_positive = torch.norm(self.W_r(h_u) + h_P - self.W_r(h_j), p=2, dim=1) ** 2
        s_negative = torch.norm(self.W_r(h_u_neg) + h_P_neg - self.W_r(h_j_neg), p=2, dim=1) ** 2

        # Step 6: 计算知识感知损失
        knowledge_loss = -torch.log(torch.sigmoid(s_negative - s_positive) + 1e-8).mean()

        # Step 7: 生成增强的节点表示
        new_h_mirna = self.W_r(h_mirna)  # 基于元路径的miRNA新表示
        new_h_disease = self.W_r(h_disease)  # 基于映射的疾病新表示

        return new_h_mirna, new_h_disease, knowledge_loss




class ContrastiveLearning(nn.Module):
    def __init__(self, hidden_dim, eta=1.1):
        super(ContrastiveLearning, self).__init__()
        self.eta = eta

    def sce_loss(self, x, z):
        x = F.normalize(x, p=2, dim=-1)
        z = F.normalize(z, p=2, dim=-1)
        return torch.mean((1 - torch.sum(x * z, dim=-1)).pow(self.eta))

    def forward(self, original_features, metapath_encodings):
        losses = []
        for encoding in metapath_encodings:
            losses.append(self.sce_loss(original_features, encoding))
        return sum(losses)


class SemanticAttention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(SemanticAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax(dim=-1)
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
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)

        z_agg = 0
        for i in range(len(embeds)):
            z_agg += embeds[i] * beta[i]
        return z_agg
