import torch.nn as nn
import torch.nn.functional as F
from .mp_encoder import Mp_encoder
from .sc_encoder import *

from .contrast import Contrast
import torch
import random


# class Hemdap(nn.Module):
#     def __init__(self, hidden_dim, feats_dim_list1, feats_dim_list2, feat_drop, attn_drop,
#                  P1, P2, sample_rate, sample_rate1, nei_num, tau, lam, gamma):
#         super(Hemdap, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.gamma = gamma
#
#         # 初始化特征转换层
#         self.fc_list1 = nn.ModuleList([
#             nn.Linear(feats_dim, hidden_dim, bias=True)
#             for feats_dim in feats_dim_list1
#         ])
#         self.fc_list2 = nn.ModuleList([
#             nn.Linear(feats_dim, hidden_dim, bias=True)
#             for feats_dim in feats_dim_list2
#         ])
#         for fc in self.fc_list1:
#             nn.init.xavier_normal_(fc.weight, gain=1.414)
#         for fc in self.fc_list2:
#             nn.init.xavier_normal_(fc.weight, gain=1.414)
#
#         # 定义特征丢弃
#         if feat_drop > 0:
#             self.feat_drop = nn.Dropout(feat_drop)
#         else:
#             self.feat_drop = lambda x: x
#
#         # 定义元路径编码器和结构编码器
#         self.mp1 = Mp_encoder(P1, hidden_dim, attn_drop)
#         self.sc1 = Sc_encoder(hidden_dim, sample_rate, nei_num, attn_drop)
#         self.mp2 = Mp_encoder(P2, hidden_dim, attn_drop)
#         self.sc2 = Sc_encoder(hidden_dim, sample_rate1, nei_num, attn_drop)
#
#         self.decoder1 = Mp_decoder(1, hidden_dim, attn_drop)
#         self.decoder2 = Mp_decoder(1, hidden_dim, attn_drop)
#
#         # 定义对比损失
#         self.contrast = Contrast(hidden_dim, tau, lam)
#
#         dropout_rate = 0.5
#         self.link_pred_mlp = nn.Sequential(
#             nn.Linear(2*hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_dim, 1),
#             # nn.ReLU(),
#             # nn.Dropout(dropout_rate),
#             # nn.Linear(hidden_dim // 2, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, feats1, feats2, mps1, mps2, nei_index1, nei_index2):
#         # Feature transformation
#         h_all1 = [F.elu(self.feat_drop(self.fc_list1[i](feats1[i]))) for i in range(len(feats1))]
#         h_all2 = [F.elu(self.feat_drop(self.fc_list2[i](feats2[i]))) for i in range(len(feats2))]
#
#         # Masking strategy
#         masked_feats1, masked_indices1 = self.mask_nodes(h_all1[0])
#         masked_feats2, masked_indices2 = self.mask_nodes(h_all2[0])
#
#         # Generating embeddings for masked nodes
#         z_masked_mp1 = [self.mp1(masked_feats1, mps1[i]) for i in range(len(mps1))]
#         z_masked_mp2 = [self.mp2(masked_feats2, mps2[i]) for i in range(len(mps2))]
#
#         # Reconstructing the features of masked nodes
#         reconstructed_feats1 = [self.decoder1(z_masked_mp1[i], mps1[i]) for i in range(len(mps1))]
#         reconstructed_feats2 = [self.decoder2(z_masked_mp2[i], mps2[i]) for i in range(len(mps2))]
#
#         # Compute SCE loss for intra-view contrast (miRNA)
#         sce_loss_intra_mirna = 0
#         for i in range(len(mps1)):
#
#              sce_loss_intra_mirna += self.sce_loss(reconstructed_feats1[i], h_all1[0])
#
#         sce_loss_intra_mirna /= len(mps1) * (len(mps1) - 1) / 2
#
#         # Compute SCE loss for intra-view contrast (disease)
#         sce_loss_intra_disease = 0
#         for i in range(len(mps2)):
#
#             sce_loss_intra_disease += self.sce_loss(reconstructed_feats2[i], h_all1[0])
#         sce_loss_intra_disease /= len(mps2) * (len(mps2) - 1) / 2
#
#         # Total loss
#         loss = sce_loss_intra_mirna + sce_loss_intra_disease
#
#         # 获取元路径和结构编码的嵌入
#         z_mp1 = self.mp1(h_all1[0], mps1)
#         z_sc1 = self.sc1(h_all1, nei_index1)
#         z_mp2 = self.mp2(h_all2[0], mps2)
#         z_sc2 = self.sc2(h_all2, nei_index2)
#
#         return z_mp1, z_sc1,z_mp2,z_sc2, loss
#
#     def sce_loss(self, z1, z2, eta=2.0):
#         # Normalize embeddings
#         z1_norm = F.normalize(z1, p=2, dim=-1)
#         z2_norm = F.normalize(z2, p=2, dim=-1)
#
#         # Compute cosine similarity
#         sim = torch.sum(z1_norm * z2_norm, dim=-1)
#
#         # Compute SCE loss
#         loss = torch.mean((1 - sim) ** eta)
#
#         return loss
#
#     def mask_nodes(self, feats):
#         num_nodes = feats.size(0)
#         num_masked = int(num_nodes * 0.15)
#         masked_indices = torch.randperm(num_nodes)[:num_masked]
#
#         masked_feats = feats.clone()
#         masked_feats[masked_indices] = self.mask_value
#
#         return masked_feats, masked_indices
#
#
#     def get_link_prediction(self, embedding_u, embedding_v):
#         # 将两个节点的嵌入拼接后通过 MLP 预测链接概率
#
#         # x = self.film_fusion(embedding_u, embedding_v)
#         x = torch.cat([embedding_u, embedding_v], dim=1)
#         prob = self.link_pred_mlp(x)
#         return prob

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


class NMF(nn.Module):
    def __init__(self, latent_dim=64, max_iter=500, tol=1e-4):
        super(NMF, self).__init__()
        self.latent_dim = latent_dim
        self.max_iter = max_iter
        self.tol = tol  # 停止条件的阈值

    def forward(self, md_matrix):
        """
        使用非负矩阵分解（NMF）将关联矩阵md_matrix分解为miRNA和疾病的64维潜在因子表示。

        参数:
        - md_matrix: 关联矩阵，形状 [num_mirna, num_disease]

        返回:
        - h_mirna: miRNA的64维潜在因子表示，形状 [num_mirna, latent_dim]
        - h_disease: 疾病的64维潜在因子表示，形状 [latent_dim, num_disease]
        """
        num_mirna, num_disease = md_matrix.shape
        W = torch.rand(num_mirna, self.latent_dim, requires_grad=True, device=md_matrix.device)  # miRNA矩阵
        H = torch.rand(self.latent_dim, num_disease, requires_grad=True, device=md_matrix.device)  # 疾病矩阵

        optimizer = torch.optim.Adam([W, H], lr=1e-3)

        for iteration in range(self.max_iter):
            md_reconstructed = torch.matmul(W, H)
            loss = torch.norm(md_matrix - md_reconstructed, p='fro')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < self.tol:
                break

            W.data = torch.clamp(W.data, min=0)
            H.data = torch.clamp(H.data, min=0)

        return W, H  # W: [num_mirna, latent_dim], H: [latent_dim, num_disease]


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


# class SemanticAttention(nn.Module):
#     def __init__(self, hidden_dim):
#         super(SemanticAttention, self).__init__()
#         self.project = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, 1, bias=False)
#         )
#
#     def forward(self, z):
#         w = self.project(torch.stack(z, dim=1)).squeeze(-1)
#         beta = torch.softmax(w, dim=1)
#         return (torch.stack(z, dim=1) * beta.unsqueeze(-1)).sum(1)
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
