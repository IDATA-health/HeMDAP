
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam, num_negatives=10):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau  # 温度参数
        self.lam = lam  # 平衡GCL损失的系数
        self.num_negatives = num_negatives  # 每个miRNA的负样本数量
        self.eta = 2
        # 初始化投影层的权重
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        """
        计算两个向量集合的相似度矩阵。
        """
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def sce_loss(self, z_mp1, pos1):
        """
        计算缩放余弦误差（SCE）。
        z_mp1: 元路径的嵌入 (dense tensor)
        pos1: 正样本对的标签 (sparse COO tensor)
        """
        # 归一化嵌入向量
        z_mp1_norm = F.normalize(z_mp1, p=2, dim=1)

        # 计算余弦相似度，矩阵乘法
        cos_sim = torch.matmul(z_mp1_norm, z_mp1_norm.t())  # (batch_size, batch_size)

        # 从稀疏矩阵中提取正样本对的索引
        pos_indices = pos1.coalesce().indices()  # 取出稀疏矩阵的索引，形状为 (2, nnz)

        # 提取正样本对的余弦相似度
        pos_sim = cos_sim[pos_indices[0], pos_indices[1]]  # 提取正样本对的相似度

        # 防止 (1 - pos_sim) 中产生极小值
        epsilon = 1e-8
        pos_sim_stable = torch.clamp(1 - pos_sim, min=epsilon)  # 避免负数或零值

        # 计算缩放余弦误差 (SCE)
        sce_loss = torch.mean(pos_sim_stable ** self.eta)

        return sce_loss

    def gclloss(self, z_mp, z_sc, pos):
        """
        计算无监督对比学习（GCL）的损失。
        """
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()

        matrix_mp2sc = matrix_mp2sc / (torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        lori_mp = -torch.log(matrix_mp2sc.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2mp.mul(pos.to_dense()).sum(dim=-1)).mean()

        return self.lam * lori_mp + (1 - self.lam) * lori_sc

    def scl_loss(self, z_mp, z_sc, positive_pairs, negative_pairs):
        """
        有监督对比学习损失（SCL）。
        """
        m_indices = [pair[0] for pair in positive_pairs]
        d_indices = [pair[1] for pair in positive_pairs]

        # 获取正样本对的嵌入
        e_m = z_mp[m_indices]  # 形状: (N, D)
        e_d = z_sc[d_indices]  # 形状: (N, D)

        # 计算正样本对的相似度
        sim_pos = F.cosine_similarity(e_m, e_d)  # 形状: (N,)

        # 初始化负样本相似度的张量
        sim_neg = torch.zeros_like(sim_pos)

        # 遍历每个正样本对来计算负样本相似度
        for i, negatives in enumerate(negative_pairs):
            if len(negatives) == 0:
                sim_neg[i] = -1e9  # 没有负样本时，避免NaN
            else:
                e_q = z_sc[negatives]  # 形状: (K, D)
                sim_q = F.cosine_similarity(e_m[i].unsqueeze(0), e_q)  # 形状: (K,)
                sim_neg[i] = torch.logsumexp(sim_q / self.tau, dim=0)

        # 计算SCL损失
        loss = -F.logsigmoid(sim_pos / self.tau) + sim_neg
        loss = loss.mean()

        return loss

    def forward(self, z_mp1, z_sc1, pos1, z_mp2, z_sc2, pos2, md_matrix,gamma):
        """
        计算总损失：GCL + SCL。
        """
        # 计算GCL损失
        loss1 = self.gclloss(z_mp1, z_sc1, pos1)
        loss2 = self.gclloss(z_mp2, z_sc2, pos2)

        sce_loss1 = self.sce_loss(z_mp1, pos1)
        sce_loss2 = self.sce_loss(z_mp2, pos2)

        # ------------------- 获取正样本和负样本 -------------------
        # positive_pairs, negative_pairs = self.get_positive_negative_samples(md_matrix)
        z_1 = (1 - gamma) * z_mp1 + gamma * z_sc1
        z_2 = (1 - gamma) * z_mp2 + gamma * z_sc2
        #
        #
        # positive_pairs1, negative_pairs1 = self.get_positive_negative_samples(md_matrix.T)
        # loss_scl1 = self.scl_loss(z_2, z_1, positive_pairs1, negative_pairs1)
        # # 计算SCL损失
        # loss_scl = self.scl_loss(z_1, z_2, positive_pairs, negative_pairs)
        # 假设 z_mp, z_sc 是你的 miRNA 和疾病嵌入向量，md_matrix 是关联矩阵
        loss_scl = self.optimized_infonce_loss(z_1, z_2, md_matrix, num_negatives=100, tau=0.1)

        # 组合总损失
        combined_loss = loss1 + loss2 + loss_scl

        return combined_loss

    def optimized_infonce_loss(self, z_mp, z_sc, md_matrix, num_negatives=100, tau=0.4):
        """
        优化的 InfoNCE 损失计算函数，支持 CUDA 和 CPU 操作。

        参数:
        - z_mp: miRNA 的嵌入向量 (N, D)
        - z_sc: 疾病的嵌入向量 (M, D)
        - md_matrix: miRNA 和疾病的关联矩阵 (N, M)
        - num_negatives: 每个正样本对应的负样本数量，默认为100
        - tau: 温度参数，默认为0.1

        返回:
        - InfoNCE 损失值
        """
        # 确保所有输入都是 PyTorch 张量并在同一设备上
        device = z_mp.device
        if isinstance(z_mp, np.ndarray):
            z_mp = torch.from_numpy(z_mp).to(device)
        if isinstance(z_sc, np.ndarray):
            z_sc = torch.from_numpy(z_sc).to(device)
        if isinstance(md_matrix, np.ndarray):
            md_matrix = torch.from_numpy(md_matrix).to(device)

        z_sc = z_sc.to(device)
        md_matrix = md_matrix.to(device).float()

        N, M = md_matrix.shape
        D = z_mp.shape[1]

        # 找到所有正样本对
        positive_pairs = torch.nonzero(md_matrix, as_tuple=True)
        num_positives = positive_pairs[0].size(0)

        # 计算所有可能对的相似度
        all_similarities = torch.mm(z_mp, z_sc.t())  # (N, M)

        # 获取正样本对的相似度
        positive_similarities = all_similarities[positive_pairs[0], positive_pairs[1]]  # (num_positives,)

        # 为每个正样本选择负样本
        negative_mask = ~(md_matrix.bool())
        negative_similarities = all_similarities[negative_mask]

        # 计算每个 miRNA 的负样本数量
        negative_counts = (M - md_matrix.sum(dim=1)).long()
        max_negatives = negative_counts.max().item()

        # 创建一个填充了 -inf 的张量来存储负样本相似度
        padded_negative_similarities = torch.full((N, max_negatives), float('-inf'), device=device)

        # 填充实际的负样本相似度
        for i in range(N):
            padded_negative_similarities[i, :negative_counts[i]] = negative_similarities[
                                                                   negative_counts[:i].sum():negative_counts[
                                                                                             :i + 1].sum()]

        # 对每个miRNA，随机选择固定数量的负样本
        selected_negative_similarities = []
        for i in range(N):
            available_negatives = padded_negative_similarities[i, :negative_counts[i]]
            num_available = available_negatives.size(0)
            if num_available >= num_negatives:
                selected = available_negatives[torch.randperm(num_available)[:num_negatives]]
            else:
                selected = torch.cat([available_negatives,
                                      available_negatives[
                                          torch.randint(num_available, (num_negatives - num_available,))]
                                      ])
            selected_negative_similarities.append(selected)

        selected_negative_similarities = torch.stack(selected_negative_similarities)  # (N, num_negatives)

        # 计算 InfoNCE 损失
        positive_similarities = positive_similarities / tau
        negative_similarities = selected_negative_similarities[positive_pairs[0]] / tau

        numerator = torch.exp(positive_similarities)
        denominator = numerator + torch.sum(torch.exp(negative_similarities), dim=1)

        losses = -torch.log(numerator / denominator)
        return torch.mean(losses)
