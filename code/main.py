import numpy
import scipy as sp
import torch
from matplotlib import pyplot as plt
from load_data import *
from params import *
from utils import *

import pandas as pd
from module import Hemdap
from module import Contrast
import warnings
import datetime
import pickle as pkl
import os
import random
import os
import gc
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.metrics import precision_score
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, auc
from numpy import *

import os
import gc
import xgboost as xgb
import numpy as np
from torch import nn, optim

from sklearn.metrics import precision_score
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
from numpy import *
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score

import copy
import torch as th

import numpy as np

from scipy.sparse import coo_matrix

import torch.nn.functional as F

import datetime

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(device)
else:
    device = torch.device("cpu")

args = model_params()
seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class Config(object):
    def __init__(self):
        self.fold = 5
        self.seed = 4


def calculate_metrics(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)  # Handle any NaN values that may arise

    # Find the threshold that gives the maximum F1 score
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]

    # Calculate AUPR
    aupr = auc(recall, precision)

    # Use the best threshold to generate binary predictions
    y_pred_binary = (y_pred > best_threshold).astype(int)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)

    return aupr, precision, recall, f1, best_threshold


if __name__ == "__main__":

    D = np.genfromtxt(r"../data/HMDD3.2.txttgitg")
    own_str = 'model'

    opt = Config()

    [row, col] = np.shape(D)
    print(np.shape(D))
    indexn = np.argwhere(D == 0)
    Index_zeroRow = indexn[:, 0]
    Index_zeroCol = indexn[:, 1]

    indexp = np.argwhere(D == 1)
    Index_PositiveRow = indexp[:, 0]
    Index_PositiveCol = indexp[:, 1]
    totalassociation = np.size(Index_PositiveRow)
    fold = int(totalassociation / 5)
    zero_length = np.size(Index_zeroRow)
    fold1 = int(zero_length / 5)
    cv_num = 5

    varauc = []
    AAuc_list1 = []

    for time in range(1, 2):

        Auc_per = []

        f1_score_per = []
        precision_per = []
        recall_per = []
        aupr_per = []

        np.random.seed(opt.seed)
        p = np.random.permutation(totalassociation)
        all_f = np.random.permutation(np.size(Index_zeroRow))

        for f in range(1, cv_num + 1):
            print("cross_validation:", '%01d' % (f))
            if f == cv_num:
                testset = p[((f - 1) * fold): totalassociation + 1]
            else:
                testset = p[((f - 1) * fold): f * fold]

            test_p = list(testset)

            test_f = all_f[cv_num * len(test_p): (cv_num+1) * len(test_p)]

            difference_set_f = list(set(all_f).difference(set(test_f)))
            train_p = list(set(p).difference(set(testset)))

            train_f = difference_set_f

            X = copy.deepcopy(D)
            Xn = copy.deepcopy(X)

            zero_index = []
            for ii in range(len(train_f)):
                zero_index.append([Index_zeroRow[train_f[ii]], Index_zeroCol[train_f[ii]]])
            true_list = np.zeros((len(test_p) + len(test_f), 1))
            for ii in range(len(test_p)):
                Xn[Index_PositiveRow[testset[ii]], Index_PositiveCol[testset[ii]]] = 2
                true_list[ii, 0] = 1
            for ii in range(len(test_f)):
                Xn[Index_zeroRow[test_f[ii]], Index_zeroCol[test_f[ii]]] = 3
            D1 = copy.deepcopy(Xn)
            print(D1)

            generate_and_save_neighborhood_arrays(D1)
            generate_npz_from_d1(D1)
            create_mpositive_matrix()
            create_dpositive_matrix()
            nei_index1, feats1, mps1, pos1 = load_m()
            nei_index2, feats2, mps2, pos2 = load_d()
            feats_dim_list1 = [i.shape[1] for i in feats1]
            feats_dim_list2 = [i.shape[1] for i in feats2]
            P1 = int(len(mps1))
            P2 = int(len(mps2))
            print("seed ", args.seed)
            print("Dataset: ", args.dataset)
            print("The number of meta-paths: ", P1, P2)

            model = Hemdap(args.hidden_dim, feats_dim_list1, feats_dim_list2, args.feat_drop, args.attn_drop,
                           P1, P2, args.sample_rate, args.sample_rate1, args.nei_num, args.tau, args.lam, args.gamma)
            LOSS = Contrast(args.hidden_dim, args.tau, args.lam)

            optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

            if torch.cuda.is_available():
                print('Using CUDA')
                model.cuda()
                LOSS.cuda()
                feats1 = [feat.cuda() for feat in feats1]
                feats2 = [feat.cuda() for feat in feats2]
                mps1 = [mp.cuda() for mp in mps1]
                mps2 = [mp.cuda() for mp in mps2]
                pos1 = pos1.cuda()
                pos2 = pos2.cuda()

            cnt_wait = 0
            best = 1e9
            best_t = 0
            starttime = datetime.datetime.now()
            for epoch in range(args.nb_epochs):
                model.train()
                optimiser.zero_grad()
                torch.cuda.empty_cache()
                z_mp1, z_sc1, z_mp2, z_sc2, loss2 = model(feats1, feats2, mps1, mps2, nei_index1, nei_index2, D1)

                # # 计算嵌入
                z1 = (1 - model.gamma) * z_mp1 + model.gamma * z_sc1
                z2 = (1 - model.gamma) * z_mp2 + model.gamma * z_sc2

                # 计算对比损失
                loss_contrastive = LOSS(z_mp1, z_sc1, pos1, z_mp2, z_sc2, pos2, D1, model.gamma)

                # 准备链接预测的训练数据
                # 获取训练集中正负样本的索引
                train_positive_indices = np.argwhere(D1 == 1)
                train_negative_indices = np.argwhere(D1 == 0)

                # 为了平衡数据，随机采样与正样本数目相同的负样本
                num_positive = len(train_positive_indices)
                if len(train_negative_indices) > num_positive:
                    sampled_neg_indices = np.random.choice(len(train_negative_indices), num_positive, replace=False)
                    train_negative_indices = train_negative_indices[sampled_neg_indices]
                else:
                    train_negative_indices = train_negative_indices

                # 转换为tensor并移动到设备
                positive_mirna_indices = torch.LongTensor(train_positive_indices[:, 0]).to(device)
                positive_disease_indices = torch.LongTensor(train_positive_indices[:, 1]).to(device)
                negative_mirna_indices = torch.LongTensor(train_negative_indices[:, 0]).to(device)
                negative_disease_indices = torch.LongTensor(train_negative_indices[:, 1]).to(device)

                # 获取嵌入
                positive_mirna_embeddings = z1[positive_mirna_indices]
                positive_disease_embeddings = z2[positive_disease_indices]
                negative_mirna_embeddings = z1[negative_mirna_indices]
                negative_disease_embeddings = z2[negative_disease_indices]

                # 计算链接预测概率
                positive_probs = model.get_link_prediction(positive_mirna_embeddings, positive_disease_embeddings)
                negative_probs = model.get_link_prediction(negative_mirna_embeddings, negative_disease_embeddings)

                # 创建标签
                labels = torch.cat([torch.ones_like(positive_probs), torch.zeros_like(negative_probs)], dim=0)

                # 拼接预测概率
                probs = torch.cat([positive_probs, negative_probs], dim=0).squeeze()

                # 计算二元交叉熵损失
                loss_fn = nn.BCELoss()
                loss_link_pred = loss_fn(probs, labels.squeeze())

                # 
                total_loss = 0.1 * loss_contrastive + 0.01 * loss2 +loss_link_pred
               
                # 早停机制
                if total_loss.item() < best:
                    best = total_loss.item()
                    best_t = epoch
                    cnt_wait = 0
                    torch.save(model.state_dict(), 'model_' + own_str + '.pkl')
                else:

                    cnt_wait += 1

                # 反向传播和优化
                total_loss.backward()
                optimiser.step()

            print('Loading {}th epoch'.format(best_t))
            model.load_state_dict(torch.load('model_' + own_str + '.pkl'))
            # 确保模型处于评估模式
            model.eval()

            # 准备测试数据
            test_length_p = len(test_p)
            result_list = np.zeros((test_length_p + len(test_f), 1))

            # 获取所有miRNA和疾病的嵌入
            with torch.no_grad():
                z_mp1, z_sc1, z_mp2, z_sc2, loss = model(feats1, feats2, mps1, mps2, nei_index1, nei_index2, D1)
                z1 = (1 - model.gamma) * z_mp1 + model.gamma * z_sc1  # miRNA嵌入
                z2 = (1 - model.gamma) * z_mp2 + model.gamma * z_sc2  # 疾病嵌入


            # 对测试集中的正样本进行预测
            for i in range(test_length_p):
                mirna_idx = Index_PositiveRow[testset[i]]
                disease_idx = Index_PositiveCol[testset[i]]
                mirna_embed = z1[mirna_idx].unsqueeze(0)
                disease_embed = z2[disease_idx].unsqueeze(0)
                prob = model.get_link_prediction(mirna_embed, disease_embed)
                result_list[i, 0] = prob.item()

            # 对测试集中的负样本进行预测
            for i in range(len(test_f)):
                mirna_idx = Index_zeroRow[test_f[i]]
                disease_idx = Index_zeroCol[test_f[i]]
                mirna_embed = z1[mirna_idx].unsqueeze(0)
                disease_embed = z2[disease_idx].unsqueeze(0)
                prob = model.get_link_prediction(mirna_embed, disease_embed)
                result_list[i + test_length_p, 0] = prob.item()

            # 计算AUC
            test_predict = result_list
            label = true_list
            aucvalue = roc_auc_score(label, test_predict)
            print(f"AUC value: {aucvalue}")
            del model
            del optimiser
            del LOSS
            gc.collect()

            