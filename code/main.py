import numpy
import scipy as sp
import torch
from matplotlib import pyplot as plt
from load_data import *
from params import *
from utils import *
from module import contrast
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
from torch_geometric.data import Data
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
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class Config(object):
    def __init__(self):
        self.fold = 5
        self.seed = 4

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, positive_seq, negative_seq):
        positive_loss = nn.BCELoss()(positive_seq, torch.ones_like(positive_seq))

        negative_loss = nn.BCELoss()(negative_seq, torch.zeros_like(negative_seq))

        loss = positive_loss + negative_loss

        return loss


class LinkPrediction(nn.Module):
    def __init__(self, ft_in):
        super(LinkPrediction, self).__init__()
        self.fc = nn.Linear(ft_in, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq):
        seq = seq.float()
        logits = self.fc(seq)
        predictions = self.sigmoid(logits)
        return predictions


class LinkPredictionTrainer:
    def __init__(self, params):
        self.params = params
        self.model = xgb.XGBClassifier(**self.params)

    def train_step(self, train_data, mirna_features, disease_features):

        pos_indices = train_data['md_train'][0]
        neg_indices = train_data['md_train'][1]

        X_train = []
        y_train = []

        for i in pos_indices:
            mirna_feature = mirna_features[i[0]]
            disease_feature = disease_features[i[1]]

            feat = np.concatenate([mirna_feature, disease_feature])
            X_train.append(feat)
            y_train.append(1)

        for j in neg_indices:
            mirna_feature = mirna_features[j[0]]
            disease_feature = disease_features[j[1]]

            feat = np.concatenate([mirna_feature, disease_feature])
            X_train.append(feat)
            y_train.append(0)

        self.model.fit(np.array(X_train), np.array(y_train))

        return self.model

    def predict(self, test_data):

        probabilities = self.model.predict_proba(test_data)

        return probabilities


if __name__ == "__main__":

    D = np.genfromtxt(r"/root/HeMDAP/data/md-matrix.txt")
    own_str = 'model'

    opt = Config()

    [row, col] = np.shape(D)

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

    varf1_score = []
    f1_score_list1 = []
    varprecision = []
    precision_list1 = []
    varrecall = []
    recall_list1 = []
    varaupr = []
    aupr_list1 = []

    auc_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    aupr_list = []

    for time in range(1, 2):

        Auc_per = []

        f1_score_per = []
        precision_per = []
        recall_per = []
        aupr_per = []

        np.random.seed(opt.seed)
        p = np.random.permutation(totalassociation)
        all_f = np.random.permutation(np.size(Index_zeroRow))

        for f in range(1, opt.fold + 1):
            #  optimiser.zero_grad()
            print("cross_validation:", '%01d' % (f))
            if f == cv_num:
                testset = p[((f - 1) * fold): totalassociation + 1]
            else:
                testset = p[((f - 1) * fold): f * fold]

            test_p = list(testset)

            test_f = all_f[(opt.fold) * len(test_p): (opt.fold + 1) * len(test_p)]
            difference_set_f = list(set(all_f).difference(set(test_f)))
            train_p = list(set(p).difference(set(testset)))

            train_f = difference_set_f

            X = copy.deepcopy(D)
            Xn = copy.deepcopy(X)

            zero_index = []
            for ii in range(len(train_f)):
                zero_index.append([Index_zeroRow[train_f[ii]], Index_zeroCol[train_f[ii]]])
            true_list = zeros((len(test_p) + len(test_f), 1))
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
            LOSS = Contrast(32, args.tau, args.lam)
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
                z_mp1, z_sc1, z_mp2, z_sc2 = model(feats1, feats2, mps1, mps2, nei_index1, nei_index2)
                loss = LOSS(z_mp1, z_sc1, pos1, z_mp2, z_sc2, pos2, D1)

                print("loss ", loss.data.cpu())
                if loss < best:
                    best = loss
                    best_t = epoch
                    cnt_wait = 0
                    torch.save(model.state_dict(), 'model_' + own_str + '.pkl')
                else:
                    cnt_wait += 1
                if cnt_wait == args.patience:
                    print('Early stopping!')
                    break
                loss.backward()
                optimiser.step()

            print('Loading {}th epoch'.format(best_t))
            model.load_state_dict(torch.load('model_' + own_str + '.pkl'))
            model.eval()

            embed1, embed2 = model.get_embeds(feats1, feats2, mps1, mps2, nei_index1, nei_index2)

            endtime = datetime.datetime.now()
            time = (endtime - starttime).seconds
            print("Total time: ", time, "s")

            mirna_features = embed1
            disease_features = embed2

            mirna_features = mirna_features.detach().cpu().numpy()  # 使用 .detach() 来创建不需要梯度的副本
            disease_features = disease_features.detach().cpu().numpy()

            train_data = prepare_data(opt, D1)

            gc.collect()
            torch.cuda.empty_cache()

            params = {
                'learning_rate': 0.03,
                'max_depth': 25,
                'subsample': 0.4,
                'colsample_bytree': 0.02,
                'n_estimators': 300,  #
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'seed': 0,
            }

            trainer = LinkPredictionTrainer(params)

            model1 = trainer.train_step(train_data, mirna_features, disease_features)

            test_length_p = len(test_p)
            result_list = zeros((test_length_p + len(test_f), 1))

            for i in range(test_length_p):
                miRNA_feats_one = mirna_features[Index_PositiveRow[testset[i]]]
                disease_feats_one = disease_features[Index_PositiveCol[testset[i]]]
                feats_one = np.concatenate((miRNA_feats_one, disease_feats_one)).reshape(1, -1)
                result_list[i, 0] = model1.predict_proba(feats_one)[:, 1]

            for i in range(len(test_f)):
                miRNA_feats_zero = mirna_features[Index_zeroRow[test_f[i]]]
                disease_feats_zero = disease_features[Index_zeroCol[test_f[i]]]

                feats_zero = np.concatenate((miRNA_feats_zero, disease_feats_zero)).reshape(1, -1)

                result_list[i + test_length_p, 0] = model1.predict_proba(feats_zero)[:, 1]
            test_predict = result_list
            label = true_list
            aucvalue = roc_auc_score(label, test_predict)


            fpr, tpr, thresholds = roc_curve(label, test_predict)

            roc_auc = roc_auc_score(label, test_predict)

            plt.plot(fpr, tpr, label=f'Fold {f} (AUC = {aucvalue:.4f})')

            precision, recall, _ = precision_recall_curve(label, test_predict)


            test_predict_binary = np.where(test_predict > 0.018, 1, 0)
            f1 = f1_score(label, test_predict_binary)


            precision = precision_score(label, test_predict_binary)

            recall = recall_score(label, test_predict_binary)


            aupr = average_precision_score(label, test_predict)

            del model
            del model1
            del optimiser
            del trainer
            del LOSS
            gc.collect()


            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"AUC: {aucvalue}")
            print(f"F1 Score: {f1}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"AUPR: {aupr}")
            auc_list.append(aucvalue)
            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)
            aupr_list.append(aupr)


        auc_mean, auc_std = np.mean(auc_list), np.std(auc_list)
        f1_mean, f1_std = np.mean(f1_list), np.std(f1_list)
        precision_mean, precision_std = np.mean(precision_list), np.std(precision_list)
        recall_mean, recall_std = np.mean(recall_list), np.std(recall_list)
        aupr_mean, aupr_std = np.mean(aupr_list), np.std(aupr_list)









