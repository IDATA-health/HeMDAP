import numpy as np
import scipy.sparse as sp
import torch as th


def generate_and_save_neighborhood_arrays(D1):
    M, D = D1.shape

    nei_m_d = {}
    for miRNA_id in range(M):
        nei_m_d[miRNA_id] = list(np.where(D1[miRNA_id, :] == 1)[0])

    nei_d_m = {}
    for disease_id in range(D):
        nei_d_m[disease_id] = list(np.where(D1[:, disease_id] == 1)[0])

    nei_m_d_list = [nei_m_d[i] for i in sorted(nei_m_d.keys())]
    nei_d_m_list = [nei_d_m[i] for i in sorted(nei_d_m.keys())]
    nei_m_d_array = np.array(nei_m_d_list, dtype=object)
    nei_d_m_array = np.array(nei_d_m_list, dtype=object)

    np.save("/root/HeMDAP/data/hmdd/nei_m_d.npy", nei_m_d_array)
    np.save("/root/HeMDAP/data/hmdd/nei_d_m.npy", nei_d_m_array)


def create_mpositive_matrix():
    pos_num = 5
    p = 757
    mm = sp.load_npz("/root/HeMDAP/data/hmdd//mm.npz")
    mgm = sp.load_npz("/root/HeMDAP/data/hmdd//mgm.npz")
    mdm = sp.load_npz("/root/HeMDAP/data/hmdd//mdm.npz")
    mgdgm = sp.load_npz("/root/HeMDAP/data/hmdd/mgdgm.npz")
    all = (mgm + mm + mdm + mgdgm).A.astype("float64")
    all_ = (all > 0).sum(-1)
    print(all_.max(), all_.min(), all_.mean())
    pos = np.zeros((p, p))
    k = 0

    for i in range(len(all)):
        one = all[i].nonzero()[0]
        if len(one) > pos_num:
            oo = np.argsort(-all[i, one])
            sele = one[oo[:pos_num]]
            pos[i, sele] = 1
            k += 1
        else:
            pos[i, one] = 1
    pos = sp.coo_matrix(pos)
    sp.save_npz("/root/HeMDAP/data/hmdd/m_pos.npz", pos)


def create_dpositive_matrix():
    pos_num = 5
    p = 435
    dgd = sp.load_npz("/root/HeMDAP/data/hmdd//dgd.npz")
    dmd = sp.load_npz("/root/HeMDAP/data/hmdd//dmd.npz")
    dgmgd = sp.load_npz("/root/HeMDAP/data/hmdd/dgmgd.npz")
    all = (dgd + dmd + dgmgd).A.astype("float64")
    all_ = (all > 0).sum(-1)
    print(all_.max(), all_.min(), all_.mean())
    pos = np.zeros((p, p))
    k = 0
    for i in range(len(all)):
        one = all[i].nonzero()[0]
        if len(one) > pos_num:
            oo = np.argsort(-all[i, one])
            sele = one[oo[:pos_num]]
            pos[i, sele] = 1
            k += 1
        else:
            pos[i, one] = 1

    pos = sp.coo_matrix(pos)
    sp.save_npz("/root/HeMDAP/data/hmdd/d_pos.npz", pos)


def generate_npz_from_d1(D1):
    M = 757
    D = 435
    miRNA_ids = np.arange(M)
    disease_ids = np.arange(D)
    path1 = '/root/HeMDAP/data/hmdd/dmd.npz'
    path2 = '/root/HeMDAP/data/hmdd/mdm.npz'
    md_data = []
    dm_data = []

    for miRNA_id in miRNA_ids:
        for disease_id in disease_ids:
            if D1[miRNA_id, disease_id] == 1:
                md_data.append([miRNA_id, disease_id])
                dm_data.append([disease_id, miRNA_id])

    md_ = sp.coo_matrix((np.ones(len(md_data)), (np.array(md_data)[:, 0], np.array(md_data)[:, 1])),
                        shape=(M, D)).toarray()
    dm_ = sp.coo_matrix((np.ones(len(dm_data)), (np.array(dm_data)[:, 0], np.array(dm_data)[:, 1])),
                        shape=(D, M)).toarray()

    dmd = np.matmul(md_.T, md_) > 0
    mdm = np.matmul(dm_.T, dm_) > 0

    dmd_sparse = sp.coo_matrix(dmd)
    mdm_sparse = sp.coo_matrix(mdm)

    sp.save_npz(path1, dmd_sparse)
    sp.save_npz(path2, mdm_sparse)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def load_d():
    # The order of node types: 0 p 1 a 2 s
    path = "/root/HeMDAP/data/hmdd/"

    nei_d_g = np.load(path + "nei_d_g.npy", allow_pickle=True)
    nei_d_m = np.load(path + "nei_d_m.npy", allow_pickle=True)
    feat_d = sp.load_npz(path + "d_fea.npz")
    feat_m = sp.load_npz(path + "m_fea.npz")
    feat_g = sp.load_npz(path + "g_fea.npz")
    dmd = sp.load_npz(path + "dmd.npz")
    dgd = sp.load_npz(path + "dgd.npz")
    dgmgd = sp.load_npz(path + "dgmgd.npz")
    d_pos = sp.load_npz(path + "d_pos.npz")

    nei_d_g = [th.LongTensor(i) for i in nei_d_g]
    nei_d_m = [th.LongTensor(i) for i in nei_d_m]
    feat_d = th.FloatTensor(preprocess_features(feat_d))
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_g = th.FloatTensor(preprocess_features(feat_g))
    dmd = sparse_mx_to_torch_sparse_tensor(normalize_adj(dmd))
    dgd = sparse_mx_to_torch_sparse_tensor(normalize_adj(dgd))
    dgmgd = sparse_mx_to_torch_sparse_tensor(normalize_adj(dgmgd))
    pos = sparse_mx_to_torch_sparse_tensor(d_pos)

    return [nei_d_m, nei_d_g], [feat_d, feat_m, feat_g], [dmd, dgd, dgmgd], pos


def load_m():
    # The order of node types: 0 p 1 a 2 s
    path = "/root/HeMDAP/data/hmdd/"
    nei_m_d = np.load(path + "nei_m_d.npy", allow_pickle=True)
    nei_m_g = np.load(path + "nei_m_g.npy", allow_pickle=True)
    feat_d = sp.load_npz(path + "d_fea.npz")
    feat_m = sp.load_npz(path + "m_fea.npz")
    feat_g = sp.load_npz(path + "g_fea.npz")
    mm = sp.load_npz(path + "mm.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mgm = sp.load_npz(path + "mgm.npz")
    mgdgm = sp.load_npz(path + "mgdgm.npz")
    m_pos = sp.load_npz(path + "m_pos.npz")

    nei_m_d = [th.LongTensor(i) for i in nei_m_d]
    nei_m_g = [th.LongTensor(i) for i in nei_m_g]
    feat_d = th.FloatTensor(preprocess_features(feat_d))
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_g = th.FloatTensor(preprocess_features(feat_g))
    mm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mm))
    mdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))
    mgm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mgm))
    mgdgm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mgdgm))
    pos = sparse_mx_to_torch_sparse_tensor(m_pos)

    return [nei_m_d, nei_m_g], [feat_m, feat_d, feat_g], [mm, mdm, mgm, mgdgm], pos
