#%%
import argparse

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from model import VGAE
from tqdm import tqdm
import pickle as pkl
from pathlib import Path

#%%
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col))
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def main(input, output, dim, epoch, device):
    #%%
    print('Loading data')
    with open(input, 'rb') as f:
        data = pkl.load(f)
        num_actor, num_asser = data['df_filter'].author_id.nunique(), data['df_filter'].consolid_id.nunique()
        x_onehot = np.eye(num_actor + num_asser, num_actor + num_asser)
        print(f'{num_actor=}, {num_asser=}')
    print('Processing data')
    adj_matrix = nx.adjacency_matrix(data['filter_G'])
    adj_matrix.eliminate_zeros()
    adj_matrix[adj_matrix > 1] = 1
    #%%
    adj_N = adj_matrix.shape[0]
    adj_train = adj_matrix
    adj_norm = preprocess_graph(adj_train)
    features = sp.coo_matrix(x_onehot)
    pos_weight = float(adj_N * adj_N - adj_train.sum()) / adj_train.sum() * 10.0
    norm = adj_N * adj_N / float((adj_N * adj_N - adj_train.sum()) * 2)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label_dense = adj_label.todense()
    adj_norm = torch.sparse_coo_tensor(*sparse_to_tuple(adj_norm), dtype=torch.float32).to_dense().to(device)
    adj_label = torch.sparse_coo_tensor(*sparse_to_tuple(adj_label), dtype=torch.float32).to_dense().to(device)
    features = torch.sparse_coo_tensor(*sparse_to_tuple(features), dtype=torch.float32).to(device)

    #%%
    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight
    weight_tensor = weight_tensor.to(device)

    # # Model and optimizer
    #%%
    model = VGAE(num_actor, num_asser, features.shape[1], 32, dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.22, weight_decay=1e-4)

    #%%
    def get_acc(adj_rec, adj_label):
        labels_all = adj_label.view(-1).long()
        preds_all = (adj_rec > 0.5).view(-1).long()
        accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
        return accuracy
    #%%
    with tqdm(total=epoch, ncols=100, desc='Training') as pbar:
        for _ in range(epoch):
            optimizer.zero_grad()

            embed, mu, logstd = model.encode(adj_norm, features)
            A_pred = (model.decode(embed) - 0.5) / 0.5

            train_acc = get_acc(A_pred, adj_label)
            kl_divergence = 0.5 / A_pred.size(0) * (1 + 2 * logstd - mu ** 2 - torch.exp(logstd) ** 2).sum(1).mean()
            loss = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1), weight=weight_tensor) - 0.1 * kl_divergence
            pbar.set_postfix(kl_divergence=-kl_divergence.item(), loss=loss.item(), acc=train_acc.item())

            loss.backward()
            optimizer.step()
            pbar.update()
    print('Saving embedding')
    result_embedding, _, _ = model.encode(adj_norm, features)
    result_embedding = result_embedding.cpu().detach().numpy()
    np.save(output, result_embedding)
# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='specify input pkl')
    parser.add_argument('--output', type=str, required=True, help='specify input pkl')
    parser.add_argument('--dim', type=int, default=2, help='output feature dim')
    parser.add_argument('--epoch', type=int, default=800, help='number of training epoch')
    parser.add_argument('--device', type=str, default='cpu', help='number of training epoch')
    args = parser.parse_args()

    args.input = Path(args.input).resolve()
    args.output = Path(args.output).resolve()
    main(args.input, args.output, args.dim, args.epoch, args.device)
