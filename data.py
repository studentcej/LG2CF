import random
import pandas as pd
import numpy as np
import time
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset,DataLoader

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

class Data(Dataset):
    def __init__(self, datapair, arg, num_users, num_items):
        data = []
        self.num_users = num_users
        self.num_items = num_items
        self.arg = arg

        self.neg_prior_dict = {}


        self.pos_dict = self.get_pos(datapair)
        self.neg_dict = self.get_neg(datapair, self.pos_dict)

        for u in self.pos_dict.keys():
            for i in self.pos_dict[u]:
                data_entry = [u] + [i]
                data.append(data_entry)
        self.data = data


    def collate_fn(self,batch):
        new_data = []
        for i in batch:
            u = int(i[0])
            i = int(i[1])
            extra_pos = random.choices(self.pos_dict[u], k=self.arg.M - 1)
            neg = random.sample(self.neg_dict[u], k=self.arg.N)
            data_entry = [u] + [i] + extra_pos + neg
            new_data.append(data_entry)
        return torch.tensor(new_data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    # Get Interact Item
    def get_pos(self, datapair):
        pos_dict = dict()
        for i in datapair:
            user = i[0]
            item = i[1]
            pos_dict.setdefault(user, list())
            pos_dict[user].append(item)
        return pos_dict

    # Get Uninteract Item
    def get_neg(self, datapair, pos_dict):
        item_num = max(i[1] for i in datapair)
        item_set = {i for i in range(item_num + 1)}
        neg_dict = dict()
        for user in pos_dict.keys():
            neg_item = list(item_set - set(pos_dict[user]))
            neg_dict[user] = neg_item
        return neg_dict

    def build_graph(self):
        print('building graph adjacency matrix')
        st = time.time()
        adj_mat = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items),
                                dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.UserItemNet.tolil()
        adj_mat[:self.num_users, self.num_users:] = R
        adj_mat[self.num_users:, :self.num_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))

        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.

        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        end = time.time()
        print(f"costing {end - st}s, obtained norm_mat...")
        return norm_adj, adj_mat
'''
### A mini-batch data is orgnized as:
[[u1, i1, extra_pos1,...,extra_posN, extra_neg1,...,extranegN, candidate_item1,...,candidate_itemM],
 [u2, i2, extra_pos1,...,extra_posN, extra_neg1,...,extranegN, candidate_item1,...,candidate_itemM],
...
 [uBS, iBS, extra_pos1,...,extra_posN, extra_neg1,...,extranegN, candidate_item1,...,candidate_itemM]]
'''

def convert_spmat_to_sptensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))