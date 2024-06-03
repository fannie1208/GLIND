from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
import torch_geometric.transforms as T
from data_utils import even_quantile_labels, to_sparse_tensor

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Twitch, PPI, Reddit
from torch_geometric.transforms import NormalizeFeatures, RadiusGraph
from torch_geometric.data import Data, Batch
from torch_geometric.utils import stochastic_blockmodel_graph, subgraph, homophily, to_dense_adj, dense_to_sparse

from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv
from sklearn.neighbors import kneighbors_graph

import pickle as pkl
import os


def load_twitch_dataset(data_dir, train_num=3, train_ratio=0.5, valid_ratio=0.25):
    transform = T.NormalizeFeatures()
    sub_graphs = ['DE', 'PT', 'RU', 'ES', 'FR', 'EN']
    x_list, edge_index_list, y_list, env_list = [], [], [], []
    node_idx_list = []
    idx_shift = 0
    for i, g in enumerate(sub_graphs):
        torch_dataset = Twitch(root=f'{data_dir}/Twitch',
                              name=g, transform=transform)
        data = torch_dataset[0]
        x, edge_index, y = data.x, data.edge_index, data.y
        x_list.append(x)
        y_list.append(y)
        edge_index_list.append(edge_index + idx_shift)
        env_list.append(torch.ones(x.size(0)) * i)
        node_idx_list.append(torch.arange(data.num_nodes) + idx_shift)

        idx_shift += data.num_nodes
    x = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    env = torch.cat(env_list, dim=0)
    dataset = Data(x=x, edge_index=edge_index, y=y)
    dataset.env = env
    dataset.env_num = len(sub_graphs)
    dataset.train_env_num = train_num

    assert (train_num <= 5)

    ind_idx = torch.cat(node_idx_list[:train_num], dim=0)
    idx = torch.randperm(ind_idx.size(0))
    train_idx_ind = idx[:int(idx.size(0) * train_ratio)]
    valid_idx_ind = idx[int(idx.size(0) * train_ratio) : int(idx.size(0) * (train_ratio + valid_ratio))]
    test_idx_ind = idx[int(idx.size(0) * (train_ratio + valid_ratio)):]
    dataset.train_idx = ind_idx[train_idx_ind]
    dataset.valid_idx = ind_idx[valid_idx_ind]
    dataset.test_in_idx = ind_idx[test_idx_ind]
    dataset.test_ood_idx = [node_idx_list[-1]] if train_num>=4 else node_idx_list[train_num:]

    return dataset

def load_arxiv_dataset(data_dir, train_num=3, train_ratio=0.5, valid_ratio=0.25, inductive=True):
    from ogb.nodeproppred import NodePropPredDataset

    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root=f'{data_dir}/ogb')

    node_years = ogb_dataset.graph['node_year']

    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    node_feat = torch.as_tensor(ogb_dataset.graph['node_feat'])
    label = torch.as_tensor(ogb_dataset.labels)

    year_bound = [2005, 2010, 2012, 2014, 2016, 2018, 2021]
    env = torch.zeros(label.shape[0])
    for n in range(node_years.shape[0]):
        year = int(node_years[n])
        for i in range(len(year_bound)-1):
            if year >= year_bound[i+1]:
                continue
            else:
                env[n] = i
                break

    dataset = Data(x=node_feat, edge_index=edge_index, y=label)
    dataset.env = env
    dataset.env_num = len(year_bound)
    dataset.train_env_num = train_num

    ind_mask = (node_years < year_bound[train_num]).squeeze(1)
    idx = torch.arange(dataset.num_nodes)
    ind_idx = idx[ind_mask]
    idx_ = torch.randperm(ind_idx.size(0))
    train_idx_ind = idx_[:int(idx_.size(0) * train_ratio)]
    valid_idx_ind = idx_[int(idx_.size(0) * train_ratio): int(idx_.size(0) * (train_ratio + valid_ratio))]
    test_idx_ind = idx_[int(idx_.size(0) * (train_ratio + valid_ratio)):]
    dataset.train_idx = ind_idx[train_idx_ind]
    dataset.valid_idx = ind_idx[valid_idx_ind]
    dataset.test_in_idx = ind_idx[test_idx_ind]

    dataset.test_ood_idx = []

    for i in range(train_num, len(year_bound)-1):
        ood_mask_i = ((node_years >= year_bound[i]) * (node_years < year_bound[i+1])).squeeze(1)
        dataset.test_ood_idx.append(idx[ood_mask_i])

    return dataset