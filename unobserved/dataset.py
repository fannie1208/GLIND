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

class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        
        Usage after construction: 
        
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        
        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25, label_num_per_class=20):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        elif split_type == 'class':
            train_idx, valid_idx, test_idx = class_rand_splits(self.label, label_num_per_class=label_num_per_class)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}

        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):  
        return '{}({})'.format(self.__class__.__name__, len(self))

def load_stl10(data_dir, train_num=3, train_ratio=0.5, valid_ratio=0.25):
    path=os.path.join(data_dir, 'stl10', 'features.pkl')
    dataset = NCDataset('stl10')
    data=pkl.load(open(path,'rb'))
    x_train,y_train,x_test,y_test=data['x_train'],data['y_train'],data['x_test'],data['y_test']
    x=np.concatenate((x_train,x_test),axis=0)
    y=np.concatenate((y_train,y_test))

    label=torch.LongTensor(y)
    x=torch.Tensor(x)
    num_image=x.shape[0]
    shuffled_indices = np.random.permutation(num_image)
    x = x[shuffled_indices]
    y = y[shuffled_indices]
    num_sample = int(num_image / 6)

    synthesized_edges = torch.load('../data/stl10/edge.pt')
    dataset = Data(x=torch.Tensor(x), edge_index=synthesized_edges, y=torch.LongTensor(y))
    dataset.num_nodes = x.shape[0]

    env = [0]*num_sample + [1]*num_sample + [2]*num_sample + [3]*num_sample + [4]*num_sample + [5]*num_sample
    env = torch.tensor(env, dtype=torch.long)
    dataset.env = env
    dataset.env_num = 6
    dataset.train_env_num = 3

    ind_idx = torch.arange(num_sample*train_num)
    idx_ = torch.randperm(ind_idx.size(0))
    train_idx_ind = idx_[:int(idx_.size(0) * train_ratio)]
    valid_idx_ind = idx_[int(idx_.size(0) * train_ratio): int(idx_.size(0) * (train_ratio + valid_ratio))]
    test_idx_ind = idx_[int(idx_.size(0) * (train_ratio + valid_ratio)):]
    dataset.train_idx = ind_idx[train_idx_ind]
    dataset.valid_idx = ind_idx[valid_idx_ind]
    dataset.test_in_idx = ind_idx[test_idx_ind]
    dataset.test_ood_idx = [torch.arange(num_sample*3,num_sample*4), torch.arange(num_sample*4,num_sample*5), torch.arange(num_sample*5,num_sample*6)]
    return dataset


def load_cifar10(data_dir, num_image=30000, train_num=3, train_ratio=0.5, valid_ratio=0.25):
    path=os.path.join(data_dir, 'cifar10', 'features.pkl')
    dataset = NCDataset('cifar10')
    data=pkl.load(open(path,'rb'))
    x_train,y_train,x_test,y_test=data['x_train'],data['y_train'],data['x_test'],data['y_test']
    x=np.concatenate((x_train,x_test),axis=0)
    y=np.concatenate((y_train,y_test))
    x = x[:num_image]
    y = y[:num_image]
    shuffled_indices = np.random.permutation(num_image)
    x = x[shuffled_indices]
    y = y[shuffled_indices]
    num_sample = int(num_image / 6)
    synthesized_edges = torch.load('../data/cifar10/edge.pt')
    dataset = Data(x=torch.Tensor(x), edge_index=synthesized_edges, y=torch.LongTensor(y))
    dataset.num_nodes = x.shape[0]
    env = [0]*num_sample + [1]*num_sample + [2]*num_sample + [3]*num_sample + [4]*num_sample + [5]*num_sample
    env = torch.tensor(env, dtype=torch.long)
    dataset.env = env
    dataset.env_num = 6
    dataset.train_env_num = 3

    ind_idx = torch.arange(num_sample*train_num)
    idx_ = torch.randperm(ind_idx.size(0))
    train_idx_ind = idx_[:int(idx_.size(0) * train_ratio)]
    valid_idx_ind = idx_[int(idx_.size(0) * train_ratio): int(idx_.size(0) * (train_ratio + valid_ratio))]
    test_idx_ind = idx_[int(idx_.size(0) * (train_ratio + valid_ratio)):]
    dataset.train_idx = ind_idx[train_idx_ind]
    dataset.valid_idx = ind_idx[valid_idx_ind]
    dataset.test_in_idx = ind_idx[test_idx_ind]
    dataset.test_ood_idx = [torch.arange(num_sample*3,num_sample*4), torch.arange(num_sample*4,num_sample*5), torch.arange(num_sample*5,num_sample*6)]
    return dataset