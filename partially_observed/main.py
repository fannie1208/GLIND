import argparse
import sys
import os, random
import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from logger import Logger, save_result
from dataset import *
from eval import evaluate_node_task, eval_acc, eval_rocauc, eval_f1, eval_rmse
from parse import parser_add_main_args
from ours import *

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset_tr = DppinDataset(args.data_dir, osp.join('data/', f'DPPIN-{args.window}/'), mode='train', window=args.window, train_num=args.train_num, valid_num=1)
dataset_val = DppinDataset(args.data_dir, osp.join('data/', f'DPPIN-{args.window}/'), mode='valid', window=args.window, train_num=args.train_num, valid_num=1)
dataset_te = [DppinDataset(args.data_dir, osp.join('data/', f'DPPIN-{args.window}/'), mode=f'test{i}', window=args.window, train_num=args.train_num, valid_num=1) for i in range(11-args.train_num)]

train_loader = DataLoader(dataset_tr, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)
test_loader = [DataLoader(i, batch_size=args.batch_size, shuffle=False) for i in dataset_te]

# if len(dataset.y.shape) == 1:
#     dataset.y = dataset.y.unsqueeze(1)
class_num = 1
# class_num = max(dataset.y.max().item() + 1, dataset.y.shape[1])
feat_num = args.window

### Load method ###

criterion = nn.MSELoss(reduction='none' if args.method in ['irm', 'groupdro'] else 'mean')

model = GLIND(feat_num, class_num, args, device).to(device)

eval_func = eval_rmse

logger = Logger(args.runs, args)

# model.train()
# print('MODEL:', model)

tr_acc, val_acc = [], []

### Training loop ###
for run in range(args.runs):
    t0 = time.time()
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float('-inf')
    a,b = [], []
    for epoch in range(args.epochs):
        t1 = time.time()
        model.train()
        for batch in train_loader:
            batch.to(device)
            optimizer.zero_grad()
            loss = model.loss_compute(batch, criterion, args)
            loss.backward()
            optimizer.step()
        t2=time.time()

        result = evaluate_node_task(model, train_loader, val_loader, test_loader, eval_func, args, device)
        t3=time.time()
        logger.add_result(run, result)

        tr_acc.append(result[0])
        val_acc.append(result[2])

        if epoch % args.display_step == 0:
            m = f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {result[0]:.4f}, Valid: {result[1]:.4f} '
            for i in range(len(result)-2):
                m += f'Test OOD{i+1}: {result[i+2]:.4f} '
            print(m)
    if args.task == 'node-cls':
        logger.print_statistics(run)
    else:
        logger.print_statistics_neg(run)
    # t4 = time.time()


results = logger.print_statistics_reg()

### Save results ###
if args.save_result:
    save_result(args, results)