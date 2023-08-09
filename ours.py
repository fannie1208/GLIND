import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import erdos_renyi_graph, remove_self_loops, add_self_loops, degree, add_remaining_self_loops
from data_utils import sys_normalized_adjacency, sparse_mx_to_torch_sparse_tensor
from torch_sparse import SparseTensor, matmul

def gcn_conv(x, edge_index):
    N = x.shape[0]
    row, col = edge_index
    d = degree(col, N).float()
    d_norm_in = (1. / d[col]).sqrt()
    d_norm_out = (1. / d[row]).sqrt()
    value = torch.ones_like(row) * d_norm_in * d_norm_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    return matmul(adj, x) # [N, D]

class GraphConvolutionBase(nn.Module):

    def __init__(self, in_features, out_features, variant=False, residual=False):
        super(GraphConvolutionBase, self).__init__()
        self.variant = variant
        self.residual = residual
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if self.residual:
            self.weight_r = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_r.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, x0):
        hi = gcn_conv(x, adj)
        if self.variant:
            hi = torch.cat([hi, x0], 1)
        output = torch.mm(hi, self.weight)
        if self.residual:
            output = output + torch.mm(x, self.weight_r)
        return output

class GLINDConv(nn.Module):

    def __init__(self, in_features, out_features, K, residual=True, backbone_type='gcn', variant=False, device=None):
        super(GLINDConv, self).__init__()
        self.backbone_type = backbone_type
        self.out_features = out_features
        self.residual = residual
        if backbone_type == 'gcn':
            self.weights = Parameter(torch.FloatTensor(K, in_features*2, out_features))
        elif backbone_type == 'gat':
            self.leakyrelu = nn.LeakyReLU()
            self.weights = nn.Parameter(torch.zeros(K, in_features, out_features))
            self.a = nn.Parameter(torch.zeros(K, 2 * out_features, 1))
        self.K = K
        self.device = device
        self.variant = variant
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weights.data.uniform_(-stdv, stdv)
        if self.backbone_type == 'gat':
            nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def specialspmm(self, adj, spm, size, h):
        adj = SparseTensor(row=adj[0], col=adj[1], value=spm, sparse_sizes=size)
        return matmul(adj, h)

    def forward(self, x, adj, z, weights=None):
        if weights == None:
            weights = self.weights
        if self.backbone_type == 'gcn':
            if not self.variant:
                hi = gcn_conv(x, adj)
            else:
                adj = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1]).to(self.device), size=(x.shape[0],x.shape[0])).to(self.device)
                hi = torch.sparse.mm(adj, x)
            hi = torch.cat([hi, x], 1)
            hi = hi.unsqueeze(0).repeat(self.K, 1, 1)  # [K, N, D*2]
            outputs = torch.matmul(hi, weights) # [K, N, D]
            outputs = outputs.transpose(1, 0)  # [N, K, D]
        elif self.backbone_type == 'gat':
            xi = x.unsqueeze(0).repeat(self.K, 1, 1)  # [K, N, D]
            h = torch.matmul(xi, weights) # [K, N, D]
            N = x.size()[0]

            adj, _ = remove_self_loops(adj)
            adj, _ = add_self_loops(adj, num_nodes=N)
            edge_h = torch.cat((h[:, adj[0, :], :], h[:, adj[1, :], :]), dim=2)  # [K, E, 2*D]
            edge_e = torch.exp(self.leakyrelu(torch.matmul(edge_h, self.a)).squeeze(2))  # [K, E]

            outputs = []
            for k in range(self.K):
                edge_e_k = edge_e[k, :] # [E]
                e_expsum_k = self.specialspmm(adj, edge_e_k, torch.Size([N, N]), torch.ones(N, 1).cuda())
                assert not torch.isnan(e_expsum_k).any()

                hi_k = self.specialspmm(adj, edge_e_k, torch.Size([N, N]), h[k])
                hi_k = torch.div(hi_k, e_expsum_k)  # [N, D]
                outputs.append(hi_k)
            outputs = torch.stack(outputs, dim=1) # [N, K, D]

        # output = outputs.mean(axis=1) # Only for ablation study
        zs = z.unsqueeze(2).repeat(1, 1, self.out_features)  # [N, K, D]
        output = torch.sum(torch.mul(zs, outputs), dim=1)  # [N, D]

        if self.residual:
            output = output + x

        return output

class GLIND(nn.Module):
    def __init__(self, d, c, args, device):
        super(GLIND, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(args.num_layers):
            self.convs.append(GLINDConv(args.hidden_channels, args.hidden_channels, args.K, backbone_type=args.backbone_type, residual=True, device=device, variant=args.variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(d, args.hidden_channels))
        self.fcs.append(nn.Linear(args.hidden_channels, c))
        self.context_enc = nn.ModuleList()
        for _ in range(args.num_layers):
            if args.context_type == 'node':
                self.context_enc.append(nn.Linear(args.hidden_channels, args.K))
            elif args.context_type == 'graph':
                self.context_enc.append(GraphConvolutionBase(args.hidden_channels, args.K, variant=False, residual=True))
            else:
                raise NotImplementedError
        self.act_fn = nn.ReLU()
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.tau = args.tau
        self.context_type = args.context_type
        self.prior_type = args.prior
        self.device = device
        # self.weights = Parameter(torch.FloatTensor(args.K, args.hidden_channels*2, args.hidden_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()
        for enc in self.context_enc:
            enc.reset_parameters()

    def forward(self, x, adj, idx=None, training=False):
        self.training = training
        x = F.dropout(x, self.dropout, training=self.training)
        h = self.act_fn(self.fcs[0](x))
        h0 = h.clone()

        if self.prior_type == 'mixture' and self.training:
            n = x.shape[0]
            p = adj.size(1) / n / (n - 1)
            edge_index_r = erdos_renyi_graph(num_nodes = n, edge_prob = p)
            values_r = torch.ones(edge_index_r.size(1))
            g_r = (values_r.numpy(), (edge_index_r[0].numpy(), edge_index_r[1].numpy()))
            adj_r = sys_normalized_adjacency(g_r, size=(n, n))
            adj_r = sparse_mx_to_torch_sparse_tensor(adj_r)
            adj_r = adj_r.to(adj.device)
            print(adj_r.coalesce().indices().shape)

            logits = []
            h_r = h.clone()
            for i, con in enumerate(self.convs):
                h_r = F.dropout(h_r, self.dropout, training=self.training)
                if self.context_type == 'node':
                    logit_r = self.context_enc[i](h_r)
                else:
                    logit_r = self.context_enc[i](h_r, adj_r, h0)
                z_r = F.gumbel_softmax(logit_r, tau=self.tau, dim=-1)
                h_r = self.act_fn(con(h_r, adj_r.coalesce().indices(), z_r))
                logits.append(logit_r.detach())

        reg = 0
        for i,con in enumerate(self.convs):
            h = F.dropout(h, self.dropout, training=self.training)
            if self.training:
                if self.context_type == 'node':
                    logit = self.context_enc[i](h)
                else:
                    logit = self.context_enc[i](h, adj, h0)
                z = F.gumbel_softmax(logit, tau=self.tau, dim=-1)
                if self.prior_type == 'uniform':
                    reg += self.reg_loss(z[idx], logit[idx])
                    #reg += self.reg_loss(z, logit)
                elif self.prior_type == 'mixture':
                    reg += self.reg_loss(z[idx], logit[idx], logits[i][idx])
            else:
                if self.context_type == 'node':
                    z = F.softmax(self.context_enc[i](h), dim=-1)
                else:
                    z = F.softmax(self.context_enc[i](h, adj, h0), dim=-1)
            # h = self.act_fn(con(h, adj, z, self.weights)) # Only for ablation study
            h = self.act_fn(con(h, adj, z))

        h = F.dropout(h, self.dropout, training=self.training)
        out = self.fcs[-1](h)
        if self.training:
            return out, reg / self.num_layers
        else:
            return out

    def reg_loss(self, z, logit, logit_0 = None):
        if self.prior_type == 'uniform':
            log_pi = logit - torch.logsumexp(logit, dim=-1, keepdim=True).repeat(1, logit.size(1))
            return torch.mean(torch.sum(
                torch.mul(z, log_pi), dim=1))
        elif self.prior_type == 'mixture':
            log_pi = logit - torch.logsumexp(logit, dim=-1, keepdim=True).repeat(1, logit.size(1))
            log_pi_0 = F.softmax(logit_0, dim=1).mean(dim=1, keepdim=True).log()
            return torch.mean(torch.sum(
                torch.mul(z, log_pi - log_pi_0), dim=1))

    def sup_loss_calc(self, y, pred, criterion, args):
        if args.dataset in ('twitch', 'elliptic', 'ppi', 'proteins'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            loss = criterion(pred, true_label.squeeze(1).to(torch.float))
        else:
            out = F.log_softmax(pred, dim=1)
            target = y.squeeze(1)
            loss = criterion(out, target)
        return loss

    def print_weight(self):
        for i, con in enumerate(self.convs):
            weights = con.weights[:, :64, :].detach().cpu().numpy()
            np.save(f'weights{i}.npy', weights)

    def loss_compute(self, d, criterion, args, idx=None):
        logits, reg_loss = self.forward(d.x, d.edge_index, idx, training=True)
        sup_loss = self.sup_loss_calc(d.y[d.train_idx], logits[d.train_idx], criterion, args)
        loss = sup_loss + args.lamda * reg_loss
        return loss