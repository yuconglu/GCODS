from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class gconv_RNN(nn.Module):
    def __init__(self):
        super(gconv_RNN, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('nvc,nvw->nwc', (x, A))
        return x.contiguous()


class gconv_hyper(nn.Module):
    def __init__(self):
        super(gconv_hyper, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('nvc,vw->nwc', (x, A))
        return x.contiguous()


class gcn(nn.Module):
    def __init__(self, dims, gdep, dropout, alpha, beta, gamma, type=None):
        super(gcn, self).__init__()
        self.gdep = gdep
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.type_GNN = type

        if type == 'RNN':
            self.gconv = gconv_RNN()
            self.gconv_preA = gconv_hyper()
            self.mlp = nn.Linear((gdep + 1) * dims[0], dims[1])
        elif type == 'hyper':
            self.gconv = gconv_hyper()
            self.mlp = nn.Sequential(
                OrderedDict([('fc1', nn.Linear((gdep + 1) * dims[0], dims[1])),
                             ('sigmoid1', nn.Sigmoid()),
                             ('fc2', nn.Linear(dims[1], dims[2])),
                             ('sigmoid2', nn.Sigmoid()),
                             ('fc3', nn.Linear(dims[2], dims[3])),
                             ('tanh1', nn.Tanh())]))

    def forward(self, x, adj):
        h = x
        out = [h]
        if self.type_GNN == 'RNN':
            for _ in range(self.gdep):
                h = self.alpha * x + self.beta * self.gconv(h, adj[0]) + self.gamma * self.gconv_preA(h, adj[1])
                out.append(h)
        else:
            for _ in range(self.gdep):
                h = self.alpha * x + self.gamma * self.gconv(h, adj)
                out.append(h)
        ho = torch.cat(out, dim=-1)
        ho = self.mlp(ho)
        return ho


class MCDP_Net(nn.Module):
    def __init__(self, num_nodes, gcn_depth, dropout, alpha, beta, gamma,
                 node_dim, rnn_size, in_dim, hyperGNN_dim, predefined_A):
        super(MCDP_Net, self).__init__()
        self.num_nodes = num_nodes
        self.static_adj = predefined_A
        self.emb1 = nn.Embedding(num_nodes, node_dim)
        self.emb2 = nn.Embedding(num_nodes, node_dim)
        self.idx = torch.arange(self.num_nodes)
        dims_hyper = [rnn_size, hyperGNN_dim, 2, node_dim]
        self.gcn1 = gcn(dims_hyper, gcn_depth, dropout, alpha, beta, gamma, 'hyper')
        self.gcn2 = gcn(dims_hyper, gcn_depth, dropout, alpha, beta, gamma, 'hyper')
        self.alpha_tanh = 3

    def forward(self, x, h_state):
        nodevec1 = self.emb1(self.idx.to(x.device))
        nodevec2 = self.emb2(self.idx.to(x.device))
        hyper_input = x
        filter1 = self.gcn1(hyper_input, self.static_adj)
        filter2 = self.gcn2(hyper_input, self.static_adj)
        nodevec1 = torch.tanh(self.alpha_tanh * torch.mul(nodevec1.unsqueeze(0), filter1))
        nodevec2 = torch.tanh(self.alpha_tanh * torch.mul(nodevec2.unsqueeze(0), filter2))
        a = torch.matmul(nodevec1, nodevec2.transpose(2, 1)) - torch.matmul(nodevec2, nodevec1.transpose(2, 1))
        adj = F.relu(torch.tanh(self.alpha_tanh * a))
        return adj
