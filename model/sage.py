import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl import DGLGraph, transforms
from dgl.nn.pytorch.conv import SAGEConv
from utils.graph import numpy_to_graph

# Used for inductive case (graph classification) by default.
class GraphSAGE(nn.Module):  
    def __init__(self, in_dim, out_dim,
                 hidden_dim=[64, 32],  # GNN layers + 1 layer MLP
                 dropout=0.2,
                 activation=F.relu,
                 aggregator_type='gcn'):   # mean/gcn/pool/lstm
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(SAGEConv(in_dim, hidden_dim[0], aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(len(hidden_dim) - 1):
            self.layers.append(SAGEConv(hidden_dim[i], hidden_dim[i+1], aggregator_type, feat_drop=dropout, activation=activation))
        
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        fc.append(nn.Linear(hidden_dim[-1], out_dim))
        self.fc = nn.Sequential(*fc)


    def forward(self, data):
        batch_g = []
        for adj in data[1]:
            # cannot use tensor init DGLGraph
            batch_g.append(numpy_to_graph(adj.cpu().T.numpy(), to_cuda=adj.is_cuda)) 
        batch_g = dgl.batch(batch_g)
        
        mask = data[2]
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(2) # (B,N,1)  
        
        B,N,F = data[0].shape[:3]
        x = data[0].reshape(B*N, F)
        mask = mask.reshape(B*N, 1)
        for layer in self.layers:
            x = layer(batch_g, x)
            x = x * mask
        
        F_prime = x.shape[-1]
        x = x.reshape(B, N, F_prime)
        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes (usually performs better than average)
        # x = torch.mean(x, dim=1).squeeze()
        x = self.fc(x)
        return x