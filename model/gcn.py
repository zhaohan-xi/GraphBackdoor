import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from utils.graph import numpy_to_graph

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

# Used for inductive case (graph classification) by default.
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)


# 2 layers by default
class GCN(nn.Module):
    def __init__(self, in_dim, out_dim,
                 hidden_dim=[64, 32],  # GNN layers + 1 layer MLP
                 dropout=0.2,
                 activation=F.relu):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(GCNLayer(in_dim, hidden_dim[0]))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(GCNLayer(hidden_dim[i], hidden_dim[i+1]))
    
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        fc.append(nn.Linear(hidden_dim[-1], out_dim))
        self.fc = nn.Sequential(*fc)


    def forward(self, data):
        batch_g = []
        for adj in data[1]:
            batch_g.append(numpy_to_graph(adj.cpu().detach().T.numpy(), to_cuda=adj.is_cuda)) 
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
    
