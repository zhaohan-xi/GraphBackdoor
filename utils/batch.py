import numpy as np
import torch


def collate_batch(batch):
    '''
    function: Creates a batch of same size graphs by zero-padding node features and adjacency matrices 
            up to the maximum number of nodes in the CURRENT batch rather than in the entire dataset.
    param batch: [node_features*batch_size, A*batch_size, label*batch_size]
    return: [padded feature matrices, padded adjecency matrices, non-padding positions, nodenums, labels]
    '''
    B = len(batch)
    nodenums = [len(batch[b][1]) for b in range(B)]
    if len(batch[0][0].shape)==2:
        C = batch[0][0].shape[1]   # C is feature dim
    else:
        C = batch[0][0].shape[0]
    n_node_max = int(np.max(nodenums))

    graph_support = torch.zeros(B, n_node_max)
    A = torch.zeros(B, n_node_max, n_node_max)
    X = torch.zeros(B, n_node_max, C)
    for b in range(B):
        X[b, :nodenums[b]] = batch[b][0]                # store original values in top (no need to pad feat dim, node dim only)
        A[b, :nodenums[b], :nodenums[b]] = batch[b][1]   # store original values in top-left corner
        graph_support[b][:nodenums[b]] = 1  # mask with values of 0 for dummy (zero padded) nodes, otherwise 1

    nodenums = torch.from_numpy(np.array(nodenums)).long()
    labels = torch.from_numpy(np.array([batch[b][2] for b in range(B)])).long()
    return [X, A, graph_support, nodenums, labels]

    
    # Note: here mask "graph_support" is only a 1D mask for each graph instance.
    #       When use this mask for 2D work, should first extend into 2D.
    