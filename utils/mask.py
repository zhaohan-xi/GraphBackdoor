import numpy as np
import torch
import copy

def gen_mask(datareader, bkd_gids, bkd_nid_groups):
    """
    Input a datareader and a list of backdoor candidate nodes (train/test),
    generate 2 list of masks (2D) to each of them, for topology and feature,
    respectively.
    
    Here a adj mask is (N, N), and feat mask is (N, F), where N is maximum 
    num of nodes among all graphs in a dataset, F is fixed feat dim value.
    
    About how to use the mask: Topo- and Feat-mask are used in a same manner:
    (1) After the padding input (N, N/F) pass though its corresponding AdaptNet, 
        we get a (N, N/F) result for one graph instance.
    (2) Simply do element-wise torch.mul with mask and this result, since we 
        only want to keep mutual information inside of a backdoor pattern.
    (3) After masking redundant information, remember to remove additional dim
        in row/col, recover this masked result back to original dim same with
        corresponding graph instance.
    (4) Simply add recovered result with initialized adj / feat matrix.
    
    About inputs:
    - bkd_gids: 1D list
    - bkd_node_groups: 3D list
    """
    nodenums = [len(adj) for adj in datareader.data['adj_list']]
    N = max(nodenums)
    F = np.array(datareader.data['features'][0]).shape[1]
    topomask = {}
    featmask = {}
    
    for i in range(len(bkd_gids)):
        gid = bkd_gids[i]
        groups = bkd_nid_groups[i]
        if gid not in topomask: topomask[gid] = torch.zeros(N, N)
        if gid not in featmask: featmask[gid] = torch.zeros(N, F)
        
        for group in groups:
            for nid in group:
                topomask[gid][nid][group] = 1
                topomask[gid][nid][nid] = 0
                featmask[gid][nid][::] = 1
                
    return topomask, featmask
    
    
def recover_mask(Ni, mask, for_whom):
    """
    Step3 of the mask usage, recover each masked result back to original:
    topomask[gid]: (N, N) --> (Ni, Ni)
    featmask[gid]: (N, F) --> (Ni, F)
    
    Not change original mask
    
    About mask: 
    topomask: contains all topo masks in train/test set, dict.
    featmask: contains all feat masks in train/test set, dict.
    Return: mask for single graph instance
    """
    recovermask = copy.deepcopy(mask)

    if for_whom == 'topo':
        recovermask = recovermask[:Ni, :Ni]
    elif for_whom == 'feat':
        recovermask = recovermask[:Ni]
    
    return recovermask
    