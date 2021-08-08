import torch
import numpy as np

def gen_input(args, datareader, bkd_gids):
    """
    Prepare inputs for GTN, topo input and feat input together.
    
    About inputs (of this function):
    - args: control adapt-input type 
    
    Note: Extend input size as (N, N) / (N, F) where N is max node num among all graphs
    """
    As = {}
    Xs = {}
    for gid in bkd_gids:
        if gid not in As: As[gid] = torch.tensor(datareader.data['adj_list'][gid], dtype=torch.float)
        if gid not in Xs: Xs[gid] = torch.tensor(datareader.data['features'][gid], dtype=torch.float)
    Ainputs = {}
    Xinputs = {}
    
    if args.gtn_input_type == '1hop':
        for gid in bkd_gids:
            if gid not in Ainputs: Ainputs[gid] = As[gid].clone().detach()
            if gid not in Xinputs: Xinputs[gid] = torch.mm(Ainputs[gid], Xs[gid])
                
    elif args.gtn_input_type == '2hop':
        for gid in bkd_gids:
            As[gid] = torch.add(As[gid], torch.mm(As[gid], As[gid]))
            As[gid] = torch.where(As[gid]>0, torch.tensor(1.0, requires_grad=True),
                                             torch.tensor(0.0, requires_grad=True))
            As[gid].fill_diagonal_(0.0)
            
        for gid in bkd_gids:
            if gid not in Ainputs: Ainputs[gid] = As[gid].clone().detach()
            if gid not in Xinputs: Xinputs[gid] = torch.mm(Ainputs[gid], Xs[gid])
    
    
    elif args.gtn_input_type == '1hop_degree': 
        rowsums = [torch.add(torch.sum(As[gid], dim=1), 1e-6) for gid in bkd_gids]
        re_Ds = [torch.diag(torch.pow(rowsum, -1)) for rowsum in rowsums]
        
        for i in range(len(bkd_gids)):
            gid = bkd_gids[i]
            if gid not in Ainputs: Ainputs[gid] = torch.mm(re_Ds[i], As[gid])
            if gid not in Xinputs: Xinputs[gid] = torch.mm(Ainputs[gid], Xs[gid])
                
                
    elif args.gtn_input_type == '2hop_degree':
        for gid in bkd_gids:
            As[gid] = torch.add(As[gid], torch.mm(As[gid], As[gid]))
            As[gid] = torch.where(As[gid]>0, torch.tensor(1.0, requires_grad=True),
                                             torch.tensor(0.0, requires_grad=True))
            As[gid].fill_diagonal_(0.0)
            
        rowsums = [torch.add(torch.sum(As[gid], dim=1), 1e-6) for gid in bkd_gids]
        re_Ds = [torch.diag(torch.pow(rowsum, -1)) for rowsum in rowsums]
        
        for i in range(len(bkd_gids)):
            gid = bkd_gids[i]
            if gid not in Ainputs: Ainputs[gid] = torch.mm(re_Ds[i], As[gid])
            if gid not in Xinputs: Xinputs[gid] = torch.mm(Ainputs[gid], Xs[gid])
                                
    else: raise NotImplementedError('not support other types of aggregated inputs')

    # pad each input into maxi possible size (N, N) / (N, F)
    NodeMax = int(datareader.data['n_node_max'])
    FeatDim = np.array(datareader.data['features'][0]).shape[1]
    for gid in Ainputs.keys():
        a_input = Ainputs[gid]
        x_input = Xinputs[gid]
        
        add_dim = NodeMax - a_input.shape[0]
        Ainputs[gid] = np.pad(a_input, ((0, add_dim), (0, add_dim))).tolist()
        Xinputs[gid] = np.pad(x_input, ((0, add_dim), (0, 0))).tolist()
        Ainputs[gid] = torch.tensor(Ainputs[gid])
        Xinputs[gid] = torch.tensor(Xinputs[gid])

    return Ainputs, Xinputs
    