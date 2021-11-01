import sys, os
from utils.datareader import DataReader
sys.path.append(os.path.abspath('..'))

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.mask import recover_mask
from trojan.prop import forwarding

class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input>thrd, torch.tensor(1.0, device=device, requires_grad=True),
                                      torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None


    
class GraphTrojanNet(nn.Module):
    def __init__(self, sq_dim, layernum=1, dropout=0.05):
        super(GraphTrojanNet, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers.append(nn.Linear(sq_dim, sq_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(sq_dim, sq_dim))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, input, mask, thrd, 
                device=torch.device('cpu'), 
                activation='relu', 
                for_whom='topo',
                binaryfeat=False):

        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """
        GW = GradWhere.apply

        bkdmat = self.layers(input)
        if activation=='relu':
            bkdmat = F.relu(bkdmat)
        elif activation=='sigmoid':
            bkdmat = torch.sigmoid(bkdmat)    # nn.Functional.sigmoid is deprecated

        if for_whom == 'topo':  # not consider direct yet
            bkdmat = torch.div(torch.add(bkdmat, bkdmat.transpose(0, 1)), 2.0)
        if for_whom == 'topo' or (for_whom == 'feat' and binaryfeat):
            bkdmat = GW(bkdmat, thrd, device)
        bkdmat = torch.mul(bkdmat, mask)

        return bkdmat
    

def train_gtn(args, model, toponet: GraphTrojanNet, featnet: GraphTrojanNet,
               pset, nset, topomasks, featmasks, 
               init_dr: DataReader, bkd_dr: DataReader, Ainputs, Xinputs):
    """
    All matrix/array like inputs should already in torch.tensor format.
    All tensor parameters or models should initially stay in CPU when
    feeding into this function.
    
    About inputs of this function:
    - pset/nset: gids in trainset
    - init_dr: init datareader, keep unmodified inside of each resampling
    - bkd_dr: store temp adaptive adj/features, get by  init_dr + GTN(inputs)
    """
    if torch.cuda.is_available():
        cuda = torch.device('cuda')
        cpu = torch.device('cpu')
    
    init_As = init_dr.data['adj_list']
    init_Xs = init_dr.data['features']
    bkd_As = bkd_dr.data['adj_list']
    bkd_Xs = bkd_dr.data['features']
    
    nodenums = [len(adj) for adj in init_As]
    glabels = torch.LongTensor(init_dr.data['labels']).to(cuda)
    glabels[pset] = args.target_class
    allset = np.concatenate((pset, nset))
    
    optimizer_topo = optim.Adam(toponet.parameters(),
                       lr=args.gtn_lr,
                       weight_decay=5e-4)
    optimizer_feat = optim.Adam(featnet.parameters(),
                       lr=args.gtn_lr,
                       weight_decay=5e-4)
    
    
    #----------- training topo generator -----------#
    toponet.to(cuda)
    model.to(cuda)
    topo_thrd = torch.tensor(args.topo_thrd).to(cuda)
    criterion = nn.CrossEntropyLoss()
    
    toponet.train()    
    for _ in tqdm(range(args.gtn_epochs), desc="training topology generator"): 
        optimizer_topo.zero_grad()
        # generate new adj_list by dr.data['adj_list']
        for gid in pset:
            SendtoCUDA(gid, [init_As, Ainputs, topomasks])    # only send the used graph items to cuda
            rst_bkdA = toponet(
                Ainputs[gid], topomasks[gid], topo_thrd, cuda, args.topo_activation, 'topo')
            # rst_bkdA = recover_mask(nodenums[gid], topomasks[gid], 'topo')
            # bkd_dr.data['adj_list'][gid] = torch.add(rst_bkdA, init_As[gid])
            bkd_dr.data['adj_list'][gid] = torch.add(rst_bkdA[:nodenums[gid], :nodenums[gid]], init_As[gid])   # only current position in cuda
            SendtoCPU(gid, [init_As, Ainputs, topomasks])
            
        loss = forwarding(args, bkd_dr, model, allset, criterion)
        loss.backward()
        optimizer_topo.step()
        torch.cuda.empty_cache()
        
    toponet.eval()
    toponet.to(cpu)
    model.to(cpu)
    for gid in pset:
        SendtoCPU(gid, [bkd_dr.data['adj_list']])
    del topo_thrd
    torch.cuda.empty_cache()
    

    #----------- training feat generator -----------#
    featnet.to(cuda)
    model.to(cuda)
    feat_thrd = torch.tensor(args.feat_thrd).to(cuda)
    criterion = nn.CrossEntropyLoss()
    
    featnet.train()    
    for epoch in tqdm(range(args.gtn_epochs), desc="training feature generator"): 
        optimizer_feat.zero_grad()
        # generate new features by dr.data['features']
        for gid in pset:
            SendtoCUDA(gid, [init_Xs, Xinputs, featmasks])  # only send the used graph items to cuda
            rst_bkdX = featnet(
                Xinputs[gid], featmasks[gid], feat_thrd, cuda, args.feat_activation, 'feat')
            # rst_bkdX = recover_mask(nodenums[gid], featmasks[gid], 'feat')
            # bkd_dr.data['features'][gid] = torch.add(rst_bkdX, init_Xs[gid])
            bkd_dr.data['features'][gid] = torch.add(rst_bkdX[:nodenums[gid]], init_Xs[gid])   # only current position in cuda
            SendtoCPU(gid, [init_Xs, Xinputs, featmasks])
            
        # generate DataLoader
        loss = forwarding(
            args, bkd_dr, model, allset,  criterion)
        loss.backward()
        optimizer_feat.step()
        torch.cuda.empty_cache()
        
    featnet.eval()
    featnet.to(cpu)
    model.to(cpu)
    for gid in pset:
        SendtoCPU(gid, [bkd_dr.data['features']])
    del feat_thrd
    torch.cuda.empty_cache()
    
    return toponet, featnet

#----------------------------------------------------------------
def SendtoCUDA(gid, items):
    """
    - items: a list of dict / full-graphs list, 
             used as item[gid] in items
    - gid: int
    """
    cuda = torch.device('cuda')
    for item in items:
        item[gid] = torch.as_tensor(item[gid], dtype=torch.float32).to(cuda)
        
        
def SendtoCPU(gid, items):
    """
    Used after SendtoCUDA, target object must be torch.tensor and already in cuda.
    
    - items: a list of dict / full-graphs list, 
             used as item[gid] in items
    - gid: int
    """
    
    cpu = torch.device('cpu')
    for item in items:
        item[gid] = item[gid].to(cpu)