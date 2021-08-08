import os
import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# print('using torch', torch.__version__)

from utils.paths import *
from config.id import InductiveParser
from utils.inductive.datareader import GraphData, DataReader
from utils.inductive.batch import collate_batch
from utils.inductive.navigator import Distribution
from utils.inductive.mask import Mask, RecoverMask
from utils.inductive.pickup_bkdcdd import SelectCddGraphs, SelectCddNodes

import id_adaptnet as adpt 
from id_initbkd import InitBkd
from id_adapt_input import AdaptInput
from id_propagate import TrainGNN, TestGNN


def TransferGTA(args, model, tgt_datareader, gta_version='gta-3'):
    assert torch.cuda.is_available(), "no GPU available"
    cpu = torch.device('cpu')
    cuda = torch.device('cuda')
    
    # not directly modify input datareader
    nodenums = [adj.shape[0] for adj in tgt_datareader.data['adj_list']]
    NodeMax = max(nodenums)
    FeatDim = len(tgt_datareader.data['final_features'][0][0])
    
    # INIT ADAPTNETS (GENERATORS)
    toponet = adpt.AdaptNet(NodeMax, args.adaptnet_layernum)
    featnet = adpt.AdaptNet(FeatDim, args.adaptnet_layernum)


    """
    For transfer leanring, resample in 
    """
    for rs_step in range(args.resample_steps):   # each step, choose different sample
        
        # TASK1: RANDOMLY SELECT NEW BACKDOOR GRAPH/NODE SAMPLES
        bkd_gids_adjust = SelectCddGraphs(args,
                                       tgt_datareader.data['splits']['adjust'], 
                                       tgt_datareader.data['adj_list'], 
                                       args.bkd_gratio_adjustset)
        bkd_nids_adjust, bkd_nid_groups_adjust = SelectCddNodes(args,
                                                                bkd_gids_adjust, 
                                                                tgt_datareader.data['adj_list'])   
        pset = bkd_gids_adjust
        nset = list(set(tgt_datareader.data['splits']['adjust'])-set(pset))
        
        # duplicate pset nset
        if args.pn_rate != None:
            if len(pset) > len(nset):
                repeat = int(np.ceil(len(pset)/(len(nset)*args.pn_rate)))
                nset = list(nset) * repeat
            else:
                repeat = int(np.ceil((len(nset)*args.pn_rate)/len(pset)))
                pset = list(pset) * repeat
                
        # FOR GNN TRAINING LOSS
        posgids = bkd_gids_adjust
        neggids = list(set(tgt_datareader.data['splits']['adjust'])-set(posgids))
        
        
        # TASK2: INIT DATAREADER IN COPY, PREPARE MASKS
        _dr_adjust = InitBkd(args, 
                            copy.deepcopy(tgt_datareader), 
                            None,
                            bkd_gids_adjust, 
                            bkd_nid_groups_adjust,
                            0.0, 0.0)
        if args.dataset == 'MALWARE':
            _dr_adjust.data['final_features'] = copy.deepcopy(tgt_datareader.data['final_features'])
        _bkd_dr_adjust = copy.deepcopy(_dr_adjust)  # store temp results
        
        topomask_adjust, featmask_adjust = Mask(_dr_adjust, 
                                              bkd_gids_adjust, 
                                              bkd_nid_groups_adjust)
        
        # STEP3: GENERATE INPUTS
        Ainputs_adjust, Xinputs_adjust = AdaptInput(args, 
                                                  tgt_datareader, 
                                                  bkd_gids_adjust)

        for is_step in range(args.insample_steps):
            # print("Resampling step {}, in sampling step {}".format(rs_step, is_step))
            
            toponet, featnet = adpt.TrainAdapt(args, model, 
                                               toponet, featnet,
                                               pset, nset, 
                                               topomask_adjust, featmask_adjust, 
                                               _dr_adjust, 
                                               _bkd_dr_adjust, 
                                               Ainputs_adjust, Xinputs_adjust)
            
            # get new _bkd_datareader_adjust based on trained AdaptNets
            for gid in bkd_gids_adjust:
                _bkdA_cur = toponet(Ainputs_adjust[gid], 
                                    topomask_adjust[gid], 
                                    args.topo_thrd, 
                                    cpu, 
                                    args.topo_activation, 
                                    'topo')
                _bkdA_cur = RecoverMask(nodenums[gid], 
                                        topomask_adjust[gid], 
                                        'topo')
                _bkd_dr_adjust.data['adj_list'][gid] = torch.add(_bkdA_cur, 
                                            _dr_adjust.data['adj_list'][gid])
            
                _bkdX_cur = featnet(Xinputs_adjust[gid], 
                                    featmask_adjust[gid], 
                                    args.feat_thrd, 
                                    cpu, 
                                    args.feat_activation, 
                                    'feat')
                _bkdX_cur = RecoverMask(nodenums[gid], 
                                        featmask_adjust[gid], 
                                        'feat')
                _bkd_dr_adjust.data['final_features'][gid] = torch.add(_bkdX_cur, 
                                           _dr_adjust.data['final_features'][gid]) 
                
            # train GNN
            TrainGNN(args, _bkd_dr_adjust, model, args.fold_id, posgids, neggids)
                
    #------------------------------------------------------------#
    #                   Add Backdoor on Testset                  #
    #------------------------------------------------------------#
    
    bkd_gids_test = SelectCddGraphs(args,
                                   tgt_datareader.data['splits']['test'], 
                                   tgt_datareader.data['adj_list'], 
                                   args.bkd_gratio_testset) 
    bkd_nids_test, bkd_nid_groups_test = SelectCddNodes(args,
                                                        bkd_gids_test, 
                                                        tgt_datareader.data['adj_list'])  
    # INITS, WHITEBOARD
    _dr_test = InitBkd(args, 
                       copy.deepcopy(tgt_datareader), 
                       None,
                       bkd_gids_test, 
                       bkd_nid_groups_test,
                       0.0, 0.0)
    if args.dataset == 'MALWARE': # only augment feature, aug both has bad attack performance
        _dr_test.data['final_features'] = copy.deepcopy(tgt_datareader.data['final_features'])
    _bkd_dr_test = copy.deepcopy(_dr_test)
    
    topomask_test, featmask_test = Mask(_dr_test, 
                                        bkd_gids_test, 
                                        bkd_nid_groups_test)
    Ainputs_test, Xinputs_test = AdaptInput(args, 
                                            _dr_test, 
                                            bkd_gids_test)
    
    # feed into AdaptNets, get new dr
    for gid in bkd_gids_test:
        _bkdA_cur = toponet(Ainputs_test[gid], 
                            topomask_test[gid], 
                            args.topo_thrd, 
                            cpu, 
                            args.topo_activation, 
                            'topo')
        _bkdA_cur = RecoverMask(nodenums[gid], 
                                topomask_test[gid], 
                                'topo')
        _bkd_dr_test.data['adj_list'][gid] = torch.add(_bkdA_cur, 
                        torch.as_tensor(copy.deepcopy(_dr_test.data['adj_list'][gid])))

        _bkdX_cur = featnet(Xinputs_test[gid], 
                            featmask_test[gid], 
                            args.feat_thrd, 
                            cpu, 
                            args.feat_activation, 
                            'feat')
        _bkdX_cur = RecoverMask(nodenums[gid], 
                                featmask_test[gid], 
                                'feat')
        _bkd_dr_test.data['final_features'][gid] = torch.add(_bkdX_cur, 
                      torch.as_tensor(copy.deepcopy(_dr_test.data['final_features'][gid])))

    
    return model, [_bkd_dr_test, bkd_gids_test, bkd_nids_test, bkd_nid_groups_test]
    