import os
import sys
from utils.datareader import DataReader
sys.path.append('/home/zxx5113/BackdoorGNN/')

import numpy as np
import copy


# return 1D list
def select_cdd_graphs(args, data: list, adj_list: list, subset: str):
    '''
    Given a data (train/test), (randomly or determinately) 
    pick up some graph to put backdoor information, return ids.
    '''
    rs = np.random.RandomState(args.seed)
    graph_sizes = [np.array(adj).shape[0] for adj in adj_list]
    bkd_graph_ratio = args.bkd_gratio_train if subset == 'train' else args.bkd_gratio_test
    bkd_num = int(np.ceil(bkd_graph_ratio * len(data)))
    
    assert len(data)>bkd_num , "Graph Instances are not enough"
    picked_ids = []
    
    # Randomly pick up graphs as backdoor candidates from data
    remained_set = copy.deepcopy(data)
    loopcount = 0
    while bkd_num-len(picked_ids) >0 and len(remained_set)>0 and loopcount<=50:
        loopcount += 1
        
        cdd_ids = rs.choice(remained_set, bkd_num-len(picked_ids), replace=False)
        for gid in cdd_ids:
            if bkd_num-len(picked_ids) <=0: 
                break
            gsize = graph_sizes[gid]
            if gsize >= 3*args.bkd_size*args.bkd_num_pergraph:
                picked_ids.append(gid)

        if len(remained_set)<len(data):
            for gid in cdd_ids:
                if bkd_num-len(picked_ids) <=0: 
                    break
                gsize = graph_sizes[gid]
                if gsize >= 1.5*args.bkd_size*args.bkd_num_pergraph and gid not in picked_ids:
                    picked_ids.append(gid)
                    
        if len(remained_set)<len(data):
            for gid in cdd_ids:
                if bkd_num-len(picked_ids) <=0: 
                    break
                gsize = graph_sizes[gid]
                if gsize >= 1.0*args.bkd_size*args.bkd_num_pergraph and gid not in picked_ids:
                    picked_ids.append(gid)
                    
        picked_ids = list(set(picked_ids))
        remained_set = list(set(remained_set) - set(picked_ids))
        if len(remained_set)==0 and bkd_num>len(picked_ids):
            print("no more graph to pick, return insufficient candidate graphs, try smaller bkd-pattern or graph size")

    return picked_ids
             

def select_cdd_nodes(args, graph_cdd_ids, adj_list):
    '''
    Given a graph instance, based on pre-determined standard,
    find nodes who should be put backdoor information, return
    their ids.

    return: same sequece with bkd-gids
            (1) a 2D list - bkd nodes under each graph
            (2) and a 3D list - bkd node groups under each graph
                (in case of each graph has multiple triggers)
    '''
    rs = np.random.RandomState(args.seed)
    
    # step1: find backdoor nodes
    picked_nodes = []  # 2D, save all cdd graphs
    
    for gid in graph_cdd_ids:
        node_ids = [i for i in range(len(adj_list[gid]))]
        assert len(node_ids)==len(adj_list[gid]), 'node number in graph {} mismatch'.format(gid)

        bkd_node_num =  int(args.bkd_num_pergraph*args.bkd_size)
        assert bkd_node_num <= len(adj_list[gid]), "error in SelectCddGraphs, candidate graph too small"
        cur_picked_nodes = rs.choice(node_ids, bkd_node_num, replace=False)
        picked_nodes.append(cur_picked_nodes)
        
    # step2: match nodes
    assert len(picked_nodes)==len(graph_cdd_ids), "backdoor graphs & node groups mismatch, check SelectCddGraphs/SelectCddNodes"

    node_groups = [] # 3D, grouped trigger nodes
    for i in range(len(graph_cdd_ids)):    # for each graph, devide candidate nodes into groups
        gid = graph_cdd_ids[i]
        nids = picked_nodes[i]

        assert len(nids)%args.bkd_size==0.0, "Backdoor nodes cannot equally be divided, check SelectCddNodes-STEP1"

        # groups within each graph
        groups = np.array_split(nids, len(nids)//args.bkd_size)
        # np.array_split return list[array([..]), array([...]), ]
        # thus transfer internal np.array into list
        # store groups as a 2D list.
        groups = np.array(groups).tolist()
        node_groups.append(groups)

    assert len(picked_nodes)==len(node_groups), "groups of bkd-nodes mismatch, check SelectCddNodes-STEP2"
    return picked_nodes, node_groups
                           
    