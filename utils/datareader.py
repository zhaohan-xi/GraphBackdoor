"""
This file processes tu-dataset and saved in a 'DataReader' class,
then the DataReader objects will transfer into 'GraphData' before training

Specifically used to process dataset from
https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
"""

import os
import torch
import numpy as np

def split_ids(args, gids, rs):
    '''
    single fold
    gids: 0-based graph id list.
    '''
    train_gids = list(rs.choice(gids, int(args.train_ratio * len(gids)), replace=False))
    test_gids = list(set(gids)-set(train_gids))
    return train_gids, test_gids


#! All files should end with .txt
class DataReader():
    """
    Wil contain keys ['adj_list', 'nlabel', 'labels', 'attr', 'features', 
                      'splits', 'n_node_max', 'num_features', 'num_classes']
    - 'adj_list': generated by 'read_graph_adj' from '_A.txt', which represents 
                  a list of adj matrices, whose shape may be different. Stored in
                  same order of graph indicator.
                  In list format, each element is a np.array, refers to a square and 
                  symmetric adj matrix.
                  
                  use: datareader.data['adj_list'][gid] - a 2D adj matrix
                  
    - 'nlabels': generated by 'read_node_features' from '_node_labels.txt', which
                represents node labels within each same graph instance. Stored in 
                same sequence of '_node_labels.txt' or '_graph_indicator.txt'. 
                Stored in list format, 2D. Each internal dim refers to a feature 
                list (node label list) for a graph instance.
                  
    - 'attr': generated by 'read_node_features' from "_node_attributes.txt" if have,
              which represents original features of each node within a graph instance.
              Order are same with previous. 
              Stored as a 2D list of 1D np.array. Internal list is a series of original
              node feature vectors for a graph instance. Internal element is a 1D np.array
              represents the (maybe floating-point) feature vector for a speficif node.
              More exatcly, is [
                                [array(f1, f2, ..), array(f1, f2, ..), array(f1, f2, ..)], 
                                [similar node feature vectors in 2nd graph instance],
                                [similar node feature vectors in 3rd graph instance],
                                ...
                                ]
              
    - 'features': combination of 'nlabel' and 'attr'. Where 'nlabel' are 
                transferred as one-hot format to show which label belongs to
                a specific node. Overall sequence is same with previous.
                Each onehot feature matrix has shape (N, D1+D2), where N is 
                number of nodes within a specific graph instance, D1, D2 are 
                number of possible labels and node feature vector length within
                this graph, respectively. D2 is optional.
                
                use : datareader.data['features'][gid] - a 2D (N, D1+D2) matrix of constructed features
                       
    - 'labels': concrete label for each graph instance, with same order of 'graph_labels.txt'.
                Stored as a list of np.int64.
                
                use : datareader.data['labels'][gid] - a single int label
                 
    - 'splits': a split of train/test sets. 
                {
                    'train': [list of train graph ids, in int], 
                    'test': [list of test graph ids, in int]
                }.
                                
                use : datareader.data['splits']['train/test'] - a list of int gids
                                
                               
                                
    - 'n_node_max': max num of nodes within a graph instance among all graphs. Single int.
    - 'num_features': size of concatenate features in 'features'. Single int.
    - 'num_classes': num of graph classes. Single int.
    """

    def __init__(self, args):
        
        # self.args = args
        assert args.use_nlabel_asfeat or args.use_org_node_attr or args.use_degree_asfeat, \
            'need at least one source to construct node features'

        self.data_path = os.path.join(args.data_path, args.dataset)
        self.rnd_state = np.random.RandomState(args.seed)
        files = os.listdir(self.data_path)
        data = {}
        
        """
        Load raw graphs, nodes, record in 2 dicts.
        Load adj list for each graph with sequence of graph indicator.
        Load node labels for each graph with sequence of graph indicator.
        Load graph labels for each graph with sequence of graph indicator.
        """
        nodes, graphs = self.read_graph_nodes_relations(
                        list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0])
        data['adj_list'] = self.read_graph_adj( # in case of Tox21_Axx_...
                           list(filter(lambda f: f.find('_A.') >= 0, files))[0], nodes, graphs)

        node_labels_file = list(filter(lambda f: f.find('node_labels') >= 0, files))
        if len(node_labels_file) == 1:
            data['nlabels'] = self.read_node_features(
                               node_labels_file[0], nodes, graphs, fn=lambda s: int(s.strip()))
        else:
            data['nlabels'] = None
            
        data['labels'] = np.array(
                          self.parse_txt_file(
                          list(filter(lambda f: f.find('graph_labels') >= 0 or f.find('graph_attributes') >= 0, files))[0],
                          line_parse_fn=lambda s: int(float(s.strip()))))

        if args.use_org_node_attr:
            data['attr'] = self.read_node_features(list(filter(lambda f: f.find('node_attributes') >= 0, files))[0],
                                                   nodes, graphs,
                                                   fn=lambda s: np.array(list(map(float, s.strip().split(',')))))

        '''also include this part into GetFinalFeatures()
        '''
        # In each graph sample, treat node labels (if have) as feature for one graph.
        nlabels, n_edges, degrees = [], [], []
        for sample_id, adj in enumerate(data['adj_list']):
            N = len(adj)  # number of nodes
            
            # some verifications
            if data['nlabels'] is not None:
                assert N == len(data['nlabels'][sample_id]), (N, len(data['nlabels'][sample_id]))
            # if not np.allclose(adj, adj.T):
            #     print(sample_id, 'not symmetric')  # not symm is okay, maybe direct graph
            n = np.sum(adj)  # total sum of edges
            # assert n % 2 == 0, n
            
            n_edges.append(int(n / 2))  # undirected edges, so need to divide by 2
            degrees.extend(list(np.sum(adj, 1)))
            if data['nlabels'] is not None:
                nlabels.append(np.array(data['nlabels'][sample_id]))

        # Create nlabels over graphs as one-hot vectors for each node
        if data['nlabels'] is not None:
            nlabels_all = np.concatenate(nlabels)
            nlabels_min = nlabels_all.min()
            num_nlabels = int(nlabels_all.max() - nlabels_min + 1)  # number of possible values

            

        #--------- Generate onehot-feature ---------#
        features = GetFinalFeatures(args, data)
        
        # final graph feature dim
        num_features = features[0].shape[1]

        shapes = [len(adj) for adj in data['adj_list']]
        labels = data['labels']  # graph class labels, np.ndarray
        labels -= np.min(labels)  # to start from 0

        classes = np.unique(labels)
        num_classes = len(classes)
        
        """
        Test whether labels are successive, e.g., 0,1,2,3,4,..i, i+1,..
        If not, make them successive. New labels still store in "labels".
        """
        if not np.all(np.diff(classes) == 1):
            print('making labels sequential, otherwise pytorch might crash')
            labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
            for lbl in range(num_classes):
                labels_new[labels == classes[lbl]] = lbl
            labels = labels_new
            classes = np.unique(labels)
            assert len(np.unique(labels)) == num_classes, np.unique(labels)


        def stats(x):
            return (np.mean(x), np.std(x), np.min(x), np.max(x))

        print('N nodes avg/std/min/max: \t%.2f/%.2f/%d/%d' % stats(shapes))
        print('N edges avg/std/min/max: \t%.2f/%.2f/%d/%d' % stats(n_edges))
        print('Node degree avg/std/min/max: \t%.2f/%.2f/%d/%d' % stats(degrees))
        print('Node features dim: \t\t%d' % num_features)
        print('N classes: \t\t\t%d' % num_classes)
        print('Classes: \t\t\t%s' % str(classes))

        for lbl in classes:
            print('Class %d: \t\t\t%d samples' % (lbl, np.sum(labels == lbl)))

        if args.data_verbose:
            if data['nlabels'] is not None:
                for u in np.unique(nlabels_all):
                    print('nlabels {}, count {}/{}'.format(u, np.count_nonzero(nlabels_all == u), len(nlabels_all)))
    
        # some datasets like "Fingerprint" may lack graph in _indicator.txt
#         N_graphs = len(labels)  # number of samples (graphs) in data
#         assert N_graphs == len(data['adj_list']) == len(features), 'invalid data'
        N_graphs = len(data['adj_list'])
    
        # Create train/test sets
        train_gids, test_gids = split_ids(args, self.rnd_state.permutation(N_graphs), self.rnd_state)
        splits = {'train': train_gids,
                  'test': test_gids}
        
        data['features'] = features
        data['labels'] = labels
        data['splits'] = splits
        data['n_node_max'] = np.max(shapes)  # max number of nodes
        data['num_features'] = num_features
        data['num_classes'] = num_classes

        self.data = data
        
        # print(len(data['features']), len(data['adj_list']), len(data['labels']))
        assert len(data['features'])==len(data['adj_list'])==len(data['labels']), \
               "Graph Number Mismatch, Possible Reason: due to insuccessive graph indicator, \
                some gids are not existed in original indicator files, only thing is filtering graph labels. \
                Remember that insuccessive graph indicator is okay, graph labels-graphs are corresponding by \
                stored index in data['xxx']."
        print()
        
    def parse_txt_file(self, fpath, line_parse_fn=None):
        """
        Read a file, split each line by pre-defined pattern (e.g., ','),  
        save results in list. Transferring data into Int is done outside.
        """
        with open(os.path.join(self.data_path, fpath), 'r') as f:
            lines = f.readlines()
        data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
        return data

    
    def read_graph_nodes_relations(self, fpath):
        """
        From graph_indicator.txt file, find { node_id: graph_id } and { graph_id:[nodes] }.
        """
        graph_ids = self.parse_txt_file(fpath, 
                                        line_parse_fn=lambda s: int(s.rstrip()))
        nodes, graphs = {}, {}
        for node_id, graph_id in enumerate(graph_ids):
            if graph_id not in graphs:
                graphs[graph_id] = []
            graphs[graph_id].append(node_id)
            nodes[node_id] = graph_id
        graph_ids = np.unique(list(graphs.keys()))
        for graph_id in graph_ids:
            graphs[graph_id] = np.array(graphs[graph_id])
        return nodes, graphs

    
    # for direct graph, row is source nodes
    def read_graph_adj(self, fpath, nodes, graphs):
        edges = self.parse_txt_file(fpath, 
                                    line_parse_fn=lambda s: s.split(','))
        
        adj_dict = {}
        for edge in edges:
            # Note: TU-datasets are all 1 based node id
            node1 = int(edge[0].strip()) - 1  # -1 because of zero-indexing in our code
            node2 = int(edge[1].strip()) - 1
            graph_id = nodes[node1]
            
            # both nodes in edge side should in a same graph
            assert graph_id == nodes[node2], ('invalid data', graph_id, nodes[node2])
            if graph_id not in adj_dict:
                n = len(graphs[graph_id])
                adj_dict[graph_id] = np.zeros((n, n))
                
            ind1 = np.where(graphs[graph_id] == node1)[0]
            ind2 = np.where(graphs[graph_id] == node2)[0]
            assert len(ind1) == len(ind2) == 1, (ind1, ind2)
            adj_dict[graph_id][ind1, ind2] = 1

        # no-connection graph may not included on code above,
        # should specially add it, e.g., graph-291 in Fingerprint
        # data set only have single node 1477 (1-based index),
        # which is not in edge file since it has no connection.
        # But still, we should add it to ensure the consistent.
        # some graphs in Tox21 also only have isolated nodes.
        adj_list = []
        for gid in sorted(list(graphs.keys())):
            if gid in adj_dict:
                adj_list.append(adj_dict[gid])
            else:
                adj_list.append(np.zeros((len(graphs[gid]), len(graphs[gid]))))
        return adj_list
    
    
    def read_node_features(self, fpath, nodes, graphs, fn):
        '''
        Return 'feature' graph by graph.
        here 'feature' may refer to (1) node attributes; (2) node labels; (3) node degrees
        '''
        node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)
        node_features = {}
        for node_id, x in enumerate(node_features_all):
            graph_id = nodes[node_id]
            if graph_id not in node_features:
                node_features[graph_id] = [None] * len(graphs[graph_id])
            ind = np.where(graphs[graph_id] == node_id)[0]   # exactly find on index
            assert len(ind) == 1, ind
            assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
            node_features[graph_id][ind[0]] = x
        node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
        return node_features_lst
    
    
def GetFinalFeatures(args, data):
    '''
    Construct features for each graph instnace, may comes from 3 parts.
    Each element in 'features' refers to constructed feature mat
    to a graph. This feature mas has shape (Ni, Di), where Ni is number
    of nodes in graph_i, and Di is combined feature dimension, may comes
    from node labels, node features and degree.
    '''

    # In each graph sample, treat node labels (if have) as feature for one graph.
    nlabels, n_edges, degrees = [], [], []
    for sample_id, adj in enumerate(data['adj_list']):
        N = len(adj)  # number of nodes
        n = np.sum(adj)  # total sum of edges
        
        n_edges.append(int(n / 2))  # undirected edges, so need to divide by 2
        degrees.extend(list(np.sum(adj, 1)))
        if data['nlabels'] is not None:
            nlabels.append(np.array(data['nlabels'][sample_id]))

    # Create features over graphs as one-hot vectors for each node
    if data['nlabels'] is not None:
        nlabels_all = np.concatenate(nlabels)
        nlabels_min = nlabels_all.min()
        num_nlabels = int(nlabels_all.max() - nlabels_min + 1)  # number of possible values

    final_features = []
    max_degree = int(np.max(degrees))  # maximum node degree among all graphs
    for sample_id, adj in enumerate(data['adj_list']):
        N = adj.shape[0]

        # OneHot Feature: (N, D), where D is all possible feature nums 
        # among ondes within a graph. Each position in is 0/1 to show
        # whether it has/hasnot a corresopnding feature here. E.g., if 
        # original features (also original node labels) range from 3~8, 
        # now D = 6 (8-3+1), feature "3" will map to position "0", even 
        # though there are multiple "3" in original feature vector.

        # This is down inside of one single graph.
    

        # part 1: one-hot nlabels as feature
        if args.use_nlabel_asfeat:
            if data['nlabels'] is not None:
                x = data['nlabels'][sample_id]
                nlabels_onehot = np.zeros((len(x), num_nlabels))
                for node, value in enumerate(x):
                    if value is not None:
                        nlabels_onehot[node, value - nlabels_min] = 1
            else:
                nlabels_onehot = np.empty((N, 0))
        else:
            nlabels_onehot = np.empty((N, 0))

        # part 2 (optional, not always have): original node features
        if args.use_org_node_attr:
            if args.dataset in ['COLORS-3', 'TRIANGLES']:
                # first column corresponds to node attention and shouldn't be used as node features
                feature_attr = np.array(data['attr'][sample_id])[:, 1:]
            else:
                feature_attr = np.array(data['attr'][sample_id])
        else:
            feature_attr = np.empty((N, 0))

        # part 3 (optinal): node degree 
        if args.use_degree_asfeat:
            degree_onehot = np.zeros((N, max_degree + 1))
            degree_onehot[np.arange(N), np.sum(adj, 1).astype(np.int32)] = 1
        else:
            degree_onehot = np.empty((N, 0))

        node_features = np.concatenate((nlabels_onehot, feature_attr, degree_onehot), axis=1)
        if node_features.shape[1] == 0:
            # dummy features for datasets without node labels/attributes
            # node degree features can be used instead
            node_features = np.ones((N, 1))
        final_features.append(node_features)
        
    return final_features


class GraphData(torch.utils.data.Dataset):
    def __init__(self, datareader: DataReader, gids: list):
        self.idx = gids
        self.rnd_state = datareader.rnd_state
        self.set_fold(datareader.data)

    def set_fold(self, data):
        self.total = len(data['labels'])
        self.n_node_max = data['n_node_max']    
        self.num_classes = data['num_classes']
        self.num_features = data['num_features']
        self.labels = [data['labels'][i] for i in self.idx]
        self.adj_list = [data['adj_list'][i] for i in self.idx]
        self.features = [data['features'][i] for i in self.idx]
        # print('%s: %d/%d' % (self.split_name.upper(), len(self.labels), len(data['labels'])))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # convert to torch
        return [torch.as_tensor(self.features[index], dtype=torch.float),  # node features
                torch.as_tensor(self.adj_list[index], dtype=torch.float),  # adj matrices
                int(self.labels[index])]

