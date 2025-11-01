import os
import pickle
from scipy.spatial.distance import cdist
import numpy as np
import itertools

import dgl
import torch
import torch.utils.data

import time

import csv
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from tqdm.auto import tqdm


def sigma(dists, kth=8):
    # Compute sigma and reshape
    try:
        # Get k-nearest neighbors for each node
        knns = np.partition(dists, kth, axis=-1)[:, kth::-1]
        sigma = knns.sum(axis=1).reshape((knns.shape[0], 1))/kth
    except ValueError:     # handling for graphs with num_nodes less than kth
        num_nodes = dists.shape[0]
        # this sigma value is irrelevant since not used for final compute_edge_list
        sigma = np.array([1]*num_nodes).reshape(num_nodes,1)
        
    return sigma + 1e-8 # adding epsilon to avoid zero value of sigma


def compute_adjacency_matrix_images(coord, feat, use_feat=True, kth=8):
    coord = coord.reshape(-1, 2)
    # Compute coordinate distance
    c_dist = cdist(coord, coord)
    
    if use_feat:
        # Compute feature distance
        f_dist = cdist(feat, feat)
        # Compute adjacency
        A = np.exp(- (c_dist/sigma(c_dist))**2 - (f_dist/sigma(f_dist))**2 )
    else:
        A = np.exp(- (c_dist/sigma(c_dist))**2)
        
    # Convert to symmetric matrix
    A = 0.5 * (A + A.T)
    A[np.diag_indices_from(A)] = 0
    return A        


def compute_edges_list(A, kth=8+1):
    # Get k-similar neighbor indices for each node

    num_nodes = A.shape[0]
    new_kth = num_nodes - kth
    
    if num_nodes > 9:
        knns = np.argpartition(A, new_kth-1, axis=-1)[:, new_kth:-1]
        knn_values = np.partition(A, new_kth-1, axis=-1)[:, new_kth:-1] # NEW
    else:
        # handling for graphs with less than kth nodes
        # in such cases, the resulting graph will be fully connected
        knns = np.tile(np.arange(num_nodes), num_nodes).reshape(num_nodes, num_nodes)
        knn_values = A # NEW
        
        # removing self loop
        if num_nodes != 1:
            knn_values = A[knns != np.arange(num_nodes)[:,None]].reshape(num_nodes,-1) # NEW
            knns = knns[knns != np.arange(num_nodes)[:,None]].reshape(num_nodes,-1)
    return knns, knn_values # NEW


class SuperPixDGL(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir,
                 dataset,
                 split,
                 graph_format='edge_wt_only_coord',
                 slic_compactness=10):
        assert graph_format in ['edge_wt_only_coord', 'edge_wt_coord_feat', 'edge_wt_region_boundary']
        self.split = split
        self.graph_lists = []
        self.logits_lists = []

        print("SLIC Compactness: ", slic_compactness)
        # if slic_compactness == 10:
        #     name = 100
        # else:
        name = 500
        with open(os.path.join(data_dir, '%s/%ssp_%scmpt_%s.pkl' % (dataset, str(name), str(slic_compactness), split)), 'rb') as f:
            self.graph_labels, self.sp_data = pickle.load(f)
        
        if graph_format == 'edge_wt_region_boundary':
            with open(os.path.join(data_dir, '%s/%ssp_%scmpt_%s_rag_boundary_graphs.pkl' % (dataset, str(name), str(slic_compactness), split)), 'rb') as f:
                self.region_boundary_graphs = pickle.load(f)
        
        with open(os.path.join(data_dir, '%s/%ssp_%scmpt_%s_meta_info.pkl' % (dataset, str(name), str(slic_compactness), split)), 'rb') as f:
            self.meta_info = pickle.load(f)

        self.graph_format = graph_format 
        self.n_samples = len(self.graph_labels)
        
        self._prepare()
    
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.n_samples, self.split.upper()))
        self.Adj_matrices, self.node_features, self.edges_lists, self.edge_features = [], [], [], []
        for index, sample in enumerate(tqdm(self.sp_data)):
            mean_px, coord = sample[:2]
            logits = sample[-1] if len(sample) == 4 else None
            
            try:
                coord = coord / self.img_size
            except AttributeError:
                COCO_has_variable_image_sizes = True
                
            if self.graph_format == 'edge_wt_coord_feat':
                A = compute_adjacency_matrix_images(coord, mean_px) # using super-pixel locations + features
                edges_list, edge_values_list = compute_edges_list(A) 
            elif self.graph_format == 'edge_wt_only_coord':
                A = compute_adjacency_matrix_images(coord, mean_px, False) # using only super-pixel locations
                edges_list, edge_values_list = compute_edges_list(A) 
            elif self.graph_format == 'edge_wt_region_boundary':
                A, edges_list, edge_values_list = None, None, None

            N_nodes = mean_px.shape[0]
            
            mean_px = mean_px.reshape(N_nodes, -1)
            coord = coord.reshape(N_nodes, 2)
            x = np.concatenate((mean_px, coord), axis=1)

            if edge_values_list is not None:
                edge_values_list = edge_values_list.reshape(-1) 
            

        
        # for index in tqdm(range(len(self.sp_data))):
            if self.graph_format == 'edge_wt_region_boundary':
                if x.shape[0] == 1:
                    # handling for 1 node where the self loop would be the only edge
                    g = dgl.DGLGraph()
                    g.add_nodes(x.shape[0]) 
                    g = dgl.add_self_loop(g)
                    # dummy edge feat since no actual edge present
                    g.edata['feat'] = torch.zeros(1, 2) # 1 edge and 2 feat dim
                    A = g.adjacency_matrix().to_dense().numpy()
                else:
                    # g = dgl.from_networkx(self.region_boundary_graphs[index].to_directed(),
                    #                   edge_attrs=['weight', 'count'])
                    g = self.region_boundary_graphs[index]
                    g.edata['feat'] = torch.stack((g.edata['weight'], g.edata['count']),-1)
                    del g.edata['weight'], g.edata['count']
                    A = g.adjacency_matrix().to_dense().numpy()
            else:
                g = dgl.DGLGraph()
                g.add_nodes(x.shape[0]) 

                for src, dsts in enumerate(edges_list):
                    # handling for 1 node where the self loop would be the only edge
                    if x.shape[0] == 1:
                        g.add_edges(src, dsts)
                    else:
                        g.add_edges(src, dsts[dsts!=src])
                g.edata['feat'] = torch.Tensor(edge_values_list).unsqueeze(1).half()  # NEW 
            
            g.ndata['feat'] = torch.Tensor(x).half()

            self.graph_lists.append(g)
            self.logits_lists.append(torch.Tensor(logits) if logits is not None else torch.zeros(1))

        
        

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx], self.meta_info[idx], self.logits_lists[idx]


class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])
    
    
class C2FSegDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name, graph_format, slic_compactness):
        """ 
            This class uses results from the above SuperPixDGL class.
            
            : Contains image superpixels and the node labels for each superpixel graph
            : It is analogous to segmentation task in images where pixel has a label
            : here, each superpixel (node) has a label
        """
        t_data = time.time()
        self.name = name

        assert graph_format in ['edge_wt_only_coord', 'edge_wt_coord_feat', 'edge_wt_region_boundary']
        
        print("Graph format being used: ", graph_format)
        print("SLIC Compactness: ", slic_compactness)
        
        self.test = SuperPixDGL("../../datasets", dataset=self.name, split='test',
                                graph_format=graph_format, slic_compactness=slic_compactness)
        self.val = SuperPixDGL("../../datasets", dataset=self.name, split='val',
                                graph_format=graph_format, slic_compactness=slic_compactness)
        self.train = SuperPixDGL("../../datasets", dataset=self.name, split='train',
                                 graph_format=graph_format, slic_compactness=slic_compactness)
        

        print("[I] Data load time: {:.4f}s".format(time.time()-t_data))        
    
    def get_all_split_idx(self, dataset, name, val_meta_label):
        """
            - Split total number of graphs into 2 sets with 80:20 
            - Stratified split proportionate to original distribution of data with respect to classes
            - Using sklearn to perform the split and then save the indexes
        """
        root_idx_dir = './'
        if not os.path.exists(root_idx_dir):
            os.makedirs(root_idx_dir)
        all_idx = {}

        # If there are no idx files, do the split and store the files
        if not (os.path.exists(root_idx_dir + name + '_test.index')):
            print("[!] Splitting the original COCO 2017 val data into _val and _test ...")
            splitter = StratifiedShuffleSplit(n_splits=2, test_size=0.8, random_state=0)

            for val_index, test_index in splitter.split(dataset, val_meta_label):
                f_val_w = csv.writer(open(root_idx_dir + name + '_val.index', 'w'))
                f_test_w = csv.writer(open(root_idx_dir + name + '_test.index', 'w'))

                f_val_w.writerow(val_index)
                f_test_w.writerow(test_index)

            print("[!] Splitting done!")

        # reading idx from the files
        for section in ['val', 'test']:
            with open(root_idx_dir + name + '_'+ section + '.index', 'r') as f:
                reader = csv.reader(f)
                all_idx[section] = [list(map(int, idx)) for idx in reader][0]
        return all_idx['val'], all_idx['test']
    
    def format_dataset(self, dataset):  
        """
            Utility function to recover data,
            INTO-> dgl/pytorch compatible format 
        """
        graphs = [data[0] for data in dataset]
        labels = [data[1] for data in dataset]

        return DGLFormDataset(graphs, labels)

