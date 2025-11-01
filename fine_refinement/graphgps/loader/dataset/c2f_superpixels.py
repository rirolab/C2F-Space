import os
import os.path as osp
import shutil
import pickle

import torch
from tqdm import tqdm
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)


class C2FSuperpixels(InMemoryDataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string, optional): Option to select the graph construction format.
            If :obj: `"edge_wt_only_coord"`, the graphs are 8-nn graphs with the edge weights computed based on
            only spatial coordinates of superpixel nodes.
            If :obj: `"edge_wt_coord_feat"`, the graphs are 8-nn graphs with the edge weights computed based on
            combination of spatial coordinates and feature values of superpixel nodes.
            If :obj: `"edge_wt_region_boundary"`, the graphs region boundary graphs where two regions (i.e. 
            superpixel nodes) have an edge between them if they share a boundary in the original image.
            (default: :obj:`"edge_wt_region_boundary"`)
        slic_compactness (int, optional): Option to select compactness of slic that was used for superpixels
            (:obj:`10`, :obj:`30`). (default: :obj:`30`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """
    
    def __init__(self, root, name='edge_wt_region_boundary', slic_compactness=30, split='train',
                 transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.slic_compactness = slic_compactness
        assert split in ['train', 'val', 'test']
        assert name in ['edge_wt_only_coord', 'edge_wt_coord_feat', 'edge_wt_region_boundary']
        # assert slic_compactness in [1, 10, 30]

        # if split == 'test': return
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']

    @property
    def raw_dir(self):
        return osp.join(self.root,
                        'slic_compactness_' + str(self.slic_compactness)+
                        self.name,
                        'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root,
                        'slic_compactness_' + str(self.slic_compactness)+
                        self.name,
                        'processed')
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        raise NotImplementedError
        
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)
            
            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                
                """
                Each `graph` is a tuple (x, edge_attr, edge_index, y)
                    Shape of x : [num_nodes, 18]
                    Shape of edge_attr : [num_edges, 1] or [num_edges, 2]
                    Shape of edge_index : [2, num_edges]
                    Shape of y : [num_nodes],
                    meta_info: dict
                """
                
                x = graph[0].to(torch.float)
                edge_attr = graph[1].to(torch.float)
                edge_index = graph[2]
                y = torch.LongTensor(graph[3])
                assert x.size(0) == y.size(0)
                assert edge_index.size(1) == edge_attr.size(0)
                meta_info = graph[4]
                logits = graph[5] if len(graph) == 6 else None

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y, meta_info=meta_info, logits=logits)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
