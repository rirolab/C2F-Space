import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder,
                                               register_edge_encoder)

"""
=== Description of the VOCSuperpixels dataset === 
Each graph is a tuple (x, edge_attr, edge_index, y)
Shape of x : [num_nodes, 14]
Shape of edge_attr : [num_edges, 1] or [num_edges, 2]
Shape of edge_index : [2, num_edges]
Shape of y : [num_nodes]
"""

VOC_node_input_dim = 3*2+ 2 + 3

class VOCNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = torch.nn.Linear(VOC_node_input_dim, emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        # import pdb;pdb.set_trace()
        batch.x = self.encoder(batch.x)

        return batch

register_node_encoder('VOCNode', VOCNodeEncoder)


class VOCEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        VOC_edge_input_dim = 2 if cfg.dataset.name == 'edge_wt_region_boundary' else 1
        self.encoder = torch.nn.Linear(VOC_edge_input_dim, emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr)
        return batch


register_edge_encoder('VOCEdge', VOCEdgeEncoder)
