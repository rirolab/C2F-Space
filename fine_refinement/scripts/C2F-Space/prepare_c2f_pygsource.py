import torch
import pickle
import time
import os
import matplotlib.pyplot as plt
import pickle

from superpixels import C2FSegDatasetDGL 



def dump_coco_pyg_source(dataset, graph_format, slic_compactness):
    vallist = []
    for data in dataset.val:
        # print(data)
        x = data[0].ndata['feat'] #x
        edge_attr = data[0].edata['feat'] #edge_attr
        edge_index = torch.stack(data[0].edges(), 0) #edge_index
        y = data[1] #y
        meta_info = data[2]
        logits = data[3] if len(data) == 4 else None
        if logits is not None:
            vallist.append((x, edge_attr, edge_index, y, meta_info, logits))
        else:
            vallist.append((x, edge_attr, edge_index, y, meta_info))

    trainlist = []
    for data in dataset.train:
        # print(data)
        x = data[0].ndata['feat'] #x
        edge_attr = data[0].edata['feat'] #edge_attr
        edge_index = torch.stack(data[0].edges(), 0) #edge_index
        y = data[1] #y
        meta_info = data[2]
        logits = data[3] if len(data) == 4 else None
        if logits is not None:
            trainlist.append((x, edge_attr, edge_index, y, meta_info, logits))
        else:
            trainlist.append((x, edge_attr, edge_index, y, meta_info))

    testlist = []
    for data in dataset.test:
        # print(data)
        x = data[0].ndata['feat'] #x
        edge_attr = data[0].edata['feat'] #edge_attr
        edge_index = torch.stack(data[0].edges(), 0) #edge_index
        y = data[1] #y
        meta_info = data[2]
        logits = data[3] if len(data) == 4 else None
        if logits is not None:
            testlist.append((x, edge_attr, edge_index, y, meta_info, logits))
        else:
            testlist.append((x, edge_attr, edge_index, y, meta_info))

    print(len(trainlist), len(vallist), len(testlist))
    
    pyg_source_dir = '../../datasets/'+DATASET_NAME+'/slic_compactness_'+str(slic_compactness)+graph_format+'/raw'
    if not os.path.exists(pyg_source_dir):
        os.makedirs(pyg_source_dir)
    
    start = time.time()
    with open(pyg_source_dir+'/train.pickle','wb') as f:
        pickle.dump(trainlist,f)
    print('Time (sec):',time.time() - start) 
    
    start = time.time()
    with open(pyg_source_dir+'/val.pickle','wb') as f:
        pickle.dump(vallist,f)
    print('Time (sec):',time.time() - start)
    
    start = time.time()
    with open(pyg_source_dir+'/test.pickle','wb') as f:
        pickle.dump(testlist,f)
    print('Time (sec):',time.time() - start)
    
    
    
DATASET_NAME = 'C2FSuperpixels'
graph_format = ['edge_wt_region_boundary']
dataset = []
for gf in graph_format:
    start = time.time()
    data = C2FSegDatasetDGL(DATASET_NAME, gf, 10) 
    print('Time (sec):',time.time() - start)
    dataset.append(data)
    
for idx, gf in enumerate(graph_format):
    dump_coco_pyg_source(dataset[idx], gf, 10)
