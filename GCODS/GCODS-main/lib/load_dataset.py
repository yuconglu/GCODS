import os
import numpy as np
import torch

def load_st_dataset(dataset):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(root_dir, 'data')
    
    data_path = os.path.join(data_dir, f'X_{dataset}.npy')
    data = np.load(data_path)

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    return data

def get_adjacency_matrix(args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(root_dir, 'data')

    edge_path = os.path.join(data_dir, f'edge_index_{args.dataset}.npy')
    
    edge_idx = np.load(edge_path)
    
    edge_index = torch.tensor(edge_idx, dtype=torch.int64).to(args.device)
    return edge_index