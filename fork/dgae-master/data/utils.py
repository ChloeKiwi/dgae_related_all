import os
import json
import torch
from easydict import EasyDict

def get_indices(config, dataset, n_instances):
    with open(os.path.join(config.data.dir, f'valid_idx_{config.data.data.lower()}.json')) as f:
        test_idx = json.load(f)
        if dataset == 'qm9':
            test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]

    # Create a boolean mask for the training indices
    train_idx = torch.ones(n_instances).bool()
    train_idx[test_idx] = False
    train_idx = train_idx[train_idx]

    if config.log.debug:
        train_idx = train_idx[:20*config.training.batch_size]
        test_idx = test_idx[:1000]

    return train_idx, test_idx

def get_data_info(config, data, dataset):
    data_info = EasyDict()
    data_info.additional_node_feat = 0
    data_info.additional_edge_feat = 0
    data_info.max_node_num = config.data.max_node_num

    if dataset == 'qm9' or dataset == 'zinc':
        data_info.annotated_nodes = True
        data_info.annotated_edges = True
        data_info.mol = True
    else:
        data_info.annotated_nodes = False
        data_info.annotated_edges = False
        data_info.mol = False

    if config.data.add_spectral_feat:
        data_info.additional_node_feat = 5

    if config.data.add_cycles_feat:
        data_info.additional_node_feat += 3


    if config.data.add_path_feat:
        data_info.additional_node_feat += 3
        data_info.additional_edge_feat = 2
    if config.data.add_random_feat:
        data_info.additional_node_feat += 4

    nf = data[99].x
    ef = data[99].edge_attr_ext
    data_info.n_node_feat = nf.shape[-1] - data_info.additional_node_feat
    data_info.n_edge_feat = ef.shape[-1] - data_info.additional_edge_feat
    if data_info.n_edge_feat == 2:
        data_info.n_edge_feat = 1
    print(data_info)
    return data_info
