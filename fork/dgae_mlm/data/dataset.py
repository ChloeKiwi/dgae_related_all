import os
import json
import torch
from data.loaders import KekulizedMolDataset, FromNetworkx
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from data.transforms import AddNoFeat, AddSynFeat, AddSynFeatToUnannotated, AddRandomFeat, AddSpectralFeat, AddCyclesFeat
from easydict import EasyDict
from typing import List, Tuple
from data.utils import get_indices, get_data_info
from torch_geometric.utils import to_dense_adj
from utils.func import plot_graphs


def get_dataset(dataset: str, config: EasyDict) -> Tuple[List[DataLoader], EasyDict, EasyDict]:
    if dataset == 'zinc' or dataset == 'qm9': #mol数据集
        # Choose the appropriate transforms based on the dataset and configuration
        transforms = []
        if config.data.add_spectral_feat:
            transforms.append(AddSpectralFeat())
        if config.data.add_cycles_feat:
            transforms.append(AddCyclesFeat())
        if config.data.add_path_feat:
            transforms.append(AddSynFeat(config.data.max_node_num))
        else:
            transforms.append(AddNoFeat(config.data.max_node_num))
        if config.data.add_random_feat:
            transforms.append(AddRandomFeat())
        # Create the dataset with the chosen transforms
        #config.log.debug = config.log.ebug
        data = KekulizedMolDataset('./data/', pre_transform=Compose(transforms), dataset=dataset)

        # Load the test indices from the corresponding file
        train_idx, test_idx = get_indices(config, dataset, len(data))

        # Create DataLoaders for training and test sets
        train_loader = DataLoader(data[train_idx], batch_size=config.training.batch_size,
                                  shuffle=True, drop_last=True)
        test_loader = DataLoader(data[test_idx], batch_size=1000, drop_last=True, shuffle=True)
        loaders = train_loader, test_loader
        # Update the configuration with additional features and dataset-specific properties
        data_info = get_data_info(config, data, dataset)

    elif dataset == 'ego' or dataset == 'community' or dataset == 'enzymes':
        # Choose the appropriate transforms based on the dataset and configuration
        transforms = []
        if hasattr(config.data, 'add_spectral_feat') and config.data.add_spectral_feat:
            transforms.append(AddSpectralFeat())
        if hasattr(config.data, 'add_cycles_feat') and config.data.add_cycles_feat:
            transforms.append(AddCyclesFeat())
        if hasattr(config.data, 'add_path_feat') and config.data.add_path_feat:
            transforms.append(AddSynFeat(config.data.max_node_num))
        else:
            transforms.append(AddNoFeat(config.data.max_node_num))
        if hasattr(config.data, 'add_random_feat') and config.data.add_random_feat:
            transforms.append(AddRandomFeat())
        # Create the dataset with the chosen transforms
        data = FromNetworkx('./data/', transform=Compose(transforms), dataset=dataset)

        # Determine the test set size and create DataLoaders for training and test sets
        test_size = int(config.data.test_split * len(data)) #len(data)是图的数量

        SEED = 42
        torch.manual_seed(SEED)
        idx = torch.randperm(len(data))
        train_loader = DataLoader(data[idx[test_size:]], batch_size=config.training.batch_size,
                                  shuffle=True, drop_last=True)
        test_loader = DataLoader(data[idx[:test_size]], batch_size=test_size)
        loaders = train_loader, test_loader

        # Update the configuration with additional features and dataset-specific properties
        data_info = get_data_info(config, data, dataset)
        
        # 获取节点类别、图类别信息
        if hasattr(data, 'label'):
            data_info.n_node_class = data.label.unique().shape[0]
        if hasattr(data, 'graph_label'):
            data_info.n_graph_class = data.graph_label.unique().shape[0]
        
        # import ipdb; ipdb.set_trace()
        # Optional: Visualize a batch of graphs (uncomment the following lines [and import] if you want to use this)
        batch = next(iter(train_loader))
        batch = to_dense_adj(batch.edge_index, batch=batch.batch)
        plot_graphs(batch[:20], max_plot=20, wandb=None, title='original graph', filepath=f"./plots/{config.exp_name}/{config.run_name}", filename=f'ori_{dataset}') #原始数据绘制
    else:
        raise NotImplemented('Dataset not available now... or check your spelling')

    return loaders, config, data_info
