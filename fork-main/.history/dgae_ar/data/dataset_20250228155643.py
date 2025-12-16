import os
# import json
import torch
from data.loaders import KekulizedMolDataset, FromNetworkx, SpectureDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from data.transforms import AddNoFeat, AddSynFeat, AddSynFeatToUnannotated, AddRandomFeat, AddSpectralFeat, AddCyclesFeat
from easydict import EasyDict
from typing import List, Tuple
from data.utils import get_indices, get_data_info
from torch_geometric.utils import to_dense_adj
from utils.func import plot_graphs
from pathlib import Path
import pickle
import networkx as nx
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
# from torch_geometric.data import Data


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
        data = KekulizedMolDataset('./data/', pre_transform=Compose(transforms), dataset=dataset)
        print_dataset_statistics(data, dataset)
        
        # Load the test indices from the corresponding file
        train_idx, test_idx = get_indices(config, dataset, len(data))

        # Create DataLoaders for training and test sets
        train_loader = DataLoader(data[train_idx], batch_size=config.training.batch_size,
                                  shuffle=True, drop_last=True)
        test_loader = DataLoader(data[test_idx], batch_size=1000, drop_last=True, shuffle=True)
        loaders = train_loader, test_loader
        # Update the configuration with additional features and dataset-specific properties
        data_info = get_data_info(config, data, dataset)

    elif dataset == 'ego' or dataset == 'community' or dataset == 'enzymes' or dataset == 'grid' or dataset == 'protein' or dataset == 'citeseer' or dataset == 'community_ego':
        # Choose the appropriate transforms based on the dataset and configuration
        transforms = []
        if hasattr(config.data, 'add_spectral_feat') and config.data.add_spectral_feat:
            transforms.append(AddSpectralFeat())
        if hasattr(config.data, 'add_cycles_feat') and config.data.add_cycles_feat:
            transforms.append(AddCyclesFeat())
        if hasattr(config.data, 'add_path_feat') and config.data.add_path_feat:
            transforms.append(AddSynFeat(config.data.max_node_num))
        # if hasattr(config.data, 'add_random_feat') and config.data.add_random_feat:
        #     transforms.append(AddRandomFeat())
        else:
            transforms.append(AddNoFeat(config.data.max_node_num))
            
        # Create the dataset with the chosen transforms
        data = FromNetworkx('./data/', transform=Compose(transforms), dataset=dataset)
        
        # import ipdb; ipdb.set_trace()
        print_dataset_statistics(data, dataset)
        
        # Determine the test set size and create DataLoaders for training and test sets
        test_size = int(config.data.test_split * len(data)) #len(data)是图的数量
        SEED = 42
        if dataset == 'citeseer':
            # 由于有节点类别信息，根据类别均匀分割数据
            random_state = np.random.RandomState(SEED)
            idx_train, _, idx_test = get_train_val_test_split(
                    random_state, data, (1-config.data.test_split), 0, config.data.test_split
                )
            train_loader = DataLoader(data[idx_train], batch_size=config.training.batch_size,
                                      shuffle=True, drop_last=True)
            test_loader = DataLoader(data[idx_test], batch_size=test_size)
            loaders = train_loader, test_loader

        else: #随机分
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
        
        batch = next(iter(train_loader)) 
        batch = to_dense_adj(batch.edge_index, batch=batch.batch)
        if dataset!='protein':
            plot_graphs(batch[:20], max_plot=20, wandb=None, title='Original Graph', filepath=f"./plots/{config.exp_name}/{config.run_name}", filename=f'ori_{dataset}') #原始数据绘制
        
    # elif dataset=='planar' or dataset=='sbm':   
    #      # Choose the appropriate transforms based on the dataset and configuration
    #     transforms = []
    #     if hasattr(config.data, 'add_spectral_feat') and config.data.add_spectral_feat:
    #         transforms.append(AddSpectralFeat())
    #     if hasattr(config.data, 'add_cycles_feat') and config.data.add_cycles_feat:
    #         transforms.append(AddCyclesFeat())
    #     if hasattr(config.data, 'add_path_feat') and config.data.add_path_feat:
    #         transforms.append(AddSynFeat(config.data.max_node_num))
    #     if hasattr(config.data, 'add_random_feat') and config.data.add_random_feat:
    #         transforms.append(AddRandomFeat())
    #     else:
    #         transforms.append(AddNoFeat(config.data.max_node_num))
            
    #     # with open(Path("./data/processed") / f"{dataset}.pkl", "rb") as f:
    #     with open(Path("./data/processed") / f"{dataset}.pt", "rb") as f:
    #         data = pickle.load(f)
        
    #     # keep only largest connected component for train graphs
    #     train_graphs = [
    #         G.subgraph(max(nx.connected_components(G), key=len)) for G in data["train"] 
    #     ]
    #     test_graphs = [
    #         G.subgraph(max(nx.connected_components(G), key=len)) for G in data["test"] 
    #     ]
    #     # 创建数据集
    #     train_dataset = SpectureDataset(train_graphs, transform=Compose(transforms), dataset=dataset)        
    #     test_dataset = SpectureDataset(test_graphs, transform=Compose(transforms),dataset=dataset)
    #     train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, drop_last=True)
    #     test_loader = DataLoader(test_dataset, batch_size=len(test_graphs))
    #     loaders = train_loader, test_loader
    #     # Update the configuration with additional features and dataset-specific properties
        
    #     data_info = get_data_info(config, train_dataset, dataset)
    #     # Optional: Visualize a batch of graphs (uncomment the following lines [and import] if you want to use this)
    #     batch = next(iter(train_loader)) #TODO: 这里的可视化非常标准，baseline mmd可以从这里计算？
    #     batch = to_dense_adj(batch.edge_index, batch=batch.batch)
    #     plot_graphs(batch[:20], max_plot=20, wandb=None, title='Original Graph', filepath=f"./plots/{config.exp_name}/{config.run_name}", filename=f'ori_{dataset}') #原始数据绘制
    
    elif dataset=='planar' or dataset=='sbm':
         # Choose the appropriate transforms based on the dataset and configuration
        transforms = []
        if hasattr(config.data, 'add_spectral_feat') and config.data.add_spectral_feat:
            transforms.append(AddSpectralFeat())
        if hasattr(config.data, 'add_cycles_feat') and config.data.add_cycles_feat:
            transforms.append(AddCyclesFeat())
        if hasattr(config.data, 'add_path_feat') and config.data.add_path_feat:
            transforms.append(AddSynFeat(config.data.max_node_num))
        if hasattr(config.data, 'add_random_feat') and config.data.add_random_feat:
            transforms.append(AddRandomFeat())
        else:
            transforms.append(AddNoFeat(config.data.max_node_num))
        
        # with open(Path("./data/processed") / f"{dataset}_64_200.pt", "rb") as f:
        # with open(Path("./data/raw") / f"{dataset}.pkl", "rb") as f:
            # data = pickle.load(f)
        # data_pt = torch.load(Path("./data/processed") / f"{dataset}_64_200.pt")
        # data = pickle.load(open(Path("./data/raw") / f"{dataset}.pkl", "rb"))
        # keep only largest connected component for train graphs
        # train_graphs = [G.subgraph(max(nx.connected_components(G), key=len)) for G in data["train"]]
        # test_graphs = [G.subgraph(max(nx.connected_components(G), key=len)) for G in data["test"]]
        
        import ipdb; ipdb.set_trace()
        # data = FromNetworkx('./data/', transform=Compose(transforms), dataset=dataset)
        adjs = torch.load(Path("./data/processed")/f"{dataset}_64_200.pt")
        graphs = [nx.from_numpy_array(adj.numpy().astype(bool)) for adj in adjs]

        
    
    else:
        raise NotImplemented('Dataset not available now... or check your spelling')

    return loaders, config, data_info


def print_dataset_statistics(dataset, name: str) -> None:
    """打印数据集的统计信息"""
    if isinstance(dataset, list):  # 对于planar和sbm数据集
        graphs = dataset
    else:
        graphs = dataset.graphs if hasattr(dataset, 'graphs') else dataset
    
    num_graphs = len(graphs)
    num_nodes = []
    num_edges = []
    degrees = []
    
    for g in graphs:
        if isinstance(g, nx.Graph):  # 如果是networkx图
            num_nodes.append(g.number_of_nodes())
            num_edges.append(g.number_of_edges())
            degrees.extend([d for n, d in g.degree()])
        else:  # 如果是PyG数据
            n = g.num_nodes if hasattr(g, 'num_nodes') else g.x.shape[0]
            e = g.edge_index.shape[1] // 2  # 除以2因为是无向图
            num_nodes.append(n)
            num_edges.append(e)
            # 计算度数
            degrees.extend(torch.bincount(g.edge_index[0]).tolist())
    
    # 将统计信息写入文件
    stats_dir = "./data/stats"
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = os.path.join(stats_dir, f"{name}_stats.txt")
    
    with open(stats_file, "w") as f:
        f.write(f"=== {name} 数据集统计信息 ===\n")
        f.write(f"图的数量: {num_graphs}\n")
        f.write(f"节点数: 平均={np.mean(num_nodes):.1f}, 最小={min(num_nodes)}, 最大={max(num_nodes)}\n")
        f.write(f"边数: 平均={np.mean(num_edges):.1f}, 最小={min(num_edges)}, 最大={max(num_edges)}\n")
        f.write(f"平均度数: {np.mean(degrees):.2f}\n")
        f.write(f"度数分布: 最小={min(degrees)}, 最大={max(degrees)}\n")
        f.write("==============================\n")
        
    # 同时打印到控制台
    print(f"\n=== {name} 数据集统计信息 ===")
    print(f"图的数量: {num_graphs}")
    print(f"节点数: 平均={np.mean(num_nodes):.1f}, 最小={min(num_nodes)}, 最大={max(num_nodes)}")
    print(f"边数: 平均={np.mean(num_edges):.1f}, 最小={min(num_edges)}, 最大={max(num_edges)}")
    print(f"平均度数: {np.mean(degrees):.2f}")
    print(f"度数分布: 最小={min(degrees)}, 最大={max(degrees)}")
    print("==============================\n")


def print_dataset_statistics(dataset, name: str) -> None:
    """打印数据集的统计信息"""
    if isinstance(dataset, list):  # 对于planar和sbm数据集
        graphs = dataset
    else:
        graphs = dataset.graphs if hasattr(dataset, 'graphs') else dataset
    
    num_graphs = len(graphs)
    num_nodes = []
    num_edges = []
    degrees = []
    densities = []  # 图密度
    clustering_coeffs = []  # 聚类系数
    diameters = []  # 图直径
    avg_path_lengths = []  # 平均路径长度
    
    for g in graphs:
        if isinstance(g, nx.Graph):  # 如果是networkx图
            G = g
        else:  # 如果是PyG数据
            edge_index = g.edge_index.cpu().numpy()
            G = nx.Graph()
            G.add_edges_from(edge_index.T)
            
        n = G.number_of_nodes()
        e = G.number_of_edges()
        num_nodes.append(n)
        num_edges.append(e)
        
        # 计算度数
        degrees.extend([d for n, d in G.degree()])
        
        # 计算图密度
        density = 2 * e / (n * (n-1)) if n > 1 else 0
        densities.append(density)
        
        # 计算平均聚类系数
        try:
            clustering_coeffs.append(nx.average_clustering(G))
        except:
            clustering_coeffs.append(0)
            
        # 计算连通分量
        largest_cc = max(nx.connected_components(G), key=len)
        largest_cc_size = len(largest_cc)
        
        # 对最大连通分量计算直径和平均路径长度
        if largest_cc_size > 1:
            largest_cc_subgraph = G.subgraph(largest_cc)
            try:
                diameters.append(nx.diameter(largest_cc_subgraph))
                avg_path_lengths.append(nx.average_shortest_path_length(largest_cc_subgraph))
            except:
                diameters.append(0)
                avg_path_lengths.append(0)
        else:
            diameters.append(0)
            avg_path_lengths.append(0)
    
    # 将统计信息写入文件
    stats_dir = "./data/data_info"
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = os.path.join(stats_dir, f"{name}_stats.txt")
    
    with open(stats_file, "w") as f:
        f.write(f"=== {name} 数据集统计信息 ===\n")
        f.write(f"图的数量: {num_graphs}\n")
        f.write(f"节点数: 平均={np.mean(num_nodes):.1f}, 最小={min(num_nodes)}, 最大={max(num_nodes)}, 标准差={np.std(num_nodes):.1f}\n")
        f.write(f"边数: 平均={np.mean(num_edges):.1f}, 最小={min(num_edges)}, 最大={max(num_edges)}, 标准差={np.std(num_edges):.1f}\n")
        f.write(f"平均度数: {np.mean(degrees):.2f}, 标准差={np.std(degrees):.2f}\n")
        f.write(f"度数分布: 最小={min(degrees)}, 最大={max(degrees)}\n")
        f.write(f"图密度: 平均={np.mean(densities):.3f}, 最小={min(densities):.3f}, 最大={max(densities):.3f}\n")
        f.write(f"聚类系数: 平均={np.mean(clustering_coeffs):.3f}, 最小={min(clustering_coeffs):.3f}, 最大={max(clustering_coeffs):.3f}\n")
        f.write(f"图直径: 平均={np.mean(diameters):.1f}, 最小={min(diameters)}, 最大={max(diameters)}\n")
        f.write(f"平均路径长度: 平均={np.mean(avg_path_lengths):.2f}, 最小={min(avg_path_lengths):.2f}, 最大={max(avg_path_lengths):.2f}\n")
        f.write("==============================\n")
        
    # 同时打印到控制台
    print(f"\n=== {name} 数据集统计信息 ===")
    print(f"图的数量: {num_graphs}")
    print(f"节点数: 平均={np.mean(num_nodes):.1f}, 最小={min(num_nodes)}, 最大={max(num_nodes)}, 标准差={np.std(num_nodes):.1f}")
    print(f"边数: 平均={np.mean(num_edges):.1f}, 最小={min(num_edges)}, 最大={max(num_edges)}, 标准差={np.std(num_edges):.1f}")
    print(f"平均度数: {np.mean(degrees):.2f}, 标准差={np.std(degrees):.2f}")
    print(f"度数分布: 最小={min(degrees)}, 最大={max(degrees)}")
    print(f"图密度: 平均={np.mean(densities):.3f}, 最小={min(densities):.3f}, 最大={max(densities):.3f}")
    print(f"聚类系数: 平均={np.mean(clustering_coeffs):.3f}, 最小={min(clustering_coeffs):.3f}, 最大={max(clustering_coeffs):.3f}")
    print(f"图直径: 平均={np.mean(diameters):.1f}, 最小={min(diameters)}, 最大={max(diameters)}")
    print(f"平均路径长度: 平均={np.mean(avg_path_lengths):.2f}, 最小={min(avg_path_lengths):.2f}, 最大={max(avg_path_lengths):.2f}")
    print("==============================\n")

def sample_per_class(
    random_state, labels, num_examples_per_class, forbidden_indices=None
):
    """
    Used in get_train_val_test_split, when we try to get a fixed number of examples per class
    """

    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [
            random_state.choice(
                sample_indices_per_class[class_index],
                num_examples_per_class,
                replace=False,
            )
            for class_index in range(len(sample_indices_per_class))
        ]
    )

def get_train_val_test_split(
    random_state,
    data,
    train_examples_per_class=None,
    val_examples_per_class=None,
    test_examples_per_class=None,
    train_size=None,
    val_size=None,
    test_size=None,
):
    # 遍利数据获取label矩阵
    labels = binarize_labels(data.graph_label.numpy())    
    num_samples, num_classes = labels.shape
    
    # 打印每个类别的样本数量
    samples_per_class = labels.sum(axis=0)
    print("Samples per class:", samples_per_class)

    
    remaining_indices = list(range(num_samples))
    if train_examples_per_class is not None: #按照类别均匀划分
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else: #随机选择
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False
        )

    if val_examples_per_class is not None: #按照类别均匀划分
        val_indices = sample_per_class(
            random_state,
            labels,
            val_examples_per_class,
            forbidden_indices=train_indices,
        )
    else: #随机选择
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None: #按照类别均匀划分
        test_indices = sample_per_class(
            random_state,
            labels,
            test_examples_per_class,
            forbidden_indices=forbidden_indices,
        )
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert (
            len(np.concatenate((train_indices, val_indices, test_indices)))
            == num_samples
        )

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    print(f"len(train): {len(train_indices)}, len(test): {len(test_indices)}")
    return train_indices, val_indices, test_indices

def get_train_val_test_split(
    random_state,
    data,
    train_rate=0.8,
    val_size=None,
    test_size=None
):
    # 获取标签矩阵
    labels = binarize_labels(data.graph_label.numpy())
    num_samples, num_classes = labels.shape
    
    # 获取每个类别的样本数量
    samples_per_class = labels.sum(axis=0)
    # print("Samples per class:", samples_per_class)
    
    # 按比例计算每个类别需要的训练样本数
    train_samples_per_class = {
        i: int(count * train_rate) for i, count in enumerate(samples_per_class)
    }
    
    # 获取每个类别的样本索引
    indices_per_class = {index: [] for index in range(num_classes)}
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                indices_per_class[class_index].append(sample_index)
    
    # 按比例从每个类别中抽样
    train_indices = []
    test_indices = []
    for class_index in range(num_classes):
        class_indices = indices_per_class[class_index]
        n_train = train_samples_per_class[class_index]
        
        # 随机打乱该类别的索引
        class_indices = random_state.permutation(class_indices)
        
        # 分割训练集和测试集
        train_indices.extend(class_indices[:n_train])
        test_indices.extend(class_indices[n_train:])

    # 转换为numpy数组
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    # print(f"训练集大小: {len(train_indices)}, 测试集大小: {len(test_indices)}")
    # for i in range(num_classes):
    #     n_train = sum(labels[train_indices, i])
    #     n_test = sum(labels[test_indices, i])
    #     print(f"类别 {i}: 训练集 {n_train} 样本 ({n_train/samples_per_class[i]:.2%}), "
    #           f"测试集 {n_test} 样本 ({n_test/samples_per_class[i]:.2%})")
    
    return train_indices, None, test_indices
                
def binarize_labels(labels, sparse_output=False, return_classes=False):
    """Convert labels vector to a binary label matrix.

    In the default single-label case, labels look like
    labels = [y1, y2, y3, ...].
    Also supports the multi-label format.
    In this case, labels should look something like
    labels = [[y11, y12], [y21, y22, y23], [y31], ...].

    Parameters
    ----------
    labels : array-like, shape [num_samples]
        Array of node labels in categorical single- or multi-label format.
    sparse_output : bool, default False
        Whether return the label_matrix in CSR format.
    return_classes : bool, default False
        Whether return the classes corresponding to the columns of the label matrix.

    Returns
    -------
    label_matrix : np.ndarray or sp.csr_matrix, shape [num_samples, num_classes]
        Binary matrix of class labels.
        num_classes = number of unique values in "labels" array.
        label_matrix[i, k] = 1 <=> node i belongs to class k.
    classes : np.array, shape [num_classes], optional
        Classes that correspond to each column of the label_matrix.

    """
    if hasattr(labels[0], "__iter__"):  # labels[0] is iterable <=> multilabel format
        binarizer = MultiLabelBinarizer(sparse_output=sparse_output)
    else:
        binarizer = LabelBinarizer(sparse_output=sparse_output)
    label_matrix = binarizer.fit_transform(labels).astype(np.float32)
    return (label_matrix, binarizer.classes_) if return_classes else label_matrix


def mark_structural_groups(data, radius=1, ged_threshold=1):
    """
    为数据集中的每个图标记结构相似的节点组
    Args:
        data: PyG数据集对象
        radius: 局部子图的半径
        ged_threshold: GED阈值
    """
    def get_local_subgraph(G, node, radius=1):
        """获取节点的k跳邻域子图"""
        subgraph = nx.ego_graph(G, node, radius=radius, undirected=True) #0中心，一共6个节点
        # 重新标记节点,使中心节点为0
        mapping = {node: 0}
        for d in range(1, radius + 1):
            neighbors = set(n for n in subgraph.nodes() 
                          if nx.shortest_path_length(subgraph, node, n) == d)
            mapping.update({n: len(mapping) for n in neighbors})
        return nx.relabel_nodes(subgraph, mapping)

    def compute_ged(G1, G2):
        """计算两个图之间的GED"""
        try:
            ged = nx.graph_edit_distance(G1, G2, 
                                       node_match=lambda n1, n2: True,
                                       edge_match=lambda e1, e2: True)
            return ged if ged is not None else float('inf')
        except:
            return float('inf')
    
    def find_structural_groups(G, radius=1, ged_threshold=1):
        """找出所有局部结构相似的节点组"""
        nodes = list(G.nodes()) #len:14
        n_nodes = len(nodes) #14
        groups = []
        used = set()
        
        for i in range(n_nodes): #第一个图
            if i in used:
                continue
                
            current_group = [i]
            subgraph_i = get_local_subgraph(G, nodes[i], radius) #i中心节点的子图
            
            for j in range(i + 1, n_nodes): #第2-n个图
                if j in used:
                    continue
                    
                subgraph_j = get_local_subgraph(G, nodes[j], radius)
                ged = compute_ged(subgraph_i, subgraph_j)
                
                if ged <= ged_threshold:
                    current_group.append(j)
                    used.add(j)
            
            if len(current_group) > 1:
                groups.append([nodes[idx] for idx in current_group])
                used.add(i)
                
        return groups

    # 为数据集中的每个图标记结构组
    processed = []
    for i in range(len(data)):
        import ipdb; ipdb.set_trace()        
        # graph_new = Data()
        graph_old = data[i]
        
        for key in graph_old.keys():
            graph_new[key] = graph_old[key]
        
        G = nx.Graph()
        G.add_edges_from(graph_old.edge_index.t().tolist())
        
        # 找出结构相似的节点组
        structural_groups = find_structural_groups(G, radius, ged_threshold)
        
        # graph_new.structural_groups = structural_groups
        
        n_nodes = graph_old.edge_index.max().item() + 1
        structure_labels = torch.zeros(n_nodes, dtype = torch.long)
        for group_idx, nodes in enumerate(structural_groups):
            for node in nodes:
                structure_labels[node] = group_idx + 1
        
        # graph_new.structure_labels = structure_labels
        
        # 替换原始数据
        # data[i] = graph_new
        # processed.append(graph_new)
         # 检查是否成功添加
        print(f"Keys after adding: {processed[i].keys()}")
        print(f"Has structural_groups: {hasattr(processed[i], 'structural_groups')}")
     
    return processed
