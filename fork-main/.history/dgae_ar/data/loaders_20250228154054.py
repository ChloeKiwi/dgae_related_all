import os
import torch
import numpy as np
import pickle
import time
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import from_networkx
from utils.func import atom_number_to_one_hot, from_dense_numpy_to_sparse
from torch_geometric.data import Dataset
import networkx as nx
from collections import defaultdict


class KekulizedMolDataset(InMemoryDataset):
    def __init__(self, root, dataset=None, transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        torch.load(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.dataset == 'zinc':
            return ['zinc250k_kekulized.npz']
        elif self.dataset == 'qm9':
            return ['qm9_kekulized.npz']
        else:
            raise NotImplementedError()

    @property
    def processed_file_names(self):
        if self.dataset == 'zinc':
            return ['zinc_data.pt']
        elif self.dataset == 'qm9':
            return ['data_qm9.pt']

    def download(self):
        # Download to `self.raw_dir`.
        if self.dataset == 'zinc':
            download_url('https://drive.switch.ch/index.php/s/D8ilMxpcXNHtVUb/download', self.raw_dir,
                         filename='zinc250k_kekulized.npz')
        elif self.dataset == 'qm9':
            download_url('https://drive.switch.ch/index.php/s/SESlx1ylQAopXsi/download', self.raw_dir,
                         filename='qm9_kekulized.npz')


    def process(self):
        if self.dataset == 'zinc':
            filepath = os.path.join(self.raw_dir, 'zinc250k_kekulized.npz')
            max_num_nodes = 38
        elif self.dataset == 'qm9':
            filepath = os.path.join(self.raw_dir, 'qm9_kekulized.npz')
            max_num_nodes = 9
        start = time.time()
        load_data = np.load(filepath, allow_pickle=True)
        xs = load_data['arr_0']
        adjs = load_data['arr_1']
        load_data = 0
        data_list = []

        for i, (x, adj) in enumerate(zip(xs, adjs)):
            x = atom_number_to_one_hot(x, self.dataset)
            edge_index, edge_attr = from_dense_numpy_to_sparse(adj)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
            if (i+1) % 1000 == 0:
                print(f'{i+1} graphs processed... process continue')

        print(f'{len(data_list)} graphs processed')
        data, slices = self.collate(data_list)
        data_list = 0
        print('Data collated')
        torch.save((data, slices), self.processed_paths[0])
        time_taken = time.time() - start
        print(f'Preprocessing took {time_taken} seconds')

class FromNetworkx(InMemoryDataset):
    def __init__(self, root, dataset=None, transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        torch.load(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    def __init__(self, root, dataset=None, transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        # torch.load(self.processed_paths[0])
        # import ipdb; ipdb.set_trace()
        # self.data, self.slices = torch.load(self.processed_paths[0])
        
        # 加载数据
        loaded_data = torch.load(self.processed_paths[0])
        import ipdb; ipdb.set_trace()
        # 处理不同的数据格式
        if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
            # 标准格式：(data, slices)
            self.data, self.slices = loaded_data
        else:
            # planar数据集格式：直接是数据列表
            print("Converting data format for planar dataset...")
            data_list = loaded_data
            if isinstance(data_list, list):
                # 转换每个图为PyG数据对象
                processed_list = []
                for g in data_list:
                    if not hasattr(g, 'edge_index'):  # 如果不是PyG数据对象
                        data = from_networkx(g)
                        data.max_num_nodes = 64  # planar数据集的最大节点数
                        processed_list.append(data)
                    else:
                        processed_list.append(g)
                
                # 使用collate合并数据
                self.data, self.slices = self.collate(processed_list)
                
                # 保存转换后的格式
                torch.save((self.data, self.slices), self.processed_paths[0])
            else:
                raise ValueError(f"Unexpected data format for {self.dataset} dataset")

            
    @property
    def raw_file_names(self):
        if self.dataset == 'ego':
            return ['ego_small.pkl']
            # return ['GDSS_ego.pkl']
        if self.dataset == 'citeseer':
            # return ['citeseer_r1.pkl']
            # return ['citeseer_r1_min3.pkl']
            return ['citeseer_r3_min4.pkl']
        elif self.dataset == 'community':
            return ['community_small.pkl']
        elif self.dataset == 'community_ego':
            return ['community_small.pkl']
        elif self.dataset == 'enzymes':
            return ['ENZYMES.pkl']
        elif self.dataset == 'grid':
            return ['grid.pkl']
        elif self.dataset == 'protein':
            # return ['PROTEINS_full_feat.pkl']
            return ['PROTEINS_small_500.pkl']
        elif self.dataset == 'planar':
            return ['planar.pkl']
            # return ['planar_64_200.pt']
        elif self.dataset == 'sbm':
            return ['sbm.pkl']
        else:
            raise NotImplementedError()

    def download(self):
        # Download to `self.raw_dir`.
        if self.dataset == 'ego':
            download_url('https://drive.switch.ch/index.php/s/KezKAJHY4bWNl9E/download', self.raw_dir,
                         filename='ego_small.pkl')
        elif self.dataset == 'community':
            download_url('https://drive.switch.ch/index.php/s/SLDFLYSBgsfV0ZA/download', self.raw_dir,
                         filename='community_small.pkl')

        elif self.dataset == 'enzymes':
            download_url('https://drive.switch.ch/index.php/s/dGo2OUFmOIqqDNt/download', self.raw_dir,
                         filename='ENZYMES.pkl')

    @property
    def processed_file_names(self):
        if self.dataset == 'ego':
            return ['ego_data.pt']
            # return ['ego.pt']
            # return ['GDSS_ego.pt']
        elif self.dataset == 'citeseer':
            # return ['citeseer.pt']
            # return ['citeseer_r1_min3.pt']
            return ['citeseer_r3_min4.pt']
        elif self.dataset == 'community':
            return ['community_data.pt']
        elif self.dataset == 'community_ego':
            return ['community_data_ego.pt']
        elif self.dataset == 'enzymes':
            return ['ENZYMES.pkl']
        elif self.dataset == 'grid':
            # return ['grid_data.pt']
            return ['grid.pt']
        elif self.dataset == 'planar':
            return ['planar_64_200.pt']
            # return ['planar_64_200.pt']
        elif self.dataset == 'sbm':
            return ['sbm_200.pt']
        elif self.dataset == 'protein':
            # return ['protein_feat.pt']
            return ['PROTEINS_small_500.pt']

    def process(self):
        filepath = os.path.join(self.raw_dir, self.raw_file_names[0])
        if self.dataset == 'planar' and filepath.endswith('.pt'):
            # 对于planar数据集的.pt文件，直接加载
            graph_list = torch.load(filepath)
        else:
            # 其他数据集使用pickle加载
            with open(filepath, 'rb') as pickle_file:
                graph_list = pickle.load(pickle_file)

        data_list = []
        # import ipdb; ipdb.set_trace()
        for g in graph_list:
            if not hasattr(g, 'edge_index'):  # 如果不是PyG数据对象
                data = from_networkx(g)
                if self.dataset == 'ego':
                    data.max_num_nodes = 18
                elif self.dataset == 'community':
                    data.max_num_nodes = 20
                elif self.dataset == 'community_ego':
                    data.max_num_nodes = 20
                elif self.dataset == 'enzymes':
                    data.max_num_nodes = 125
                elif self.dataset == 'grid':
                    data.max_num_nodes = 361
                elif self.dataset == 'protein':
                    data.max_num_nodes = 481
                elif self.dataset == 'citeseer':
                    data.max_num_nodes = 20
                elif self.dataset == 'planar':
                    data.max_num_nodes = 64
                elif self.dataset == 'sbm':
                    data.max_num_nodes = 187
                else:
                    raise NotImplementedError
                data_list.append(data)
            else:
                data_list.append(g)
        data, slices = self.collate(data_list)
        # print(f"{self.dataset} max_num_nodes: {data.max_num_nodes}")
        torch.save((data, slices), self.processed_paths[0])
        
        
    def get_local_subgraph(self, G, node, radius=1):
        """
        获取节点的k跳邻域子图
        """
        subgraph = nx.ego_graph(G, node, radius=radius, undirected=True)
        # 重新标记节点,使中心节点为0,其他节点按距离排序
        mapping = {node: 0}
        for d in range(1, radius + 1):
            neighbors = set(n for n in subgraph.nodes() 
                          if nx.shortest_path_length(subgraph, node, n) == d)
            mapping.update({n: len(mapping) for n in neighbors})
        return nx.relabel_nodes(subgraph, mapping)
    
    def compute_ged(self, G1, G2):
        """
        计算两个图之间的Graph Edit Distance
        """
        try:
            # 使用NetworkX的图编辑距离算法
            ged = nx.graph_edit_distance(G1, G2, 
                                       node_match=lambda n1, n2: True,  # 节点标签无关
                                       edge_match=lambda e1, e2: True)  # 边标签无关
            return ged if ged is not None else float('inf')
        except:
            return float('inf')
    
    def find_structural_groups(self, G, radius=1, ged_threshold=1):
        """
        找出所有局部结构相似的节点组
        Args:
            G: networkx图
            radius: 考虑的邻域半径
            ged_threshold: GED阈值,小于此值认为结构相同
        """
        nodes = list(G.nodes())
        n_nodes = len(nodes)
        groups = []
        used = set()
        
        # 对每个未分组的节点
        for i in range(n_nodes):
            if i in used:
                continue
                
            current_group = [i]
            subgraph_i = self.get_local_subgraph(G, nodes[i], radius)
            
            # 与其他未分组的节点比较
            for j in range(i + 1, n_nodes):
                if j in used:
                    continue
                    
                subgraph_j = self.get_local_subgraph(G, nodes[j], radius)
                ged = self.compute_ged(subgraph_i, subgraph_j)
                
                if ged <= ged_threshold:
                    current_group.append(j)
                    used.add(j)
            
            if len(current_group) > 1:  # 只保存有多个节点的组
                groups.append([nodes[idx] for idx in current_group])
                used.add(i)
                
        return groups
    
    def mark_structural_equivalence(self, radius=1, ged_threshold=1):
        """标记所有具有相同局部结构的节点"""
        import ipdb; ipdb.set_trace()
        for i, data in enumerate(self.data):
            # 转换为networkx图
            # edge_index = data.edge_index.numpy()
            edge_index = data[1].numpy()
            G = nx.Graph()
            G.add_edges_from(edge_index.T)
            
            # 找出结构相似的节点组
            structural_groups = self.find_structural_groups(G, radius, ged_threshold)
            data.structural_groups = structural_groups
            
            # 创建节点标签
            n_nodes = data.x.size(0)
            data.structure_labels = torch.zeros(n_nodes, dtype=torch.long)
            for group_idx, nodes in enumerate(structural_groups):
                for node in nodes:
                    data.structure_labels[node] = group_idx + 1
    

class SpectureDataset(Dataset):
    def __init__(self, nx_graphs, transform=None, dataset=None):
        super().__init__()
        self.graphs = nx_graphs
        self.transform = transform
        self.dataset = dataset
        # 预先转换所有图以提高效率
        self.data_list = [convert_nx_to_pyg(g,dataset) for g in nx_graphs]
        
        
    def len(self):
        return len(self.data_list)
        
    def get(self, idx):
        data = self.data_list[idx]
        
        if self.transform is not None:
            data = self.transform(data)
            
        return data

def convert_nx_to_pyg(nx_graph,dataset=None):
    # 确保图是连通的
    # largest_cc = max(nx.connected_components(nx_graph), key=len)
    # nx_graph = nx_graph.subgraph(largest_cc).copy()
    
    # 转换为PyG格式
    data = from_networkx(nx_graph)
    
    # 确保edge_index存在且格式正确
    if not hasattr(data, 'edge_index') or data.edge_index is None:
        raise ValueError("图转换后没有边信息(edge_index)")
    
    # 确保edge_index是LongTensor类型
    data.edge_index = data.edge_index.long()
    if dataset == 'planar':
        data.max_num_nodes = 64
    elif dataset == 'sbm':
        data.max_num_nodes = 187
    else:
        data.max_num_nodes = None
    
    # # 如果需要，可以添加其他图级别的属性
    # data.num_nodes = data.x.size(0)
    
    return data
