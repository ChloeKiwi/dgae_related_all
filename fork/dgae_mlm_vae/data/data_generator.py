import json
import logging
import os
import pickle
import networkx as nx
import numpy as np
import scipy.sparse as sp
import argparse

"""
create small synthetic dataset from a complete graph
"""
# -------- Generate community graphs --------
def n_community(num_communities, max_nodes, p_inter=0.05):
    # -------- From Niu et al. (2020) --------
    assert num_communities > 1
    
    one_community_size = max_nodes // num_communities
    c_sizes = [one_community_size] * num_communities
    total_nodes = one_community_size * num_communities
    p_make_a_bridge = p_inter * 2 / ((num_communities - 1) * one_community_size)
    
    print(num_communities, total_nodes, end=' ')
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.7, seed=i) for i in range(len(c_sizes))]

    G = nx.disjoint_union_all(graphs)
    communities = list(G.subgraph(c) for c in nx.connected_components(G))
    add_edge = 0
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i + 1, len(communities)):  # loop for C_M^2 times
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:  # loop for N times
                for n2 in nodes2:  # loop for N times
                    if np.random.rand() < p_make_a_bridge:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
                        add_edge += 1
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
                add_edge += 1
    print('connected comp: ', len( list(G.subgraph(c) for c in nx.connected_components(G)) ), 
            'add edges: ', add_edge)
    print(G.number_of_edges())
    return G


NAME_TO_NX_GENERATOR = {
    'community': n_community,
    # -------- Additional datasets --------
    'gnp': nx.generators.fast_gnp_random_graph,  # fast_gnp_random_graph(n, p, seed=None, directed=False)
    'ba': nx.generators.barabasi_albert_graph,  # barabasi_albert_graph(n, m, seed=None)
    'pow_law': lambda **kwargs: nx.configuration_model(nx.generators.random_powerlaw_tree_sequence(**kwargs, gamma=3,
                                                                                                   tries=2000)),
    'except_deg': lambda **kwargs: nx.expected_degree_graph(**kwargs, selfloops=False),
    'cycle': nx.cycle_graph,
    'c_l': nx.circular_ladder_graph,
    'lobster': nx.random_lobster
}


class GraphGenerator:
    def __init__(self, graph_type='community', possible_params_dict=None, corrupt_func=None):
        if possible_params_dict is None:
            possible_params_dict = {}
        assert isinstance(possible_params_dict, dict)
        self.count = {k: 0 for k in possible_params_dict}
        self.possible_params = possible_params_dict
        self.corrupt_func = corrupt_func
        self.nx_generator = NAME_TO_NX_GENERATOR[graph_type]

    def __call__(self):
        params = {}
        for k, v_list in self.possible_params.items():
            params[k] = np.random.choice(v_list)
        graph = self.nx_generator(**params)
        graph = nx.relabel.convert_node_labels_to_integers(graph)
        if self.corrupt_func is not None:
            graph = self.corrupt_func(self.corrupt_func)
        return graph


# -------- Generate synthetic graphs --------
def gen_graph_list(graph_type='community', possible_params_dict=None, corrupt_func=None, length=1024, save_dir=None,
                   file_name=None, max_node=None, min_node=None):
    params = locals()
    logging.info('gen data: ' + json.dumps(params))
    if file_name is None:
        file_name = graph_type + '_' + str(length)
    file_path = os.path.join(save_dir, file_name)
    graph_generator = GraphGenerator(graph_type=graph_type,
                                     possible_params_dict=possible_params_dict,
                                     corrupt_func=corrupt_func)
    graph_list = []
    i = 0
    max_N = 0
    while i < length:
        graph = graph_generator()
        if max_node is not None and graph.number_of_nodes() > max_node:
            continue
        if min_node is not None and graph.number_of_nodes() < min_node:
            continue
        print(i, graph.number_of_nodes(), graph.number_of_edges())
        max_N = max(max_N, graph.number_of_nodes())
        if graph.number_of_nodes() <= 1:
            continue
        graph_list.append(graph)
        i += 1
    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(file_path + '.pkl', 'wb') as f:
            pickle.dump(obj=graph_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(file_path + '.txt', 'w') as f:
            f.write(json.dumps(params))
            f.write(f'max node number: {max_N}')
    print(max_N)
    return graph_list


def load_dataset(data_dir='data', file_name=None, need_set=False):
    file_path = os.path.join(data_dir, file_name)
    with open(file_path + '.pkl', 'rb') as f:
        graph_list = pickle.load(f)
    return graph_list 


# -------- load ENZYMES, PROTEIN and DD dataset --------
def graph_load_batch(min_num_nodes=20, max_num_nodes=1000, name='ENZYMES', node_attributes=True, graph_labels=True):
    """
    load many graphs, e.g. enzymes
    :return: a list of graphs
    """
    print('Loading graph dataset: ' + str(name))
    G = nx.Graph()
    # -------- load data --------
    path = 'dataset/' + name + '/'
    data_adj = np.loadtxt(path + name + '_A.txt', delimiter=',').astype(int)
    data_node_att = []
    if node_attributes:
        data_node_att = np.loadtxt(path + name + '_node_attributes.txt', delimiter=',')
    data_node_label = np.loadtxt(path + name + '_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(path + name + '_graph_indicator.txt', delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(path + name + '_graph_labels.txt', delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))

    # -------- add edges --------
    G.add_edges_from(data_tuple)

    # -------- add node attributes --------
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i + 1, feature=data_node_att[i])
        G.add_node(i + 1, label=data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    print(G.number_of_nodes())
    print(G.number_of_edges())

    # -------- split into graphs --------
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # -------- find the nodes for each graph --------
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['label'] = data_graph_labels[i]
        if min_num_nodes <= G_sub.number_of_nodes() <= max_num_nodes:
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
    print(f'Graphs loaded, total num: {len(graphs)}')
    return graphs


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


# -------- load cora, citeseer and pubmed dataset --------
def graph_load(dataset='cora'):
    """
    Load a single graph dataset
    :param dataset: dataset name
    :return:
    """
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        load = pickle.load(open("/CIS_LAB/yqy/fork/dgae_mlm_benchmark/data/data/ind.{}.{}".format(dataset, names[i]), 'rb'), encoding='latin1')
        objects.append(load)
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("/CIS_LAB/yqy/fork/dgae_mlm_benchmark/data/data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)
    
    # 加载标签
    y = pickle.load(open(f"/CIS_LAB/yqy/fork/dgae_mlm_benchmark/data/data/ind.{dataset}.y",'rb'),encoding='latin1')
    ty = pickle.load(open(f"/CIS_LAB/yqy/fork/dgae_mlm_benchmark/data/data/ind.{dataset}.ty",'rb'),encoding='latin1')
    ally = pickle.load(open(f"/CIS_LAB/yqy/fork/dgae_mlm_benchmark/data/data/ind.{dataset}.ally",'rb'),encoding='latin1')

    test_idx_reorder = parse_index_file(f"/CIS_LAB/yqy/fork/dgae_mlm_benchmark/data/data/ind.{dataset}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer' or dataset == 'cora':
        # -------- Fix citeseer dataset (there are some isolated nodes in the graph) --------
        # -------- Find isolated nodes, add them as zero-vecs into the right position --------
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil() #3327*3703
    features[test_idx_reorder, :] = features[test_idx_range, :]
    
    # 合并标签
    labels = np.vstack((ally, ty)) #3327*6
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    G = nx.from_dict_of_lists(graph)
    return features, labels, G


def citeseer_ego(radius=3, node_min=50, node_max=400):
    features, labels, G = graph_load(dataset='citeseer')  # nodes:3327
    features = features.toarray()
    
    G = max([G.subgraph(c) for c in nx.connected_components(G)], key=len)
    
    # 在重新编号之前，保存原始节点ID到特征和标签的映射
    old_to_feature = {node: features[node] for node in G.nodes()}
    old_to_label = {node: np.argmax(labels[node]) for node in G.nodes()}  # 将one-hot转换为类别索引
    
    # 手动创建节点映射并重新编号
    mapping = {old: new for new, old in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    # 将特征和标签添加到重新编号后的图中
    for new_id in G.nodes():
        old_id = {v: k for k, v in mapping.items()}[new_id]  # 反向查找原始节点ID
        G.nodes[new_id]['feature'] = old_to_feature[old_id]
        G.nodes[new_id]['label'] = old_to_label[old_id]
    
    # # 将特征和标签添加到重新编号后的图中
    # for new_id, old_id in G.graph['node_labels'].items():
    #     G.nodes[new_id]['feature'] = old_to_feature[old_id]
    #     G.nodes[new_id]['label'] = old_to_label[old_id]
        
    graphs = []
    for i in range(G.number_of_nodes()):
        G_ego = nx.ego_graph(G, i, radius=radius) #获取以i为中心，半径为radius的子图
        assert isinstance(G_ego, nx.Graph)
        if G_ego.number_of_nodes() >= node_min and (G_ego.number_of_nodes() <= node_max):
            G_ego.remove_edges_from(nx.selfloop_edges(G_ego))
            # 确保子图保留节点特征和标签
            for node in G_ego.nodes():
                G_ego.nodes[node]['feature'] = G.nodes[node]['feature']
                G_ego.nodes[node]['label'] = G.nodes[node]['label']
            graphs.append(G_ego)
    return graphs

def cora_ego(radius=3, node_min=50, node_max=400):
    """
    从Cora数据集生成ego-networks
    Args:
        radius: ego-network的半径
        node_min: 子图最小节点数
        node_max: 子图最大节点数
    Returns:
        graphs: 符合大小要求的ego-networks列表
    """
    features, labels, G = graph_load(dataset='cora')  # nodes:3327
    features = features.toarray()
    
    # 获取最大连通分量
    G = max([G.subgraph(c) for c in nx.connected_components(G)], key=len)
    
    # 保存原始节点ID到特征和标签的映射
    old_to_feature = {node: features[node] for node in G.nodes()}
    old_to_label = {node: np.argmax(labels[node]) for node in G.nodes()}  # 将one-hot转换为类别索引
    
    # 重新编号节点
    mapping = {old: new for new, old in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    # 将特征和标签添加到重新编号后的图中
    for new_id in G.nodes():
        old_id = {v: k for k, v in mapping.items()}[new_id]  # 反向查找原始节点ID
        G.nodes[new_id]['feature'] = old_to_feature[old_id]
        G.nodes[new_id]['label'] = old_to_label[old_id]
    
    # 生成ego-networks
    graphs = []
    for i in range(G.number_of_nodes()):
        G_ego = nx.ego_graph(G, i, radius=radius)  # 获取以i为中心，半径为radius的子图
        if node_min <= G_ego.number_of_nodes() <= node_max:
            # 移除自环
            G_ego.remove_edges_from(nx.selfloop_edges(G_ego))
            
            # 确保子图保留节点特征和标签
            for node in G_ego.nodes():
                G_ego.nodes[node]['feature'] = G.nodes[node]['feature']
                G_ego.nodes[node]['label'] = G.nodes[node]['label']
            
            graphs.append(G_ego)
    
    print(f"生成的ego-networks统计信息:")
    print(f"总数量: {len(graphs)}")
    print(f"平均节点数: {np.mean([g.number_of_nodes() for g in graphs]):.2f}")
    print(f"平均边数: {np.mean([g.number_of_edges() for g in graphs]):.2f}")
    print(f"最大节点数: {max([g.number_of_nodes() for g in graphs])}")
    return graphs

def save_dataset(data_dir, graphs, save_name):
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    file_path = os.path.join('data', save_name)
    print(f"{save_name}, total graphs: {len(graphs)}")
    print(f"Node feature dimension: {graphs[0].nodes[0]['feature'].shape}")
    print(f"Number of unique labels: {len(set(data['label'] for g in graphs for _, data in g.nodes(data=True)))}")
    
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(obj=graphs, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_path + '.txt', 'w') as f:
        f.write(save_name + '\n')
        f.write(str(len(graphs)))


# -------- Generate datasets --------
def generate_dataset(data_dir='data', dataset='community_small'):

    if dataset == 'community_small':
        res_graph_list = gen_graph_list(graph_type='community', possible_params_dict={
                                        'num_communities': [2],
                                        'max_nodes': np.arange(12, 21).tolist()},
                                        corrupt_func=None, length=100, save_dir=data_dir, file_name=dataset)

    elif dataset == 'ego_small': #! citeseer
        # import ipdb; ipdb.set_trace()
        graphs = citeseer_ego(radius=1, node_min=4, node_max=18)[:200] #only 200 graphs
        save_dataset(data_dir, graphs, dataset)
        print(max([g.number_of_nodes() for g in graphs]))

    elif dataset == 'ENZYMES':
        graphs = graph_load_batch(min_num_nodes=10, max_num_nodes=1000, name=dataset,
                                    node_attributes=False, graph_labels=True)
        save_dataset(data_dir, graphs, dataset)
        print(max([g.number_of_nodes() for g in graphs]))
        
    elif dataset == 'cora':
        graphs = cora_ego(radius=1, node_min=4, node_max=18)[:200] #only 100 graphs
        save_dataset(data_dir, graphs, dataset)
        print(max([g.number_of_nodes() for g in graphs]))

    else:
        raise NotImplementedError(f'Dataset {dataset} not supproted.')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('--data-dir', type=str, default='data', help='directory to save the generated dataset')
    parser.add_argument('--dataset', type=str, default='community_small', help='dataset to generate',
                        choices=['ego_small', 'community_small', 'ENZYMES'])
    args = parser.parse_args()
    args.data_dir = '/CIS_LAB/yqy/fork/dgae_mlm_benchmark/data'
    # args.dataset = 'ego_small'
    args.dataset = 'cora'
    generate_dataset(args.data_dir, args.dataset)
