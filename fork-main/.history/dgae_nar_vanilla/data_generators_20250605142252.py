import json
import logging
import os
import pickle
import networkx as nx
import numpy as np
import scipy.sparse as sp
import argparse


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
    'grid': nx.generators.grid_2d_graph,  
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
    def __init__(self, graph_type='grid', possible_params_dict=None, corrupt_func=None):
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
        graph = nx.relabel.convert_node_labels_to_integers(graph) #this means the node id is from 0 to N-1
        if self.corrupt_func is not None:
            graph = self.corrupt_func(self.corrupt_func)
        return graph


# -------- Generate synthetic graphs --------
def gen_graph_list(graph_type='grid', possible_params_dict=None, corrupt_func=None, length=1024, save_dir=None,
                   file_name=None, max_node=None, min_node=None):
    params = locals()
    # logging.info('gen data: ' + json.dumps(params))
    if file_name is None:
        file_name = graph_type + '_' + str(length)
    file_path = os.path.join(save_dir, file_name)
    graph_generator = GraphGenerator(graph_type=graph_type,
                                     possible_params_dict=possible_params_dict,
                                     corrupt_func=corrupt_func)
    # import ipdb; ipdb.set_trace()
    graph_list = []
    i = 0
    max_N = 0
    while i < length:
        graph = graph_generator()
        if max_node is not None and graph.number_of_nodes() > max_node:
            continue
        if min_node is not None and graph.number_of_nodes() < min_node:
            continue
        print(i, graph.number_of_nodes(), graph.number_of_edges()) #0 221 412
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
            # f.write(json.dumps(params))
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
            # G.add_node(i + 1, feature=data_node_att[i])
            G.add_node(i + 1, x=data_node_att[i])
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
        nodes = node_list[data_graph_indicator == i + 1] #0-42个节点对应第一个图
        G_sub = G.subgraph(nodes) #42个节点构成的一个子图
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
        load = pickle.load(open("/mnt/diskLv/yy/yqy/fork/dgae-main/dataset/ind.{}.{}".format(dataset, names[i]), 'rb'), encoding='latin1')
        objects.append(load)
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("/mnt/diskLv/yy/yqy/fork/dgae-main/dataset/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)
    
    # 加载标签
    y = pickle.load(open(f"/mnt/diskLv/yy/yqy/fork/dgae-main/dataset/ind.{dataset}.y",'rb'),encoding='latin1')
    ty = pickle.load(open(f"/mnt/diskLv/yy/yqy/fork/dgae-main/dataset/ind.{dataset}.ty",'rb'),encoding='latin1')
    ally = pickle.load(open(f"/mnt/diskLv/yy/yqy/fork/dgae-main/dataset/ind.{dataset}.ally",'rb'),encoding='latin1')

    test_idx_reorder = parse_index_file(f"/mnt/diskLv/yy/yqy/fork/dgae-main/dataset/ind.{dataset}.test.index")
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


def save_dataset(data_dir, graphs, save_name):
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    file_path = os.path.join('./data/raw', save_name)
    print(f"{save_name}: {len(graphs)} graphs")
    
    # 保存一些统计信息
    stats = {
        'num_graphs': len(graphs),
        'max_num_nodes': max([g.number_of_nodes() for g in graphs]),
        'min_num_nodes': min([g.number_of_nodes() for g in graphs]),
        'avg_nodes': np.mean([g.number_of_nodes() for g in graphs]),
        'avg_edges': np.mean([g.number_of_edges() for g in graphs]),
        'num_classes': len(set([data['label'] for g in graphs for _, data in g.nodes(data=True)]))
    }
    
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
                                        corrupt_func=None, length=100, save_dir=data_dir, file_name=dataset) #! only 100 graphs
    elif dataset == 'grid':
        res_graph_list = gen_graph_list(graph_type='grid', possible_params_dict={
                                        'm': np.arange(10, 20).tolist(), #means the number of rows
                                        'n': np.arange(10, 20).tolist()},  #
                                        corrupt_func=None, length=100, save_dir=data_dir, file_name=dataset)
    
    # 1k grid
    elif dataset == 'grid_1k':
         # 生成约1000节点的grid图
        res_graph_list = gen_graph_list(graph_type='grid', possible_params_dict={
                                        'm': [31, 32, 33],  # 行数
                                        'n': [31, 32, 33]}, # 列数  
                                        corrupt_func=None, length=100, save_dir=data_dir, file_name=dataset)

    elif dataset == 'ego_small':
        graphs = citeseer_ego(radius=1, node_min=4, node_max=18)[:200] #! radius
        save_dataset(data_dir, graphs, dataset)
        print(max([g.number_of_nodes() for g in graphs]))

    elif dataset == 'ENZYMES' or dataset=='PROTEINS_full':
        graphs = graph_load_batch(min_num_nodes=13, max_num_nodes=500, name=dataset,
                                    node_attributes=True, graph_labels=True)
        save_dataset(data_dir, graphs[:600], 'PROTEINS_small_500')
        print(max([g.number_of_nodes() for g in graphs]))
    
    elif dataset == 'citeseer':
        graphs = citeseer(radius=3, node_min=4, node_max=20)
    
    elif dataset == 'ba':
        res_graph_list = gen_graph_list(graph_type='ba', possible_params_dict={
                                    'n': np.arange(20, 100).tolist(),  # 节点数
                                    'm': np.arange(1, 5).tolist()},     # 每次添加的边数
                                    # corrupt_func=None, length=100, save_dir=data_dir, file_name=dataset)
                                    corrupt_func=None, length=1000, save_dir=data_dir, file_name=dataset)
        
    elif dataset == 'ba_large':
        res_graph_list = gen_graph_list(graph_type='ba', possible_params_dict={
                                    'n': np.arange(500, 2000).tolist(),  # 节点数
                                    'm': np.arange(2, 10).tolist()},     # 每次添加的边数
                                    corrupt_func=None, length=100, save_dir=data_dir, file_name=dataset)
    
    elif dataset == 'gnp_large':
        res_graph_list = gen_graph_list(graph_type='gnp', possible_params_dict={
                                    'n': np.arange(500, 2000).tolist(),  # 节点数
                                    'p': [0.01, 0.02, 0.05, 0.1]},       # 连接概率
                                    corrupt_func=None, length=100, save_dir=data_dir, file_name=dataset)
        
    elif dataset == 'gnp':
        res_graph_list = gen_graph_list(graph_type='gnp', possible_params_dict={
                                    'n': np.arange(20, 100).tolist(),  # 节点数
                                    'p': [0.05, 0.1, 0.15, 0.2]},       # 连接概率
                                    corrupt_func=None, length=100, save_dir=data_dir, file_name=dataset)
    
    # 幂律树 (Power-law Tree) - 通常使用200-1000节点
    elif dataset == 'pow_law_tree':
        res_graph_list = gen_graph_list(graph_type='pow_law', possible_params_dict={
                                        'n': np.arange(200, 800).tolist(),  # 节点数
                                        'gamma': [2.5, 3.0, 3.5]},          # 幂律指数
                                        corrupt_func=None, length=100, save_dir=data_dir, file_name=dataset)
        
    # 环图 (Cycle Graph) - 通常使用50-500节点
    elif dataset == 'cycle':
        res_graph_list = gen_graph_list(graph_type='cycle', possible_params_dict={
                                        'n': np.arange(50, 500).tolist()},  # 节点数
                                        corrupt_func=None, length=100, save_dir=data_dir, file_name=dataset)

    # 圆形梯子图 (Circular Ladder Graph) - 通常使用25-250节点（总节点数为2n）
    elif dataset == 'circular_ladder':
        res_graph_list = gen_graph_list(graph_type='c_l', possible_params_dict={
                                        'n': np.arange(25, 250).tolist()},  # 每个环的节点数，总节点数为2n
                                        corrupt_func=None, length=100, save_dir=data_dir, file_name=dataset)

    # 龙虾图 (Lobster Graph) - 通常使用50-300主干节点
    elif dataset == 'lobster':
        res_graph_list = gen_graph_list(graph_type='lobster', possible_params_dict={
                                        'n': np.arange(50, 300).tolist(),   # 主干节点数
                                        'p1': [0.3, 0.4, 0.5, 0.6, 0.7],   # 添加主干边的概率
                                        'p2': [0.2, 0.3, 0.4, 0.5]},       # 添加二级边的概率
                                        corrupt_func=None, length=100, save_dir=data_dir, file_name=dataset)
    
    
    else:
        raise NotImplementedError(f'Dataset {dataset} not supproted.')
    
    
def eliminate_self_loops_adj(A):
    """Remove self-loops from the adjacency matrix."""
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A

def largest_connected_components(sparse_graph, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = sp.csgraph.connected_components(sparse_graph.adj_matrix)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][
        :n_components
    ]  # reverse order to sort descending
    nodes_to_keep = [
        idx
        for (idx, component) in enumerate(component_indices)
        if component in components_to_keep
    ]
    return create_subgraph(sparse_graph, nodes_to_keep=nodes_to_keep)
    
    
def create_subgraph(
    sparse_graph, _sentinel=None, nodes_to_remove=None, nodes_to_keep=None
):
    """Create a graph with the specified subset of nodes.

    Exactly one of (nodes_to_remove, nodes_to_keep) should be provided, while the other stays None.
    Note that to avoid confusion, it is required to pass node indices as named arguments to this function.

    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    _sentinel : None
        Internal, to prevent passing positional arguments. Do not use.
    nodes_to_remove : array-like of int
        Indices of nodes that have to removed.
    nodes_to_keep : array-like of int
        Indices of nodes that have to be kept.

    Returns
    -------
    sparse_graph : SparseGraph
        Graph with specified nodes removed.

    """
    # Check that arguments are passed correctly
    if _sentinel is not None:
        raise ValueError(
            "Only call `create_subgraph` with named arguments',"
            " (nodes_to_remove=...) or (nodes_to_keep=...)"
        )
    if nodes_to_remove is None and nodes_to_keep is None:
        raise ValueError("Either nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None and nodes_to_keep is not None:
        raise ValueError(
            "Only one of nodes_to_remove or nodes_to_keep must be provided."
        )
    elif nodes_to_remove is not None:
        nodes_to_keep = [
            i for i in range(sparse_graph.num_nodes()) if i not in nodes_to_remove
        ]
    elif nodes_to_keep is not None:
        nodes_to_keep = sorted(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    sparse_graph.adj_matrix = sparse_graph.adj_matrix[nodes_to_keep][:, nodes_to_keep]
    if sparse_graph.attr_matrix is not None:
        sparse_graph.attr_matrix = sparse_graph.attr_matrix[nodes_to_keep]
    if sparse_graph.labels is not None:
        sparse_graph.labels = sparse_graph.labels[nodes_to_keep]
    if sparse_graph.node_names is not None:
        sparse_graph.node_names = sparse_graph.node_names[nodes_to_keep]
    return sparse_graph

class SparseGraph:
    """Attributed labeled graph stored in sparse matrix form."""

    def __init__(
        self,
        adj_matrix,
        attr_matrix=None,
        labels=None,
        node_names=None,
        attr_names=None,
        class_names=None,
        metadata=None,
    ):
        """Create an attributed graph.

        Parameters
        ----------
        adj_matrix : sp.csr_matrix, shape [num_nodes, num_nodes]
            Adjacency matrix in CSR format.
        attr_matrix : sp.csr_matrix or np.ndarray, shape [num_nodes, num_attr], optional
            Attribute matrix in CSR or numpy format.
        labels : np.ndarray, shape [num_nodes], optional
            Array, where each entry represents respective node's label(s).
        node_names : np.ndarray, shape [num_nodes], optional
            Names of nodes (as strings).
        attr_names : np.ndarray, shape [num_attr]
            Names of the attributes (as strings).
        class_names : np.ndarray, shape [num_classes], optional
            Names of the class labels (as strings).
        metadata : object
            Additional metadata such as text.

        """
        # Make sure that the dimensions of matrices / arrays all agree
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError(
                "Adjacency matrix must be in sparse format (got {0} instead)".format(
                    type(adj_matrix)
                )
            )

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError(
                    "Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)".format(
                        type(attr_matrix)
                    )
                )

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError(
                    "Dimensions of the adjacency and attribute matrices don't agree"
                )

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError(
                    "Dimensions of the adjacency matrix and the label vector don't agree"
                )

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError(
                    "Dimensions of the adjacency matrix and the node names don't agree"
                )

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError(
                    "Dimensions of the attribute matrix and the attribute names don't agree"
                )

        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels
        self.node_names = node_names
        self.attr_names = attr_names
        self.class_names = class_names
        self.metadata = metadata

    def num_nodes(self):
        """Get the number of nodes in the graph."""
        return self.adj_matrix.shape[0]

    def num_edges(self):
        """Get the number of edges in the graph.

        For undirected graphs, (i, j) and (j, i) are counted as single edge.
        """
        if self.is_directed():
            return int(self.adj_matrix.nnz)
        else:
            return int(self.adj_matrix.nnz / 2)

    def get_neighbors(self, idx):
        """Get the indices of neighbors of a given node.

        Parameters
        ----------
        idx : int
            Index of the node whose neighbors are of interest.

        """
        return self.adj_matrix[idx].indices

    def is_directed(self):
        """Check if the graph is directed (adjacency matrix is not symmetric)."""
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def to_undirected(self):
        """Convert to an undirected graph (make adjacency matrix symmetric)."""
        if self.is_weighted():
            raise ValueError("Convert to unweighted graph first.")
        else:
            self.adj_matrix = self.adj_matrix + self.adj_matrix.T
            self.adj_matrix[self.adj_matrix != 0] = 1
        return self

    def is_weighted(self):
        """Check if the graph is weighted (edge weights other than 1)."""
        return np.any(np.unique(self.adj_matrix[self.adj_matrix != 0].A1) != 1)

    def to_unweighted(self):
        """Convert to an unweighted graph (set all edge weights to 1)."""
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        return self

    # Quality of life (shortcuts)
    def standardize(self):
        """Select the LCC of the unweighted/undirected/no-self-loop graph.

        All changes are done inplace.

        """
        G = self.to_unweighted().to_undirected() #make the graph unweighted and undirected  
        G.adj_matrix = eliminate_self_loops_adj(G.adj_matrix) #eliminate self loops
        G = largest_connected_components(G, 1) #select the largest connected component
        return G

    def unpack(self):
        """Return the (A, X, z) triplet."""
        return self.adj_matrix, self.attr_matrix, self.labels

def load_npz_to_sparse_graph(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.

    """
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix(
            (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
            shape=loader["adj_shape"],
        )

        if "attr_data" in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix(
                (loader["attr_data"], loader["attr_indices"], loader["attr_indptr"]),
                shape=loader["attr_shape"],
            )
        elif "attr_matrix" in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader["attr_matrix"]
        else:
            attr_matrix = None

        if "labels_data" in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix(
                (
                    loader["labels_data"],
                    loader["labels_indices"],
                    loader["labels_indptr"],
                ),
                shape=loader["labels_shape"],
            )
        elif "labels" in loader:
            # Labels are stored as a numpy array
            labels = loader["labels"]
        else:
            labels = None

        node_names = loader.get("node_names")
        attr_names = loader.get("attr_names")
        class_names = loader.get("class_names")
        metadata = loader.get("metadata")
        
        return SparseGraph(
        adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata
    )
        
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):
    adj = normalize(adj + sp.eye(adj.shape[0]))
    return adj

def citeseer(radius,node_min=2,node_max=18):
    import ipdb; ipdb.set_trace()
    # data = np.load('./dataset/citeseer.npz')
    # print(data.files)
    data_path = "./dataset/citeseer.npz"
    if os.path.isfile(data_path): #isfile() 方法用于判断指定路径是否为文件
        data = load_npz_to_sparse_graph(data_path)
    else:
        raise ValueError(f"{data_path} doesn't exist.")
    
    # 标准化处理(移除自环、提取最大连通分量)
    data = data.standardize()
    adj, features, labels = data.unpack()
    adj = normalize_adj(adj)
    
    # 转换为NetworkX图
    G = nx.from_scipy_sparse_array(adj)
    
     # 添加节点特征和标签
    for i in range(G.number_of_nodes()):
        G.nodes[i]['x'] = features[i].toarray().flatten() if sp.issparse(features) else features[i]
        G.nodes[i]['label'] = np.argmax(labels[i]) if len(labels[i].shape) > 0 else labels[i]
     
    # 提取ego-networks作为子图
    graphs = []
    for i in G.nodes():
        G_ego = nx.ego_graph(G, i, radius=radius)
        if G_ego.number_of_nodes() >= node_min and G_ego.number_of_nodes() <= node_max:
            G_ego.remove_edges_from(nx.selfloop_edges(G_ego))
            # 确保子图保留节点特征和标签
            for node in G_ego.nodes():
                G_ego.nodes[node]['x'] = G.nodes[node]['x']
                G_ego.nodes[node]['label'] = G.nodes[node]['label']
            G_ego.graph['label']=G.nodes[i]['label'] # # 使用中心节点i的标签作为图标签
            graphs.append(G_ego)
    
    print(f"Total number of ego-networks: {len(graphs)}")
    print(f"Node range: {min([g.number_of_nodes() for g in graphs])} - {max([g.number_of_nodes() for g in graphs])}")
    
    # 保存处理后的图
    save_name = f'citeseer_r{radius}_min{node_min}'
    save_dataset('data', graphs, save_name)
    
    return graphs
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('--data-dir', type=str, default='./data/raw', help='directory to save the generated dataset')
    parser.add_argument('--dataset', type=str, default='community_small', help='dataset to generate',
                        choices=['ego_small', 'community_small', 'ENZYMES', 'grid', 'citeseer'])
    args = parser.parse_args()
    # args.dataset='ego_small'
    # args.dataset='grid' 
    # args.dataset='grid_1k'
    args.dataset='PROTEINS_full'
    generate_dataset(args.data_dir, args.dataset)
    # import ipdb; ipdb.set_trace()
    
    
    # # 生成BA图      
    # args.dataset='ba_large'
    # generate_dataset(args.data_dir, args.dataset)
    
    # # 生成GNP图
    # args.dataset='gnp_large'
    # generate_dataset(args.data_dir, args.dataset)
    
    # args.dataset='ba'
    # generate_dataset(args.data_dir, args.dataset)
    
    # # 生成GNP图
    # args.dataset='gnp'
    # generate_dataset(args.data_dir, args.dataset)
    
    # # 生成proteins图
    # args.dataset='PROTEINS_full'
    # generate_dataset(args.data_dir, args.dataset)
    
    # args.dataset='pow_law_tree' #error
    # generate_dataset(args.data_dir, args.dataset)
    
    # args.dataset='cycle'
    # generate_dataset(args.data_dir, args.dataset)
    
    # args.dataset='circular_ladder'
    # generate_dataset(args.data_dir, args.dataset)
    
    # args.dataset='lobster'
    # generate_dataset(args.data_dir, args.dataset)
    
    