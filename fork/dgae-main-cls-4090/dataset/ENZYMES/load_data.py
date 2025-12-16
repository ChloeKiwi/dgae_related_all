import numpy as np
import networkx as nx
import pickle
import scipy.sparse as sp
import argparse
import os

G = nx.Graph()
# load data
data_adj = np.loadtxt('ENZYMES_A.txt', delimiter=',').astype(int)
data_node_att = np.loadtxt('ENZYMES_node_attributes.txt', delimiter=',')
data_node_label = np.loadtxt('ENZYMES_node_labels.txt', delimiter=',').astype(int)
data_graph_indicator = np.loadtxt('ENZYMES_graph_indicator.txt', delimiter=',').astype(int)
data_graph_labels = np.loadtxt('ENZYMES_graph_labels.txt', delimiter=',').astype(int)

data_tuple = list(map(tuple, data_adj))
print(len(data_tuple))
print(data_tuple[0])

# add edges
G.add_edges_from(data_tuple)
# add node attributes
for i in range(data_node_att.shape[0]):
    # G.add_node(i+1, feature = data_node_att[i])
    G.add_node(i+1, x = data_node_att[i])
    G.add_node(i+1, label = data_node_label[i])
    
import ipdb; ipdb.set_trace()

isolates = list(nx.isolates(G))
G.remove_nodes_from(isolates)

print(G.number_of_nodes())
print(G.number_of_edges())

# split into graphs
graph_num = 600
node_list = np.arange(data_graph_indicator.shape[0])+1
graphs = []
node_num_list = []
edge_num_list = []
for i in range(graph_num):
    # find the nodes for each graph
    nodes = node_list[data_graph_indicator==i+1]
    G_sub = G.subgraph(nodes)
    graphs.append(G_sub)
    G_sub.graph['label'] = data_graph_labels[i]
    # print('nodes', G_sub.number_of_nodes())
    # print('edges', G_sub.number_of_edges())
    # print('label', G_sub.graph)
    node_num_list.append(G_sub.number_of_nodes())
    edge_num_list.append(G_sub.number_of_edges())
    print(f"node:{G_sub.number_of_nodes()}, edges:{G_sub.number_of_edges()}")
print('average nodes', sum(node_num_list)/len(node_num_list))
print('max nodes',max(node_num_list))
print('all nodes', len(node_num_list))

print('average edges', sum(edge_num_list)/len(edge_num_list))
print('max edges', max(edge_num_list))
print('all edges', len(edge_num_list))

# node_num_list = np.array(node_num_list)
# print('selected', len(node_num_list[node_num_list>10]))
# print(graphs[0].nodes(data=True)[0][1]['feature'])
# print(graphs[0].nodes())
keys = tuple(graphs[0].nodes())
# print(nx.get_node_attributes(graphs[0], 'feature'))
# dictionary = nx.get_node_attributes(graphs[0], 'feature')
dictionary = nx.get_node_attributes(graphs[0], 'x')
# print('keys', keys)
# print('keys from dict', list(dictionary.keys()))
# print('valuse from dict', list(dictionary.values()))

features = np.zeros((len(dictionary), list(dictionary.values())[0].shape[0]))
for i in range(len(dictionary)):
    features[i,:] = list(dictionary.values())[i]
# print(features)
# print(features.shape)

# import ipdb; ipdb.set_trace()
# data_dir = '/yqy/fork/dgae-main-cls/data/raw'
# save_name = 'ENZYMES_attr'
# file_path = os.path.join(data_dir, save_name)
# if not os.path.isdir(data_dir):
#     os.makedirs(data_dir)
#     file_path = os.path.join(data_dir, save_name)
#     print(f"{save_name}: {len(graphs)} graphs")
    
# with open(file_path + '.pkl', 'wb') as f:
#     pickle.dump(obj=graphs, file=f, protocol=pickle.HIGHEST_PROTOCOL)
# with open(file_path + '.txt', 'w') as f:
#     f.write(save_name + '\n')
#     f.write(str(len(graphs)))