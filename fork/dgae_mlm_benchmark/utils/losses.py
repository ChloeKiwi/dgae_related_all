import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_dense_batch
from utils.func import discretize, get_edge_target



def get_losses(batch, nodes_rec, edges_rec, node_masks, annotated_nodes,
               annotated_edges, max_node_num, n_node_feat, edge_masks, is_mol_data=False):
    # import ipdb; ipdb.set_trace()
    if annotated_nodes:
        node_loss, nodes_rec, nodes_target = get_node_loss_and_recon(batch, nodes_rec, node_masks,
                                                                     max_node_num, n_node_feat, annotated_nodes, is_mol_data)

    if annotated_edges:
        edge_loss, edges_rec = get_edge_loss_and_recon(batch, edges_rec, edge_masks)
    else:
        edge_loss, edges_rec = get_unannotated_loss_and_rec(batch, edges_rec, edge_masks, max_node_num)

    # Calculate the total loss
    if annotated_nodes:
        tot = node_masks.sum() + edge_masks.sum()
        recon_loss = (edge_masks.sum() / tot) * edge_loss + (node_masks.sum() / tot) * node_loss
    else:
        recon_loss = edge_loss
        nodes_rec = None
        node_loss = None
    return recon_loss, (node_loss, edge_loss, nodes_rec, edges_rec)



def get_node_loss_and_recon(batch, nodes_rec, node_masks, max_node_num, n_node_feat, annotated_nodes=True, is_mol_data=False):
    '''
    Return the loss for the node features and the reconstructed and discretized instance
    ''' 
    dense_nodes, _ = to_dense_batch(batch.x, batch.batch, max_num_nodes=max_node_num) #todo: batch.x 和 batch.feature 不一样
    if is_mol_data:
        target = dense_nodes[:, :, :n_node_feat].argmax(-1) #bs,max_node_num
        nodes_rec = nodes_rec.log_softmax(-1) * node_masks
        node_loss = F.nll_loss(nodes_rec.permute(0, 2, 1).contiguous(), target, reduction='none') 
    elif not is_mol_data and annotated_nodes: #非mol但有node feat
        target = dense_nodes[:, :, :n_node_feat] #bs,max_node_num,n_node_feat
        nodes_rec = torch.sigmoid(nodes_rec) * node_masks #bs,max_node_num,n_node_feat
        node_loss = F.binary_cross_entropy(nodes_rec, target, reduction='none')
    # nodes_rec = nodes_rec.log_softmax(-1) * node_masks #bs,max_node_num,n_node_feat
    # node_loss = F.nll_loss(nodes_rec.permute(0, 2, 1).contiguous(), target, reduction='none') #bs,max_node_num
    node_loss = node_loss.mean()
    if is_mol_data:
        nodes_rec = discretize(nodes_rec, node_masks) #bs,max_node_num,n_node_feat 
    elif not is_mol_data and annotated_nodes: #非mol但有node feat
        nodes_rec = nodes_rec.round().float()
    none_type = 1-node_masks.float()
    nodes_rec = torch.cat((nodes_rec[:, :, :n_node_feat], none_type), dim=-1) #bs,max_node_num,n_node_feat+1
    return node_loss, nodes_rec, target



def get_edge_loss_and_recon(batch, edges_rec, edge_masks):
    '''
    Return the loss for the edge features and the reconstructed and discretized instance
    '''
    edges_rec = (edges_rec.transpose(1, 2) + edges_rec) * .5
    edges_rec = edges_rec.log_softmax(-1)
    edge_target = get_edge_target(batch)


    edge_loss = F.nll_loss(edges_rec.permute(0, 3, 1, 2).contiguous(), edge_target, reduction='none')
    edge_loss = edge_loss * edge_masks.squeeze()
    edge_loss = edge_loss.mean()
    edges_rec = discretize(edges_rec, edge_masks)
    edges_rec[:, :, :, 0] = edges_rec[:, :, :, 0] + (1-edge_masks.squeeze())
    return edge_loss, edges_rec


def get_unannotated_loss_and_rec(batch, edges_rec, edge_masks, max_node_num):
    '''
    Return the loss for the edges and the reconstructed and discretized instance
    '''
    if edges_rec.ndim==3:
        print("重构边是3维度!!!error!!!")
    edges_rec = (edges_rec.transpose(1, 2) + edges_rec) * .5 #ensure edges_rec is symmetric
    adjs = to_dense_adj(batch.edge_index, batch=batch.batch, max_num_nodes=max_node_num)
    edges_rec = edges_rec.sigmoid()
    edges_rec = edges_rec * edge_masks
    edge_loss = F.binary_cross_entropy(edges_rec.flatten(1), adjs.flatten(1), reduction='mean')
    edges_rec = edges_rec.round() #将边特征进行四舍五入 #TODO: use threshold or round? edges:(bs,n_max,n_max,1)
    return edge_loss, edges_rec