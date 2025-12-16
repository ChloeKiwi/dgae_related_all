import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import dense_to_sparse, to_dense_adj, get_laplacian
from utils.func import get_features


class Qm9DropH(BaseTransform):
    r""" Remove Hydrogen from the molecular graph.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        n_heavy_atom = (data.x[:, 0] == 0).sum()
        data.x = data.x[(data.x[:, 0] == 0)]
        data.x = data.x[:, 1:5]
        edge_index_to_keep = data.edge_index < n_heavy_atom
        edges_to_keep = (edge_index_to_keep[0]*edge_index_to_keep[1]) > 0
        data.edge_index = data.edge_index[:, edges_to_keep]
        data.edge_attr = data.edge_attr[edges_to_keep]
        if data.edge_index.size()[1] == 0:
            data.edge_attr = torch.zeros(0, 4)
        return data



class AddSynFeat(BaseTransform):
    r""" Add the 2-hop and 3-hop edges as edge attr. and the 1st, 2nd and 3rd order of degrees
    as node attr.
    """

    def __init__(self, max_num_nodes):
        super().__init__()
        self.max_num_nodes = max_num_nodes
        self.max = 0
        self.min = 200
    def __call__(self, data):
        if data.edge_index.size()[1] > 0:
            adj = to_dense_adj(data.edge_index)
            node_feat, edge_feat, _ = get_features(adj.reshape(1, 1, adj.shape[-1], adj.shape[-1]), moment=3)

            data.x = torch.cat((data.x, node_feat.squeeze()), dim=-1)

            dense_edge_attr = to_dense_adj(data.edge_index, edge_attr=data.edge_attr)
            if len(dense_edge_attr.shape) == 4:
                edge_feat = torch.cat((edge_feat, dense_edge_attr), dim=-1)
            edge_index_ext, _ = dense_to_sparse(edge_feat.sum(-1) >= 1)

            edge_feat = edge_feat.squeeze()
            edge_feat = edge_feat[edge_index_ext[0], edge_index_ext[1]].float()
            data.edge_index_ext = edge_index_ext
            data.edge_attr_ext = edge_feat


            dense_edge_attr = to_dense_adj(data.edge_index, edge_attr=data.edge_attr, max_num_nodes=self.max_num_nodes)
            # if len(dense_edge_attr.shape) == 4:
            #     no_edge = 1 - dense_edge_attr.sum(-1, keepdim=True)
            #     dense_edge_attr = torch.cat((no_edge, dense_edge_attr), dim=-1)
            #     data.edge_target = dense_edge_attr.argmax(-1)
            # else:
            #     data.edge_target = dense_edge_attr

        else:
            data.x = torch.cat((data.x, torch.zeros(data.x.shape[0], 3)), dim=-1)
            data.edge_index_ext = data.edge_index
            data.edge_attr_ext = torch.zeros(0, 6)
            # data.edge_target = (torch.zeros(1, self.max_num_nodes, self.max_num_nodes)).long()
        return data

class AddNoFeat(BaseTransform):
    r""" Add edge_index_ext and edge_attr_ext as object attribute and fill it with edge_index
    and edge_attr.
    """

    def __init__(self, max_num_nodes):
        super().__init__()

    def __call__(self, data):
        data.edge_index_ext = data.edge_index
        data.edge_attr_ext = data.edge_attr
        # data.edge_target = to_dense_adj(data.edge_index, max_num_nodes=38)
        # data.edge_target = data.edge_target.long()
        return data

class AddRandomFeat(BaseTransform):
    r""" Add edge_index_ext and edge_attr_ext as object attribute and fill it with edge_index
    and edge_attr.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.x = torch.cat((data.x, torch.rand(data.x.shape[0], 4)), dim=-1)
        return data

class AddSpectralFeat(BaseTransform):
    r""" Add edge_index_ext and edge_attr_ext as object attribute and fill it with edge_index
    and edge_attr.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        lap,_ = get_laplacian(data.edge_index)
        lap = to_dense_adj(lap).squeeze()
        eigvals, eigvectors = torch.linalg.eigh(lap)
        K = 5
        eigfeat = eigvectors[..., :K]
        if eigfeat.shape[-1] < K:
            missing = K - eigfeat.shape[-1]
            if data.x.shape[0] == 1:
                eigfeat = torch.zeros(1, 5)
            else:
                eigfeat = torch.cat((eigfeat, torch.zeros(data.x.shape[0], missing)), dim=-1)

        if data.x is not None:
            data.x = torch.cat((data.x, eigfeat), dim=-1)
        else:
            data.x = eigfeat
        return data

class AddCyclesFeat(BaseTransform):
    r""" Add edge_index_ext and edge_attr_ext as object attribute and fill it with edge_index
    and edge_attr.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        A = to_dense_adj(data.edge_index).squeeze()
        A2 = A @ A
        A3 = A2 @ A
        A4 = A3 @ A
        A5 = A4 @ A
        d = A.sum(-1)

        c3 = A3.diagonal().unsqueeze(-1)/2
        diag_a4 = A4.diagonal()
        c4 = diag_a4 - d * (d - 1) - (A @ d.unsqueeze(-1)).sum(dim=-1)
        c4 = c4.unsqueeze(-1)/2

        diag_a5 = A5.diagonal()
        triangles = A3.diagonal()

        c5 = diag_a5 - 2 * triangles * d - (A @ triangles.unsqueeze(-1)).sum(dim=-1) + triangles
        c5 = c5.unsqueeze(-1) / 2
        if data.x.shape[0] == 1:
            data.x = torch.cat((data.x, torch.zeros(1, 3)), dim=-1)
        else:
            data.x = torch.cat((data.x, c3, c4, c5), dim=-1)
        return data

class AddSynFeatToUnannotated(BaseTransform):
    r""" Add the 2-hop and 3-hop edges as edge attr. and the 1st, 2nd and 3rd order of degrees
    as node attr.
    """

    def __init__(self, max_num_nodes):
        super().__init__()
        self.max_num_nodes = max_num_nodes
    def __call__(self, data):
        adj = to_dense_adj(data.edge_index)
        node_feat, edge_feat, _ = get_features(adj.reshape(1, 1, adj.shape[-1], adj.shape[-1]), moment=3)
        data.x = node_feat.squeeze()
        edge_index_ext, _ = dense_to_sparse(edge_feat.sum(-1) >= 1)
        edge_feat = edge_feat.squeeze()
        edge_feat = edge_feat[edge_index_ext[0], edge_index_ext[1]].float()
        data.edge_index_ext = edge_index_ext
        data.edge_attr_ext = edge_feat
        data.edge_attr = torch.ones(data.edge_index.shape[1], 1)

        # data.edge_target = to_dense_adj(data.edge_index, max_num_nodes=self.max_num_nodes)
        return data
