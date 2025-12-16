import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing



class Mlp(nn.Module):
    def __init__(self,
                 in_,
                 out_,
                 hidden_,
                 activation=nn.ReLU()
                 ):
        super().__init__()
        n_layers = len(hidden_) - 1

        layers = [nn.Linear(in_, hidden_[0]), activation]
        for i in range(n_layers):
            layers.append(nn.Linear(hidden_[i], hidden_[i + 1]))
            layers.append(activation)
        layers.append(nn.Linear(hidden_[-1], out_))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Gnn(nn.Module):
    def __init__(self, nnf_in, nef_in, nnf_out, nef_out, n_layers, hidden_size, normalization):
        super().__init__()
        hidden = [hidden_size] * n_layers
        self.edge_linear = Mlp(2 * nnf_in + nef_in, nef_out, hidden)
        self.node_linear = Mlp(2 * nnf_in + nef_in, nnf_out, hidden)

        if normalization == 'batch_norm':
            self.edge_norm = nn.BatchNorm2d(nef_out)
            self.node_norm = nn.BatchNorm1d(nnf_out)

        elif normalization == 'layer_norm':
            self.edge_norm = nn.LayerNorm(nef_out)
            self.node_norm = nn.LayerNorm(nnf_out)

        self.normalization = normalization

    def forward(self, node_feat, edge_feat=None, skip_connection=False):
        nodes2nodes = nodes2edges(node_feat)
        if edge_feat is not None:
            edge_feat = torch.cat((nodes2nodes, edge_feat), dim=3)
        else:
            edge_feat = nodes2nodes
        #node_feat = self.node_linear(edge_feat).permute(0, 3, 1, 2)  # * (1 - torch.eye(n))
        edge_out = self.edge_linear(edge_feat)
        node_out = self.node_linear(edge_feat)
        node_out = node_out.sum(-2)
        if skip_connection:
            node_out = node_out + node_feat

        if self.normalization == 'batch_norm':
            edge_out = self.edge_norm(edge_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            node_out = self.node_norm(node_out.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.normalization == 'layer_norm':
            edge_out = self.edge_norm(edge_out)
            node_out = self.node_norm(node_out)
        return node_out, edge_out


class GnnSparse(MessagePassing):
    def __init__(self, nnf_in, nef_in, nnf_out, nef_out, n_layers, hidden_size, normalization=None):
        super().__init__(aggr='add')
        hidden = [hidden_size] * n_layers
        self.mlp_e = Mlp(2 * nnf_in + nef_in, nef_out, hidden)
        self.mlp_n = Mlp(2 * nnf_in + nef_in, nnf_out, hidden)

        if normalization == 'batch_norm':
            self.edge_norm = nn.BatchNorm1d(nef_out)
            self.node_norm = nn.BatchNorm1d(nnf_out)

        elif normalization == 'layer_norm':
            self.edge_norm = nn.LayerNorm(nef_out)
            self.node_norm = nn.LayerNorm(nnf_out)

        self.normalization = normalization
        self.edge_feat = 0

    def forward(self, x, edge_index, edge_attr=None, skip_connection=False):
        node_feat = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        if skip_connection:
            node_feat = node_feat + x

        if self.normalization == 'batch_norm':
            self.edge_feat = self.edge_norm(self.edge_feat)
            node_feat = self.node_norm(node_feat)
        elif self.normalization == 'layer_norm':
            self.edge_feat = self.edge_norm(self.edge_feat)
            node_feat = self.node_norm(node_feat)
        return node_feat, self.edge_feat

    def message(self, x_i, x_j, edge_attr=None):
        if edge_attr is not None:
            x_j = torch.cat((x_i, x_j, edge_attr), dim=1)
        else:
            x_j = torch.cat((x_i, x_j), dim=1)
        self.edge_feat = self.mlp_e(x_j)
        x_j = self.mlp_n(x_j)
        return x_j

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, first_layer=False):
        super().__init__()
        if not first_layer:
            self.mlp_q = Mlp(d_model, d_model, 4 * [2 * d_model])
            self.mlp_k = Mlp(d_model, d_model, 4 * [2 * d_model])
            self.mlp_v = Mlp(d_model, d_model, 4 * [2 * d_model])
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    num_heads,
                                                    dropout=0.0,
                                                    add_zero_attn=False,
                                                    bias=False,
                                                    batch_first=True)
        self.layerNorm1 = torch.nn.LayerNorm(d_model)
        self.mlp = Mlp(d_model, d_model, 2 * [2 * d_model])
        self.layerNorm2 = torch.nn.LayerNorm(d_model)
        self.first_layer = first_layer

    def forward(self, q, k, v, embeddings, mask=None, normalize_output=True):
        '''
        if not self.first_layer:
            #q = self.mlp_q(embeddings)
            #k = self.mlp_k(embeddings)
            #v = self.mlp_v(embeddings)
        else:
            q, k, v = embeddings
        '''
        attn_output, aw = self.multihead_attn(q, k, v, attn_mask=mask)
        if not self.first_layer:
            output_norm = self.layerNorm1(attn_output + embeddings)
        else:
            output_norm = self.layerNorm1(attn_output)
        # output_norm = attn_output + embeddings
        output = self.mlp(output_norm)
        if normalize_output:
            return self.layerNorm2(output + output_norm)
            # return output + output_norm
        else:
            return output

def symetric_matrix_product(tensor):
    message = "The 2 last dimentsions of the tensor should have the same shape"
    assert tensor.shape[-2] == tensor.shape[-1], message
    tensorT = tensor.transpose(-2, -1)
    return tensor * tensorT


def symetric_matrix_mean(tensor):
    message = "The 2 last dimentsions of the tensor should have the same shape"
    assert tensor.shape[-2] == tensor.shape[-1], message
    tensorT = tensor.transpose(-2, -1)
    return (tensor + tensorT) / 2
class CodeBook(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(torch.randn(1, 20, 1, 8))
    def forward(self):
        return self.params
def get_inverse_degree_vector(adjacency, leak=0):
    degree = adjacency.sum([1, 2]).type(torch.long)
    idx_vector_is_0 = (degree == 0) * 1.
    degree_inv = degree + idx_vector_is_0
    degree_inv = 1 / degree_inv
    degree_inv = degree_inv - idx_vector_is_0 * (1 - leak)
    return degree_inv

def nodes2edges(nodes):
    nodes_ = nodes.unsqueeze(3).permute(0, 3, 1, 2)
    nodes_ = nodes_.repeat(1, nodes.shape[1], 1, 1)
    nodesT = nodes_.transpose(1, 2)
    return torch.cat([nodes_, nodesT], dim=3)
