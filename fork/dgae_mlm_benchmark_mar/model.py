import torch
import torch.nn as nn
import torch.nn.functional as F
import nn as nn_
import numpy as np
from nn import Mlp
from scipy.cluster.vq import kmeans2


class Decoder(nn.Module):
    def __init__(self, config, data_info):
        super().__init__()
        n_layers = config.model.decoder.n_layers
        nz = config.model.quantizer.nz
        nhf = config.model.decoder.nhf
        normalization = config.model.decoder.normalization
        self.skip_connection = config.model.decoder.skip_connection
        nnf = data_info.n_node_feat
        nef = data_info.n_edge_feat
        mlp_n_layers = config.model.decoder.mlp_n_layers
        mlp_hidden_size = config.model.decoder.mlp_hidden_size

        layers = [nn_.Gnn(nz, 0, nhf, nhf, mlp_n_layers, mlp_hidden_size, normalization=normalization)]
        for layer in range(1, n_layers - 1):
            layers.append(nn_.Gnn(nhf, nhf, nhf, nhf, mlp_n_layers, mlp_hidden_size, normalization=normalization))
        layers.append(nn_.Gnn(nhf, nhf, nnf, nef, mlp_n_layers, mlp_hidden_size, normalization=None))

        self.layers = nn.Sequential(*layers)

    def forward(self, node_feat, mask=None): #node_feat:bs,n_max,nc*nz, mask:bs,n_max
        node_feat, edge_feat = self.layers[0](node_feat)
        for i, layer in enumerate(self.layers[1:-1]):
            node_feat_new, edge_feat_new = layer(node_feat, edge_feat,
                                                 skip_connection=self.skip_connection)
            node_feat = node_feat + node_feat_new
            edge_feat = edge_feat + edge_feat_new
        node_feat, edge_feat = self.layers[-1](node_feat, edge_feat, skip_connection=False)
        if mask is not None:
            node_feat = node_feat * mask.unsqueeze(-1)
            edge_feat = edge_feat * mask.reshape(node_feat.shape[0], -1, 1, 1)
            edge_feat = edge_feat * mask.reshape(node_feat.shape[0], 1, -1, 1)
        return node_feat, edge_feat


class Encoder(nn.Module):
    def __init__(self, config, data_info):
        super().__init__()
        nnf = data_info.n_node_feat + data_info.additional_node_feat
        nef = data_info.n_edge_feat + data_info.additional_edge_feat
        nhf = config.model.encoder.nhf
        nz = config.model.quantizer.nz
        normalization = config.model.encoder.normalization
        self.skip_connection = config.model.encoder.skip_connection

        n_layers = config.model.encoder.n_layers

        mlp_n_layers = config.model.encoder.mlp_n_layers
        mlp_hidden_size = config.model.encoder.mlp_hidden_size
        layers = [nn_.GnnSparse(nnf, nef, nhf, nhf, mlp_n_layers, mlp_hidden_size, normalization=normalization)]
        for layer in range(1, n_layers - 1):
            layers.append(nn_.GnnSparse(nhf, nhf, nhf, nhf, mlp_n_layers, mlp_hidden_size, normalization=normalization))
        layers.append(nn_.GnnSparse(nhf, nhf, nz, 1, mlp_n_layers, mlp_hidden_size, normalization=normalization))
        #self.add_input_noise = config.model.encoder.add_input_noise
        self.layers = nn.Sequential(*layers)

    def forward(self, batch):
        if not hasattr(batch, 'edge_index_ext'):
            batch.edge_index_ext = batch.edge_index
            batch.edge_attr_ext = None
            
        node_feat, edge_feat = self.layers[0](batch.x,
                                            batch.edge_index_ext,
                                            edge_attr=batch.edge_attr_ext)

        for i, layer in enumerate(self.layers[1:-1]):
            node_feat_new, edge_feat_new = layer(node_feat, batch.edge_index_ext,
                                                 edge_attr=edge_feat,
                                                 skip_connection=self.skip_connection)
            node_feat = node_feat + node_feat_new
            edge_feat = edge_feat + edge_feat_new
        node_feat, edge_feat = self.layers[-1](node_feat,
                                               batch.edge_index_ext,
                                               edge_attr=edge_feat)
        return node_feat, edge_feat

class Quantizer(nn.Module):
    # Adapted from https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/
    # blob/master/02_Vector_Quantized_Variational_AutoEncoder.ipynb
    def __init__(self, config):
        super().__init__()
        # error_msg = 'The size of your latent vector should be dividable by the dimension of the vector in the codebook'
        # assert embedding_dim % latent_vectors_by_node == 0, error_msg
        train_prior = config.train_prior
        config = config.model.quantizer
        self.init_steps = config.init_steps
        self.n_embeddings = config.codebook_size
        self.nc = config.nc
        self.embedding_dim = config.nz // self.nc
        self.commitment_cost = config.commitment_cost
        self.decay = config.decay
        self.epsilon = config.epsilon


        init_samples = torch.Tensor(0, self.nc, self.embedding_dim)
        init_bound = 1 / self.n_embeddings
        embedding = torch.Tensor(self.nc, self.n_embeddings, self.embedding_dim)
        embedding.uniform_(-init_bound, init_bound)

        self.register_buffer("init_samples", init_samples)
        if train_prior:
            self.register_buffer("init_samples", torch.zeros(0, self.nc, self.embedding_dim))
            #self.register_buffer("init_samples", torch.zeros(100_000, self.nc, self.embedding_dim))
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(self.nc, self.n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

        #if self.nc == 2:
        #    self.register_buffer("oneVectIdx", torch.zeros(self.n_embeddings ** 2))

        self.encod_sum = 0
        self.idxToRank = 0
        if self.init_steps > 0:
            self.collect = True

    def encode(self, x):
        nc, m, d = self.embedding.size()
        x_flat = x.reshape(-1, self.nc, d)
        x_flat_detached = x_flat.detach()
        diff = x_flat_detached.unsqueeze(2) - self.embedding.unsqueeze(0)
        distances = torch.sum(diff ** 2, -1)
        indices = torch.argmin(distances.float(), dim=-1)
        idx0 = torch.arange(self.nc).unsqueeze(0).repeat(indices.shape[0], 1)
        quantized = self.embedding[idx0, indices]
        return quantized, indices

    def retrieve_random_codebook(self, random_indices):
        quantized = F.embedding(random_indices, self.embedding)
        quantized = quantized.transpose(1, 3)
        return quantized

    def indices_to_zq(self, indices, padded=False):
        nc = self.embedding.shape[0]
        nv = self.embedding.shape[-1]
        device = indices.device
        indices_flatten = indices.flatten(0, 1)
        if padded:
            embedding = torch.cat((self.embedding, torch.zeros(nc, 1, nv, device=device)), dim=1)
        else:
            embedding = self.embedding
        idx0 = torch.arange(self.nc).unsqueeze(0).repeat(indices_flatten.shape[0], 1)
        quantized = embedding[idx0, indices_flatten]
        quantized = quantized.reshape(*indices.shape, -1)
        return quantized

    def sort_embedding(self):
        idx0 = torch.arange(self.nc).unsqueeze(1).repeat(1, self.embedding.shape[1])
        self.embedding = self.embedding[idx0, self.ema_count.sort(1, descending=True)[1]]
        self.ema_count = self.ema_count[idx0, self.ema_count.sort(1, descending=True)[1]]

    # def forward(self, x, mask=None):
    #     nc, m, d = self.embedding.size()
    #     x_flat = x.reshape(-1, self.nc, d)
    #     x_flat_detached = x_flat.detach()

    #     diff = x_flat_detached.unsqueeze(2) - self.embedding.unsqueeze(0)
    #     distances = torch.sum(diff ** 2, -1) 
    #     indices = torch.argmin(distances.float(), dim=-1)
    #     encodings = F.one_hot(indices, m).float()
    #     idx0 = torch.arange(self.nc).unsqueeze(0).repeat(indices.shape[0], 1)
    #     quantized = self.embedding[idx0, indices]

    #     if self.training:
    #         self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
    #         n = torch.sum(self.ema_count, 1, keepdim=True)
    #         self.ema_count = (self.ema_count + self.epsilon) / (n + m * self.epsilon) * n
    #         dw = torch.matmul(encodings.permute(1, 2, 0), x_flat_detached.permute(1, 0, 2))
    #         self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
    #         self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

    #     if mask is not None:
    #         quantized = quantized * mask.unsqueeze(-1)

    #     codebook_loss = F.mse_loss(x_flat.detach(), quantized)
    #     e_latent_loss = F.mse_loss(x_flat, quantized.detach())
    #     commitment_loss = self.commitment_cost * e_latent_loss

    #     quantized = x_flat + (quantized - x_flat).detach()
    #     quantized = quantized.reshape_as(x)

    #     avg_probs = torch.mean(encodings, dim=0)
    #     perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    #     return quantized, commitment_loss, codebook_loss, perplexity, indices
    
    
    # diffloss version, no need for codebook and indices
    def forward(self, x, mask=None): 
        # x是gnn编码后的latents，作为diffloss的target
        pass
        
    def collect_samples(self, zq):
        self.init_samples = torch.cat((self.init_samples, zq), dim=0)
        if self.init_samples.shape[0] >= 100_000:
            self.init_samples = self.init_samples[-100_000:]
            self.collect = False
            self.kmeans_init()
            self.init_samples = torch.Tensor(0, self.nc, self.embedding_dim)
            return False
        else:
            return True

    def kmeans_init(self):
        device = self.init_samples.device
        init_samples = self.init_samples.cpu().numpy()
        ks = []
        for c in range(self.nc):
            k = kmeans2(init_samples[:, c], self.n_embeddings, minit='++')
            k = torch.from_numpy(k[0])
            ks.append(k)
        self.embedding = torch.stack(ks, dim=0).to(device)


class PositionalEncoding(nn.Module):
    '''
    Originally from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    '''
    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 40):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            # x: Tensor, shape [seq_len, batch_size, embedding_dim]
            x: shape: (batch_size, seq_len, embedding_dim)
        """
        x = x.permute(1, 0, 2) #(seq_len,bs,d_model) 或写在forward里
        # x = x #(seq_len,bs,d_model)
        x = x + self.pe[:x.size(0)] #(seq_len,bs,d_model)
        return self.dropout(x).permute(1, 0, 2) #(bs,seq_len,d_model)

class Transformer2d(nn.Module):
    def __init__(self, config):
        super().__init__()
        nz = config.model.quantizer.nz
        nc = config.model.quantizer.nc
        n_max = config.data.max_node_num
        d_model = config.transformer.d_model
        num_heads = config.transformer.num_heads
        n_blocks = config.transformer.n_blocks
        out_dim = config.model.quantizer.codebook_size + 3 #+3 for pad(256), mask, sos tokens
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.cb_size = config.model.quantizer.codebook_size
        self.mask_token_id = self.cb_size + 2
        self.sos_token = self.cb_size + 1

        # 新增：token embedding层
        self.tok_emb = nn.Embedding(
            config.model.quantizer.codebook_size + 3,  # +3 for pad(256), mask, sos tokens,259
            config.transformer.d_model,
            # padding_idx=config.model.quantizer.codebook_size  # pad token的索引
        )
        # 位置编码
        self.pe = PositionalEncoding(config.transformer.d_model, dropout=0, max_len=n_max+1) #+sos_token
        # Layer Norm 和 Dropout
        self.ln = nn.LayerNorm(config.transformer.d_model)
        self.drop = nn.Dropout(p=0.1)
        
        self.mlp_inV = nn_.Mlp(d_model, d_model, 4 * [2 * d_model])
        self.mlp_inK = nn_.Mlp(d_model, d_model, 4 * [2 * d_model])
        self.mlp_inQ = nn.ModuleList([nn_.Mlp(d_model, d_model, 4 * [2 * d_model]) for c in range(nc)])
        self.mlp_inZ = nn.ModuleList([nn_.Mlp(d_model, d_model, 4 * [2 * d_model]) for c in range(nc)])

        self.blockIn = nn.ModuleList([nn_.TransformerBlock(d_model, num_heads) for c in range(nc)])
        layers = []
        for i in range(n_blocks - 1):
            layers.append(nn.ModuleList([nn_.TransformerBlock(d_model, num_heads) for c in range(nc)]))
        self.layers = nn.Sequential(*layers)

        self.mlp_out = nn.ModuleList([nn_.Mlp(d_model, out_dim, 2 * [2 * d_model]) for c in range(nc)])

        self.nc = nc
        self.nz = nz
        self.n_max = n_max
        self.out_dim =out_dim

        self.mlp_in = nn.ModuleList([])
        for c in range(nc):
            self.mlp_in.append(nn.Linear(nz + nz * c // nc, d_model))

        self.mlp_q = nn.ModuleList([nn.ModuleList([Mlp(d_model, d_model, 4 * [2 * d_model]) for c in range(nc)])
                                    for i in range(n_blocks - 1)])
        self.mlp_k = nn.ModuleList([Mlp(d_model, d_model, 4 * [2 * d_model]) for i in range(n_blocks - 1)])
        self.mlp_v = nn.ModuleList([Mlp(d_model, d_model, 4 * [2 * d_model]) for i in range(n_blocks - 1)])

    # def forward(self, embeddings,mask=None):
    #     import ipdb; ipdb.set_trace()
    #     bs, n_max, nc, nz = embeddings.shape
    #     device = embeddings.device
    #     embeddings_shifted = torch.cat((torch.zeros(bs, 1, nc, nz).to(device), embeddings[:, :-1]), dim=1)
    #     embeddings = torch.cat((embeddings_shifted, embeddings), dim=2)

    #     z, q, emb = [None] * (nc), [None] * (nc), [None] * (nc)
    #     for c in range(nc):
    #         z[c] = self.mlp_in[c](embeddings[:, :, :nc + c].flatten(2))
    #         q[c] = self.mlp_inQ[c](z[c])

    #     v = self.mlp_inV(z[0])
    #     k = self.mlp_inK(z[0])
    #     v = self.pe(v.permute(1, 0, 2)).permute(1, 0, 2)
    #     for c in range(nc):
    #         emb[c] = self.blockIn[c](q[c], k, v, q[c], mask=mask)

    #     for i, layer in enumerate(self.layers):
    #         k = self.mlp_k[i](emb[0])
    #         v = self.mlp_v[i](emb[0])
    #         for c in range(nc):
    #             q[c] = self.mlp_q[i][c](emb[c])
    #             emb[c] = layer[c](q[c], k, v, emb[c], mask=mask, normalize_output=True)
    #     for c in range(nc):
    #         emb[c] = self.mlp_out[c](emb[c])
    #     out = torch.stack(emb, dim=-2)
    #     return out
    
    # input masked indices
    def forward(self, use_mlm, indices=None, padding_mask=None, embeddings=None, mask=None):
        """
        Args:
            indices: 已经mask过的token indices (bs, n_max, nc)
            padding_mask: 指示非pad位置的布尔掩码 (bs, seq_len, nc)
        """
        if use_mlm:
            bs, n_max, nc = indices.shape 
            device = indices.device 
            # padding_mask = (indices != self.cb_size) #(bs,n_max+1,nc)
            
            # 1. 获取token embeddings 
            z, q, emb = [None] * (nc), [None] * (nc), [None] * (nc) 
            
            # 2. 处理每个通道
            for c in range(nc):
                # # 直接从indices获取embeddings
                # z[c] = self.mlp_in[c](self.embedding[c][indices[..., c]])
                
                # 使用可训练的embedding层
                z[c] = self.tok_emb(indices[..., c])  # (bs, n_max, d_model)
                
                # 2. 如果有padding_mask，将pad位置的embedding设为0
                if padding_mask is not None:
                    z[c] = z[c] * padding_mask[..., c].unsqueeze(-1) #todo: embedding层已经初始化为0，所以这里不需要再乘padding_mask?
                # 添加位置编码
                # print("use mlm zc shape:",z[c].shape)
                z[c] = z[c] + self.pe(z[c]) #bs,seqlen,d
                # z[c] = z[c] + self.pe(z[c].permute(1,0,2)) #bs,seqlen,d
                # Layer Norm 和 Dropout
                z[c] = self.drop(self.ln(z[c]))
                
                # 输入层
                q[c] = self.mlp_inQ[c](z[c])
            
            # 3. 生成key和value
            v = self.mlp_inV(z[0])
            k = self.mlp_inK(z[0]) #(bs,n_max+1,d_model)
            v = self.pe(v) #(bs,n_max+1,d_model)
            
            # 3. 第一层transformer block
            for c in range(nc):
                emb[c] = self.blockIn[c](q[c], k, v, q[c])

            # 4. 后续transformer layers
            for i, layer in enumerate(self.layers):
                k = self.mlp_k[i](emb[0])
                v = self.mlp_v[i](emb[0])
                for c in range(nc):
                    q[c] = self.mlp_q[i][c](emb[c])
                    emb[c] = layer[c](q[c], k, v, emb[c], normalize_output=True)

            # 5. 输出层
            for c in range(nc):
                emb[c] = self.mlp_out[c](emb[c])
            
            out = torch.stack(emb, dim=-2) #(32,21,1,257)
            
            # todo:是否需要去掉sos token
            # out = out[:,1:,:,:] #(32,20,1,257)
            return out

        else:
            bs, n_max, nc, nz = embeddings.shape
            device = embeddings.device
            embeddings_shifted = torch.cat((torch.zeros(bs, 1, nc, nz).to(device), embeddings[:, :-1]), dim=1)
            embeddings = torch.cat((embeddings_shifted, embeddings), dim=2)

            z, q, emb = [None] * (nc), [None] * (nc), [None] * (nc)
            for c in range(nc):
                z[c] = self.mlp_in[c](embeddings[:, :, :nc + c].flatten(2))
                q[c] = self.mlp_inQ[c](z[c])

            v = self.mlp_inV(z[0])
            k = self.mlp_inK(z[0])
            # v = self.pe(v.permute(1, 0, 2)).permute(1, 0, 2)
            # v = self.pe(v.permute(1, 0, 2)) #bs,seqlen,d
            v = self.pe(v) #bs,seqlen,d
            for c in range(nc):
                emb[c] = self.blockIn[c](q[c], k, v, q[c], mask=mask)

            for i, layer in enumerate(self.layers):
                k = self.mlp_k[i](emb[0])
                v = self.mlp_v[i](emb[0])
                for c in range(nc):
                    q[c] = self.mlp_q[i][c](emb[c])
                    emb[c] = layer[c](q[c], k, v, emb[c], mask=mask, normalize_output=True)
            for c in range(nc):
                emb[c] = self.mlp_out[c](emb[c])
            out = torch.stack(emb, dim=-2)
            return out
            
    def sample(self, z_c, c, z_completed): #sample_prior时调用
        # import ipdb; ipdb.set_trace()
        nc = z_completed.shape[2]
        z_c = torch.cat((z_completed[:, 1:, :c], z_c[:, :, nc:nc+c]), dim=1)
        z_c = torch.cat((z_completed, z_c), dim=2)
        device = z_c.device

        z_0 = self.mlp_in[0](z_completed[:, -1].unsqueeze(1).flatten(2))
        z_c = self.mlp_in[c](z_c[:, -1].unsqueeze(1).flatten(2))

        if c == 0:
            k = self.mlp_inK(z_0)
            v = self.mlp_inV(z_0)

            self.Ks[0] = torch.cat((self.Ks[0].to(device), k), dim=1)
            self.Vs[0] = torch.cat((self.Vs[0].to(device), v), dim=1)
        q_c = self.mlp_inQ[c](z_c)
        v0 = self.pe(self.Vs[0].permute(1, 0, 2)).permute(1, 0, 2)

        emb = self.blockIn[c](q_c, self.Ks[0], v0, q_c)
        for i, layer in enumerate(self.layers):
            if c == 0:
                k = self.mlp_k[i](emb)
                v = self.mlp_v[i](emb)
                self.Ks[i + 1] = torch.cat((self.Ks[i + 1].to(device), k), dim=1)
                self.Vs[i + 1] = torch.cat((self.Vs[i + 1].to(device), v), dim=1)

            q = self.mlp_q[i][c](emb)
            emb = layer[c](q, self.Ks[i + 1], self.Vs[i + 1], emb, normalize_output=True)
        p_z = self.mlp_out[c](emb)
        return p_z

    def init_sampler(self, n_samples):
        self.Ks = [torch.zeros(n_samples, 0, self.d_model)] * self.n_blocks
        self.Vs = [torch.zeros(n_samples, 0, self.d_model)] * self.n_blocks
    
    # for sample_prior
    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError