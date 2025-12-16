import torch
import torch.nn as nn
import torch.nn.functional as F
import nn as nn_
import numpy as np
from nn import Mlp
from scipy.cluster.vq import kmeans2
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Block
from diffloss import DiffLoss
import scipy.stats as stats

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
        return node_feat, edge_feat #node_feat:bs,n_max,nc*nz, edge_feat:bs,n_max,nc*nz

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

    def indices_to_zq(self, indices, padded=False): #indices:(bs, max_node_num, nc)
        # import ipdb; ipdb.set_trace()
        nc = self.embedding.shape[0]
        nv = self.embedding.shape[-1]
        device = indices.device
        indices_flatten = indices.flatten(0, 1) #(bs*max_node_num, nc) 2000,1
        if padded:
            embedding = torch.cat((self.embedding, torch.zeros(nc, 1, nv, device=device)), dim=1) #(nc, cb_size+1, nv)
        else:
            embedding = self.embedwding
        idx0 = torch.arange(self.nc).unsqueeze(0).repeat(indices_flatten.shape[0], 1) #(bs*max_node_num, nc) 为每个node的nc个位置制作索引
        quantized = embedding[idx0, indices_flatten] #(bs*max_node_num, nc, nv)
        quantized = quantized.reshape(*indices.shape, -1) #(bs, max_node_num, nc, nv)
        return quantized

    def sort_embedding(self):
        idx0 = torch.arange(self.nc).unsqueeze(1).repeat(1, self.embedding.shape[1])
        self.embedding = self.embedding[idx0, self.ema_count.sort(1, descending=True)[1]]
        self.ema_count = self.ema_count[idx0, self.ema_count.sort(1, descending=True)[1]]

    def forward(self, x, mask=None):
        nc, m, d = self.embedding.size()
        x_flat = x.reshape(-1, self.nc, d)
        x_flat_detached = x_flat.detach()

        diff = x_flat_detached.unsqueeze(2) - self.embedding.unsqueeze(0)

        distances = torch.sum(diff ** 2, -1)

        indices = torch.argmin(distances.float(), dim=-1)

        encodings = F.one_hot(indices, m).float()
        idx0 = torch.arange(self.nc).unsqueeze(0).repeat(indices.shape[0], 1)
        quantized = self.embedding[idx0, indices]

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
            n = torch.sum(self.ema_count, 1, keepdim=True)
            self.ema_count = (self.ema_count + self.epsilon) / (n + m * self.epsilon) * n
            dw = torch.matmul(encodings.permute(1, 2, 0), x_flat_detached.permute(1, 0, 2))
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        if mask is not None:
            quantized = quantized * mask.unsqueeze(-1)

        codebook_loss = F.mse_loss(x_flat.detach(), quantized)
        e_latent_loss = F.mse_loss(x_flat, quantized.detach())
        commitment_loss = self.commitment_cost * e_latent_loss

        quantized = x_flat + (quantized - x_flat).detach()
        quantized = quantized.reshape_as(x)

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return quantized, commitment_loss, codebook_loss, perplexity, indices

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
        x = x.permute(1, 0, 2) #(seq_len,bs,d_model)
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
        # out_dim = config.model.quantizer.codebook_size + 1 #+1 for pad(256)
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.cb_size = config.model.quantizer.codebook_size
        self.mask_token_id = self.cb_size + 2
        self.sos_token = self.cb_size + 1

        # 新增：token embedding层
        self.tok_emb = nn.Embedding(
            config.model.quantizer.codebook_size + 3,  # +3 for pad(256), mask, sos tokens,259
            # config.model.quantizer.codebook_size + 1,  # +1 for pad(256)
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

        self.mlp_out = nn.ModuleList([nn_.Mlp(d_model, out_dim, 2 * [2 * d_model]) for c in range(nc)]) #todo 适用于celoss
        self.mlp_out_diff = nn.ModuleList([nn_.Mlp(d_model, d_model, 2 * [2 * d_model]) for c in range(nc)]) #todo 适用于diffloss
        
        self.nc = nc
        self.nz = nz
        self.n_max = n_max
        self.out_dim =out_dim

        self.mlp_in = nn.ModuleList([])
        for c in range(nc):
            # self.mlp_in.append(nn.Linear(nz + nz * c // nc, d_model))
            self.mlp_in.append(nn.Linear(nc * nz, d_model))

        self.mlp_q = nn.ModuleList([nn.ModuleList([Mlp(d_model, d_model, 4 * [2 * d_model]) for c in range(nc)])
                                    for i in range(n_blocks - 1)])
        self.mlp_k = nn.ModuleList([Mlp(d_model, d_model, 4 * [2 * d_model]) for i in range(n_blocks - 1)])
        self.mlp_v = nn.ModuleList([Mlp(d_model, d_model, 4 * [2 * d_model]) for i in range(n_blocks - 1)])
    
    # input masked indices
    def forward(self, indices, latents=None,padding_mask=None):
        """
        Args:
            indices: 已经mask过的token indices (bs, n_max, nc)
            padding_mask: 指示非pad位置的布尔掩码 (bs, seq_len, nc)
        """
        if latents is None:
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
                z[c] = z[c] + self.pe(z[c])
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
            return out #包含sos token
        else:
            bs, n_max, nc = indices.shape 
            device = indices.device 
            
            # 1. 获取token embeddings 
            z, q, emb = [None] * (nc), [None] * (nc), [None] * (nc) 
            
            # 2. 处理每个通道
            for c in range(nc):
                # 直接从indices获取embeddings
                embeddings = latents[:,:,c] #bs,n_max,nz 
                z[c] = self.mlp_in[c](embeddings)
                
                # 使用可训练的embedding层
                # z[c] = self.tok_emb(indices[..., c])  # (bs, n_max, d_model)
                
                # 2. 如果有padding_mask，将pad位置的embedding设为0
                if padding_mask is not None:
                    z[c] = z[c] * padding_mask[..., c].unsqueeze(-1) #经过dense_zq后pad位置本身就为0
                # 添加位置编码
                z[c] = z[c] + self.pe(z[c])
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
                # emb[c] = self.mlp_out[c](emb[c])
                emb[c] = self.mlp_out_diff[c](emb[c]) # 
            
            out = torch.stack(emb, dim=-2) #(32,21,1,257) / (32,20,1,64)
            return out #包含sos token
                
                

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
 
# def weights_init(m):
#     classname = m.__class__.__name__
#     if "Linear" in classname or "Embedding" == classname: #只对Linear和embedding层初始化
#         print(f"Initializing Module {classname}.")
#         nn.init.trunc_normal_(m.weight.data, 0.0, 0.02) #截断正态分布初始化
#     # elif "Parameter" in classname:
#     #     return nn.init.trunc_normal_(m, 0.0, 0.02)
    
class MAR(nn.Module):
    """ Masked Autoencoder, adapted from MAR code
    """
    def __init__(self, config, data_info):
        super().__init__()
        # MAR Special Parameters
        # config.mar_encoder_dim = 1024 #todo write into config
        # config.mar_decoder_dim = 1024
        
        # -------------------------------------------------------------------------
        # Quantizer Info
        nc = config.model.quantizer.nc #1
        nz = config.model.quantizer.nz #8
        cb_size = config.model.quantizer.codebook_size #256
        n_max = config.data.max_node_num
        device = config.device
        
        # encoder_embed_dim = 1024
        encoder_embed_dim = 512
        # encoder_depth=16
        encoder_depth=8
        # encoder_num_heads=16
        encoder_num_heads=4
        # decoder_embed_dim=1024
        # decoder_embed_dim=512 #mae
        decoder_embed_dim=64 #mlm
        # decoder_depth=16
        decoder_depth=8
        # decoder_num_heads=16
        decoder_num_heads=4
        mlp_ratio=4.
        norm_layer=nn.LayerNorm
        attn_dropout=0.1
        proj_dropout=0.1
        buffer_size=1
        diffloss_d=3
        diffloss_w=1024
        num_sampling_steps='100'
        diffusion_batch_mul=4
        grad_checkpointing=False
        mask_ratio_min=0.7
        label_drop_prob=0.1
        class_num=1
        
        # --------------------------------------------------------------------------
        # MAR init
        self.seq_len = n_max
        self.token_embed_dim = nc*nz
        self.device = device
        self.grad_checkpointing = grad_checkpointing
        # --------------------------------------------------------------------------
        # Class Embedding
        # class_num = data_info.n_class #todo
        self.num_classes = class_num 
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim) #1000,1024 1000个类别，每个类别embedding为1024维
        self.label_drop_prob = label_drop_prob #0.1
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim)) #1,1024
        
        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25) #1.0

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.buffer_size = buffer_size
        self.seq_len = n_max
        self.z_proj = nn.Linear(nc*nz, encoder_embed_dim, bias=True) #8,1024 #TODO input channel 根据forward的x实际形状改变
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))
        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)]) #16,16,4,true,partial(nn.LayerNorm, eps=1e-6),0.1,0.1
        self.encoder_norm = norm_layer(encoder_embed_dim) #1024
        
        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True) #1024,1024
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)) #1,1,1024  mask_token是一个可学习的参数，初始化为0
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim)) #1,256+64,1024    
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)]) #16,16,4,true,partial(nn.LayerNorm, eps=1e-6),0.1,0.1 
        self.decoder_norm = norm_layer(decoder_embed_dim) #1024
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim)) #1,256,1024    

        self.initialize_weights()
        
        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim, #16
            z_channels=decoder_embed_dim, #1024
            width=diffloss_w, #1024
            depth=diffloss_d, #3
            num_sampling_steps=num_sampling_steps, #100
            grad_checkpointing=grad_checkpointing, #false
            device=device
        )
        self.diffusion_batch_mul = diffusion_batch_mul #4
            
    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
    
    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).to(self.device).long()
        return orders
    
    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device) #all 0
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens], #index位置设置为1,表示
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask #bs,seq_len
    
    def forward_mae_encoder(self, x, mask, class_embedding=None):
        # print("x shape ",x.shape) #bs,n_max,nc*nz
        # print(f"mask shape {mask.shape}") #bs,n_max
        x = self.z_proj(x) #bs,n_max,1024
        bsz, seq_len, embed_dim = x.shape
        
        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1) #bs,256+64,1024
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1) #bs,256+64

        # random drop class embedding during training
        # if self.training:
        #     drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
        #     # drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
        #     drop_latent_mask = drop_latent_mask.unsqueeze(-1).to(x.dtype).to(x.device)
        #     class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding
        if class_embedding is not None:
            x[:, :self.buffer_size] = class_embedding.unsqueeze(1) #将class_embedding添加到x的前buffer_size个位置
        else:
            x[:, :self.buffer_size] = self.fake_latent.unsqueeze(1).expand(-1, self.buffer_size, -1) #用fake_latent填充buffer位置
        
        x = x + self.encoder_pos_embed_learned #bs,n_max,1024
        x = self.z_proj_ln(x)
        
        # dropping
        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim) #将mask中为0的位置的x提取出来，只输入未mask的部分

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x) #使用checkpoint进行梯度检查点，防止内存不足
        else:
            for block in self.encoder_blocks:
                x = block(x)
        
        x = self.encoder_norm(x) #bs,seqlen,1024
        return x
        
    
    def forward_mae_decoder(self, x, mask):
        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1) #bs,seq_len+buffersize
        # pad mask tokens
        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype) #bs,seqlen,1024
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2]) 
        
        # decoder position embedding
        x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x) #32,320,1024

        x = x[:, self.buffer_size:] #32,256,1024 去掉buffer_size个位置
        x = x + self.diffusion_pos_embed_learned #32,256,1024
        return x
    
    def forward_loss(self, z, target, mask, node_mask=None):
        #!todo: debug here
        bsz, seq_len, nc, _ = target.shape
        mask = mask[:,:,:,0] #bs,n_max,nc
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1) #bsz*256*4, 16
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1) #bsz*256*4, 1024
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul) #bsz*256*4
        if node_mask is not None:
            node_mask = node_mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul) #bsz*256*4
            loss = self.diffloss(z=z, target=target, mask=mask, node_mask=node_mask) #bsz*256*4
        else:
            loss = self.diffloss(z=z, target=target, mask=mask) #bsz*256*4
        return loss
        
    def forward(self,x,labels=None, node_mask=None, use_mlm=False,target_latents=None,mask=None):
        if not use_mlm:
            gt_latents = x.clone().detach() 
            orders = self.sample_orders(bsz=x.size(0)).to(x.device)
            mask = self.random_masking(x, orders) #bs,n_max
            
            # mae encoder
            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = None
            x = self.forward_mae_encoder(x, mask, class_embedding) #bs,seq_len,8->1024

            # mae decoder
            z = self.forward_mae_decoder(x, mask) #bs,seq_len,1024
            
            # diffloss
            loss = self.forward_loss(z=z, target=gt_latents, mask=mask, node_mask=node_mask) #mask是每次随机mask的mask，node_mask是mask掉pad位置的mask

            return loss
    
        else:
            gt_latents = x.clone().detach() #bs,n_max,c,d_model, mask:bs,n_max,c,d_model, node_mask:bs,n_max
            loss = self.forward_loss(z=gt_latents, target=target_latents, mask=mask, node_mask=node_mask)
            return loss
            
    
    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):
        pass