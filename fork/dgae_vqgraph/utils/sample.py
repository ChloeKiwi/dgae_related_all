import torch
from torch.distributions.categorical import Categorical
from utils.func import get_edge_masks, discretize
import time


def sample_batch(n_samples, transformer, quantizer, decoder):
    with torch.no_grad():
        start = time.time()
        transformer.init_sampler(n_samples)
        zq, masks, indices = sample_prior(n_samples, transformer, quantizer) #生成的zq, masks, indices
        node_masks, edge_masks = get_mask_from_indices(indices, quantizer.n_embeddings)
        zq = quantizer.indices_to_zq(indices.long(), padded=True)
        masks = node_masks.unsqueeze(-1)

        nz, n_max = transformer.nz, transformer.n_max
        #zq = zq.reshape(n_samples, n_max, nc, nz//nc) * node_masks.reshape(n_samples, n_max, 1, 1)
        zq = zq.flatten(2) * masks
        

        annots_recon, edge_quan = decoder(zq.flatten(2), mask=masks.squeeze())
        adjs_recon = torch.matmul(edge_quan, edge_quan.transpose(1, 2))
        adjs_recon = (adjs_recon - adjs_recon.min()) / (adjs_recon.max() - adjs_recon.min())

        # adjs_recon = (adjs_recon.transpose(1, 2) + adjs_recon) * .5
        # edge_masks = get_edge_masks(masks)

        if adjs_recon.shape[-1] == 1:
            edge_masks = get_edge_masks(masks.squeeze())
            adjs_recon = adjs_recon.sigmoid().squeeze().round() * edge_masks.squeeze()
            annots_recon = None
        else: #mol类数据
            adjs_recon = discretize(adjs_recon, masks=edge_masks)
            adjs_recon[:, :, :, 0] = adjs_recon[:, :, :, 0] + (1 - edge_masks.squeeze())
            annots_recon = discretize(annots_recon, masks=masks)
            none_type = 1 - masks.float()
            annots_recon = torch.cat((annots_recon, none_type), dim=-1).detach().cpu()
            adjs_recon = adjs_recon.permute(0, 3, 1, 2).detach().cpu()
        print(f'Time to generate {n_samples}: {time.time()-start:.4f} sec. average: {(time.time()-start)/n_samples:.4f} sec.')
        return annots_recon, adjs_recon

def sample(n_sample, transformer, quantizer, decoder):
    if n_sample <= 1000:
        return sample_batch(n_sample, transformer, quantizer, decoder)
    else:
        K = (n_sample // 1000) #means 1000个一个batch
        mod = n_sample % 1000  #means 剩下的个数
        annots, adjs = sample_batch(n_sample // K, transformer, quantizer, decoder) #第一个batch
        for k in range(K-1): #K-1个batch
            ann, adj = sample_batch(n_sample // K, transformer, quantizer, decoder) #剩下的K-1个batch
            annots = torch.cat((annots, ann), dim=0)
            adjs = torch.cat((adjs, adj), dim=0)
        if mod != 0:
            ann, adj = sample_batch(mod, transformer, quantizer,decoder) #剩下的mod个batch
            annots = torch.cat((annots, ann), dim=0)
            adjs = torch.cat((adjs, adj), dim=0)
    return annots, adjs


# def get_mask_from_indices(indices, mask_indice):
#     bs, n_max, nc = indices.shape
#     device = indices.device
#     node_masks = (indices != mask_indice).int().to(device)
#     node_masks = node_masks.reshape(-1, n_max, nc)
#     node_masks = node_masks[:, :, 0]
#     return node_masks, get_edge_masks(node_masks)

# 1d
def get_mask_from_indices(indices, mask_indice): 
    bs, n_max = indices.shape 
    device = indices.device 
    node_masks = (indices != mask_indice).int().to(device) 
    node_masks = node_masks.reshape(-1, n_max) #bs, n_max
    return node_masks, get_edge_masks(node_masks)
    
def sample_prior(n_samples, transformer, quantizer):
    with torch.no_grad():
        transformer.eval()
        embeddings = quantizer.embedding
        n_embeddings = quantizer.n_embeddings
        n_max = transformer.n_max
        nzn = transformer.nz
        nc = transformer.nc
        nv = nzn // nc
        device = embeddings.device

        tril = torch.tril(torch.full((n_embeddings+1, n_embeddings+1), float('-inf')), diagonal=-1).to(device)
        embeddings = torch.cat((embeddings, torch.zeros(nc, 1, nv).to(device)), dim=1)
        Z = torch.zeros(n_samples, 0, nc, nv).to(device)
        indices = torch.zeros(n_samples, 0, nc).long().to(device)
        for n in range(n_max):
            c_indices = torch.zeros(n_samples, 0).to(device)
            if n == 0:
                z_c = torch.zeros(n_samples, 1, nc, nv).to(device)
            else: z_c = Z[:, -1].unsqueeze(1)
            for c in range(nc):
                # if c == 0:
                #     #z_c = torch.ones(n_samples, 1, 1, nv).to(device)
                #     z_c = torch.zeros(n_samples, 1, nc, nv).to(device)
                logit = transformer.sample(z_c, c, Z)

                if n > 0 and c == 0:
                    logit = logit + tril[indices[:, n - 1, 0].unsqueeze(1)]

                idx = Categorical(logits=logit.softmax(-1).log()).sample()
                if c > 0:
                    idx[c_indices[:, 0] == n_embeddings] = n_embeddings
                else:
                    if n > 0:
                        idx[indices[:, n-1, 0] == n_embeddings] = n_embeddings
                z_c_sampled = embeddings[c, idx]
                z_c = torch.cat((z_c, z_c_sampled.unsqueeze(1)), dim=2)
                c_indices = torch.cat((c_indices, idx), dim=-1)

            Z = torch.cat((Z, z_c[:, :, nc:]), dim=1)
            #Z = torch.cat((Z, z_c[:, :, 1:]), dim=1)
            indices = torch.cat((indices, c_indices.unsqueeze(1)), dim=1).long()
            mask = indices[:, :, 0] != n_embeddings
        return Z, mask.unsqueeze(-1), indices


def sample_prior(n_samples, transformer, quantizer, padding=True):
    with torch.no_grad():
        transformer.train()
        embeddings = quantizer.embedding
        n_embeddings = embeddings.shape[1]
        n_max = transformer.n_max
        nzn = transformer.nz
        nc = transformer.nc
        nv = nzn // nc
        device = embeddings.device
        n_cat = transformer.out_dim

        tril = torch.tril(torch.full((n_cat, n_cat), float('-inf')), diagonal=-1).to(device)

        if padding:
            embeddings = torch.cat((embeddings, torch.zeros(nc, 1, nv).to(device)), dim=1)

        z_completed = torch.zeros(n_samples, 1, nc,  nv).to(device)
        indices = torch.zeros(n_samples, n_max, nc, dtype=torch.long).to(device)

        for i in range(n_max):
            #mask = torch.triu(torch.full((i + 1, i + 1), float('-inf')), diagonal=1).to(device)
            for c in range(nc):
                if c == 0:
                    z_c = z_completed[:, -1].unsqueeze(1)
                logit = transformer.sample(z_c, c, z_completed)[:, -1].unsqueeze(1)

                if i > 0 and c == 0:
                    logit = logit + tril[indices[:, i - 1, 0].unsqueeze(1)]

                idx = Categorical(logits=logit.softmax(-1).log()).sample()
                indices[:, i, c] = idx.squeeze()
                idx_pad = indices[:, i, 0] == n_embeddings
                indices[:, i, c][idx_pad] = n_embeddings
                z_sampled = embeddings[c, indices[:, i, c]]
                z_c = torch.cat((z_c, z_sampled.unsqueeze(1).unsqueeze(2)), dim=2)

            z_completed = torch.cat((z_completed, z_c[:, :, nc:]), dim=1)

        for c in range(1, nc):
            idx = z_completed[:, :, 0, :] == n_embeddings
            z_completed[:, :, c, :][idx] = n_embeddings

        samples = z_completed[:, 1:].flatten(-2)
        mask = indices[:, :, 0] != n_embeddings
    return samples, mask.unsqueeze(-1), indices


# 1d 生成量化序列
def sample_prior(n_samples, transformer, quantizer, padding=True):
    with torch.no_grad():
        # transformer.train()
        embeddings = quantizer.embedding
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(0) # [1, n_embeddings, nz]
        n_embeddings = embeddings.shape[1]  # codebook size:32
        n_max = transformer.n_max          # 最大序列长度
        nz = transformer.nz               # 特征维度:8
        device = embeddings.device
        n_cat = transformer.out_dim      # 输出维度:33

        # 创建因果mask
        tril = torch.tril( #下三角为-inf
            torch.full((n_cat, n_cat), float('-inf')), 
            diagonal=-1
        ).to(device)

        # 添加padding token 32->33
        if padding:
            padding_token = torch.zeros(1, 1, nz).to(device)  # [1, 1, nz]
            embeddings = torch.cat(
                (embeddings, padding_token), 
                dim=1
            )  # [1, n_embeddings+1, nz]

        # 初始化序列
        # import ipdb; ipdb.set_trace()
        z_completed = torch.zeros(n_samples, 1, nz).to(device)  # [bs, 1, nz] start token
        indices = torch.zeros(n_samples, n_max, dtype=torch.long).to(device)  # [bs, seq_len]初始化indices

        # 自回归生成
        for i in range(n_max): #seqlen:num_max_nodes
            # 获取当前状态
            z_c = z_completed[:, -1].unsqueeze(1)  # 取最后一个时间步 [bs, 1, nz]
            
            # 生成下一个token
            logit = transformer.sample(z_c, z_completed)[:, -1].unsqueeze(1)

            # 添加因果mask（确保序列的单调性）
            if i > 0:
                logit = logit + tril[indices[:, i - 1].unsqueeze(1)] #从第2步开始，添加因果mask

            # 采样新token
            idx = Categorical(logits=logit.softmax(-1).log()).sample()
            indices[:, i] = idx.squeeze() #将第i步的token填入indices
            
            # 处理padding
            idx_pad = indices[:, i] == n_embeddings
            indices[:, i][idx_pad] = n_embeddings
            
            # 获取对应的embedding
            z_sampled = embeddings[0, indices[:, i]]  # [bs, nz]
            
            # 更新序列
            z_completed = torch.cat(
                (z_completed, z_sampled.unsqueeze(1)),  #新的token填入z_completed
                dim=1
            )  # [bs, seq_len+1, nz]

        # 准备输出
        samples = z_completed[:, 1:]  # 去掉初始零向量
        mask = indices != n_embeddings  # padding mask

    return samples, mask.unsqueeze(-1), indices