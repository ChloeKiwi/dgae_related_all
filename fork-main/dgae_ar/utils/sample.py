import torch
from torch.distributions.categorical import Categorical
from utils.func import get_edge_masks, discretize
import time
import torch.nn.functional as F
import numpy as np

def sample_batch(n_samples, transformer, quantizer, decoder):
    with torch.no_grad():
        start = time.time()
        transformer.init_sampler(n_samples)
        zq, masks, indices = sample_prior(n_samples, transformer, quantizer)
        # zq, masks, indices = sample_prior_greedy(n_samples, transformer, quantizer)
        # zq, masks, indices = sample_prior_topk(n_samples, transformer, quantizer)
        # zq, masks, indices = sample_prior_topk(n_samples, transformer, quantizer)
        # zq, masks, indices = sample_prior_beam_search(n_samples, transformer, quantizer)
        node_masks, edge_masks = get_mask_from_indices(indices, quantizer.n_embeddings)
        zq = quantizer.indices_to_zq(indices.long(), padded=True)
        masks = node_masks.unsqueeze(-1)

        nc, nz, n_max = transformer.nc, transformer.nz, transformer.n_max
        #zq = zq.reshape(n_samples, n_max, nc, nz//nc) * node_masks.reshape(n_samples, n_max, 1, 1)
        zq = zq.flatten(2) * masks
        annots_recon, adjs_recon = decoder(zq.flatten(2), mask=masks.squeeze())

        adjs_recon = (adjs_recon.transpose(1, 2) + adjs_recon) * .5
        #edge_masks = get_edge_masks(masks)

        if adjs_recon.shape[-1] == 1:
            edge_masks = get_edge_masks(masks.squeeze())
            adjs_recon = adjs_recon.sigmoid().squeeze().round() * edge_masks.squeeze()
            annots_recon = None
        else:
            adjs_recon = discretize(adjs_recon, masks=edge_masks)
            adjs_recon[:, :, :, 0] = adjs_recon[:, :, :, 0] + (1 - edge_masks.squeeze())
            annots_recon = discretize(annots_recon, masks=masks)
            none_type = 1 - masks.float()
            annots_recon = torch.cat((annots_recon, none_type), dim=-1).detach().cpu()
            adjs_recon = adjs_recon.permute(0, 3, 1, 2).detach().cpu()
        sample_time = time.time()-start
        print(f'Time to generate {n_samples}: {sample_time:.4f} sec. average: {(sample_time)/n_samples:.4f} sec.') #sample时间包括：transformer生成token，decoder重构图
        return annots_recon, adjs_recon, indices, sample_time

def sample(n_sample, transformer, quantizer, decoder):
    if n_sample <= 1000:
        return sample_batch(n_sample, transformer, quantizer, decoder)
    else:
        K = (n_sample // 1000)
        mod = n_sample % 1000
        annots, adjs = sample_batch(n_sample // K, transformer, quantizer, decoder)
        for k in range(K-1):
            ann, adj = sample_batch(n_sample // K, transformer, quantizer, decoder)
            annots = torch.cat((annots, ann), dim=0)
            adjs = torch.cat((adjs, adj), dim=0)
        if mod != 0:
            ann, adj = sample_batch(mod, transformer, quantizer, decoder)
            annots = torch.cat((annots, ann), dim=0)
            adjs = torch.cat((adjs, adj), dim=0)
    return annots, adjs


def get_mask_from_indices(indices, mask_indice):
    bs, n_max, nc = indices.shape
    device = indices.device
    node_masks = (indices != mask_indice).int().to(device)
    node_masks = node_masks.reshape(-1, n_max, nc)
    node_masks = node_masks[:, :, 0]
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
        # transformer.eval()
        embeddings = quantizer.embedding
        n_embeddings = embeddings.shape[1]
        n_max = transformer.n_max
        nzn = transformer.nz
        nc = transformer.nc
        nv = nzn // nc
        device = embeddings.device
        n_cat = transformer.out_dim #codebook dim

        tril = torch.tril(torch.full((n_cat, n_cat), float('-inf')), diagonal=-1).to(device) #约束token的取值范围

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

                idx = Categorical(logits=logit.softmax(-1).log()).sample() #!根据概率分布随机采样
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


def sample_prior_greedy(n_samples, transformer, quantizer, padding=True):
    with torch.no_grad():
        # transformer.train()
        transformer.eval()
        embeddings = quantizer.embedding
        n_embeddings = embeddings.shape[1]
        n_max = transformer.n_max
        nzn = transformer.nz
        nc = transformer.nc
        nv = nzn // nc
        device = embeddings.device
        n_cat = transformer.out_dim #codebook dim

        tril = torch.tril(torch.full((n_cat, n_cat), float('-inf')), diagonal=-1).to(device) #约束token的取值范围

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

                # idx = Categorical(logits=logit.softmax(-1).log()).sample() #!根据概率分布随机采样
                idx = logit.argmax(dim=-1) #!greedy
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

def sample_prior_beam_search(n_samples, transformer, quantizer, beam_width=5, padding=True):
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

        # 初始化beam容器
        beam_scores = torch.zeros(n_samples, beam_width).to(device)
        beam_sequences = torch.zeros(n_samples, beam_width, n_max, nc, dtype=torch.long).to(device)
        beam_z_completed = torch.zeros(n_samples, beam_width, n_max, nc, nv).to(device)

        for i in range(n_max):
            if i == 0:
                # 第一步：为每个样本生成beam_width个候选
                z_c = torch.zeros(n_samples, 1, nc, nv).to(device)
                logits = transformer.sample(z_c, 0, z_c)[:, -1]
                
                # 获取前beam_width个最高概率
                probs = logits.softmax(-1)
                top_probs, top_indices = probs.topk(beam_width, dim=-1)
                
                # 初始化beam
                beam_scores = torch.log(top_probs)
                for b in range(beam_width):
                    beam_sequences[:, b, 0, 0] = top_indices[:, b]
                    beam_z_completed[:, b, 0, 0] = embeddings[0, top_indices[:, b]]
                
                continue

            # 为每个beam生成候选
            all_candidates = []
            all_scores = []
            all_z_completed = []
            
            for b in range(beam_width):
                for c in range(nc):
                    if c == 0:
                        z_c = beam_z_completed[:, b, :i+1].clone()
                        logit = transformer.sample(z_c, c, z_c)[:, -1]
                        
                        if i > 0:
                            logit = logit + tril[beam_sequences[:, b, i-1, 0].unsqueeze(1)]
                        
                        # 获取前beam_width个候选
                        probs = logit.softmax(-1)
                        top_probs, top_indices = probs.topk(beam_width, dim=-1)
                        
                        for k in range(beam_width):
                            candidate = beam_sequences[:, b].clone()
                            candidate[:, i, c] = top_indices[:, k]
                            
                            # 计算新的分数
                            score = beam_scores[:, b] + torch.log(top_probs[:, k])
                            
                            all_candidates.append(candidate)
                            all_scores.append(score)
                            
                            # 更新z_completed
                            z_new = beam_z_completed[:, b].clone()
                            z_new[:, i, c] = embeddings[c, top_indices[:, k]]
                            all_z_completed.append(z_new)

            # 将所有候选转换为tensor
            all_candidates = torch.stack(all_candidates, dim=1)
            all_scores = torch.stack(all_scores, dim=1)
            all_z_completed = torch.stack(all_z_completed, dim=1)

            # 选择前beam_width个最佳候选
            top_scores, top_indices = all_scores.topk(beam_width, dim=1)
            
            # 更新beam
            beam_scores = top_scores
            for b in range(beam_width):
                beam_sequences[:, b] = all_candidates[torch.arange(n_samples), top_indices[:, b]]
                beam_z_completed[:, b] = all_z_completed[torch.arange(n_samples), top_indices[:, b]]

        # 选择最高分数的序列
        best_sequences = beam_sequences[torch.arange(n_samples), beam_scores.argmax(dim=1)]
        best_z_completed = beam_z_completed[torch.arange(n_samples), beam_scores.argmax(dim=1)]

        # 处理padding
        for c in range(1, nc):
            idx = best_z_completed[:, :, 0, :] == n_embeddings
            best_z_completed[:, :, c, :][idx] = n_embeddings

        samples = best_z_completed.flatten(-2)
        mask = best_sequences[:, :, 0] != n_embeddings
        
        return samples, mask.unsqueeze(-1), best_sequences

def sample_prior_topk(n_samples, transformer, quantizer, k=3, temperature=1.0, padding=True):
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

        z_completed = torch.zeros(n_samples, 1, nc, nv).to(device)
        indices = torch.zeros(n_samples, n_max, nc, dtype=torch.long).to(device)

        for i in range(n_max):
            for c in range(nc):
                if c == 0:
                    z_c = z_completed[:, -1].unsqueeze(1)
                logit = transformer.sample(z_c, c, z_completed)[:, -1].unsqueeze(1)

                if i > 0 and c == 0:
                    logit = logit + tril[indices[:, i - 1, 0].unsqueeze(1)]

                # Top-k sampling
                logit = logit / temperature
                top_k_logits, top_k_indices = torch.topk(logit, k, dim=-1)
                probs = F.softmax(top_k_logits, dim=-1)
                
                # 从top-k中采样
                idx_sampled = torch.multinomial(probs.squeeze(1), 1)
                idx = top_k_indices.gather(-1, idx_sampled)
                
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