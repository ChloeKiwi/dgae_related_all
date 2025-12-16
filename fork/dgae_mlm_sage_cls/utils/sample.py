import os
import torch
from torch.distributions.categorical import Categorical
from utils.func import get_edge_masks, discretize
import time
import torch.nn.functional as F
import numpy as np

def sample_batch(n_samples, transformer, quantizer, decoder, max_node_num):
    with torch.no_grad():
        start = time.time()
        transformer.init_sampler(n_samples)
        zq, masks, indices = sample_prior(n_samples, transformer, quantizer, T=max_node_num//2) #生成的zq, masks, indices
        node_masks, edge_masks = get_mask_from_indices(indices, quantizer.n_embeddings) #quantizer.n_embeddings:codebooksize
        zq = quantizer.indices_to_zq(indices.long(), padded=True)
        masks = node_masks.unsqueeze(-1)

        nc, nz, n_max = transformer.nc, transformer.nz, transformer.n_max
        #zq = zq.reshape(n_samples, n_max, nc, nz//nc) * node_masks.reshape(n_samples, n_max, 1, 1)
        zq = zq.flatten(2) * masks 
        
        # transformer->decoder
        annots_recon, adjs_recon = decoder(zq.flatten(2), mask=masks.squeeze()) #解码重构,annots_recon:bs,n_max,0, adjs_recon:bs,n_max,n_max,1
        adjs_recon = (adjs_recon.transpose(1, 2) + adjs_recon) * .5 #
        #edge_masks = get_edge_masks(masks)

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
        print(f'Time to generate {n_samples}: {time.time()-start:.4f} sec. average: {(time.time()-start)/n_samples:.4f} sec.') #sample时间包括：transformer生成token，decoder重构图
        return annots_recon, adjs_recon

def sample(n_sample, transformer, quantizer, decoder, max_node_num):
    if n_sample <= 1000:
        return sample_batch(n_sample, transformer, quantizer, decoder, max_node_num)
    else:
        K = (n_sample // 1000) #means 1000个一个batch
        mod = n_sample % 1000  #means 剩下的个数
        annots, adjs = sample_batch(n_sample // K, transformer, quantizer, decoder, max_node_num) #第一个batch
        for k in range(K-1): #K-1个batch
            ann, adj = sample_batch(n_sample // K, transformer, quantizer, decoder, max_node_num) #剩下的K-1个batch
            annots = torch.cat((annots, ann), dim=0)
            adjs = torch.cat((adjs, adj), dim=0)
        if mod != 0:
            ann, adj = sample_batch(mod, transformer, quantizer,decoder) #剩下的mod个batch
            annots = torch.cat((annots, ann), dim=0)
            adjs = torch.cat((adjs, adj), dim=0)
    return annots, adjs

#todo: 返回的indices会在前面也包括pad1024，如何确定终止位置？
def get_mask_from_indices(indices, mask_indice):
    bs, n_max, nc = indices.shape
    device = indices.device
    # import ipdb; ipdb.set_trace()
    node_masks = (indices != mask_indice).int().to(device)  #todo 这里不对，不是每一个1024都是pad，而是只有最后一个1024及之后是pad？
    node_masks = node_masks.reshape(-1, n_max, nc) #bs,n_max,nc
    node_masks = node_masks[:, :, 0] #用第一个nc的mask作为pad mask
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
        embeddings = quantizer.embedding #1,256,8 
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

        for i in range(n_max): #节点数决定自回归次数
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
        mask = indices[:, :, 0] != n_embeddings #indices等于mask_token_id的位置
    return samples, mask.unsqueeze(-1), indices

def mask_by_random_topk(mask_len, probs, temperature=1.0):
    """
    Args:
        mask_len: [bs, 1, nc] - 每个batch和channel需要mask的数量
        probs: [bs, seq_len, nc] - 每个位置的概率
        temperature: float - 温度参数
    """
    bs, seq_len, nc = probs.shape
    device = probs.device
    # 初始化结果
    masking = torch.zeros(bs, seq_len, nc, dtype=torch.bool, device=device)
    # 对每个通道分别处理
    for c in range(nc):
        # 1. 获取当前通道的数据
        channel_probs = probs[..., c]  # [bs, seq_len]
        channel_mask_len = mask_len[..., c]  # [bs, 1]
        # 2. 计算confidence
        confidence = torch.log(channel_probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(channel_probs.shape).to(device) #bs,seq_len
        # 3. 排序
        sorted_confidence, _ = torch.sort(confidence, dim=-1)  # [bs, seq_len]
        # 4. 获取阈值
        cut_off = torch.gather(sorted_confidence, -1, channel_mask_len.to(torch.long))  # [bs, 1]
        # 5. 确定mask位置
        channel_masking = (confidence < cut_off)  # [bs, seq_len]
        masking[..., c] = channel_masking
    return masking

# # mlm version
_CONFIDENCE_OF_KNOWN_TOKENS=torch.Tensor([torch.inf])

def sample_prior(n_samples, transformer, quantizer, T=10): 
    with torch.no_grad():
        transformer.eval()
        n_embeddings = quantizer.n_embeddings
        n_max = transformer.n_max #max_node_num
        nc = transformer.nc
        embeddings = quantizer.embedding 
        device = embeddings.device
        
        # 初始化全mask序列
        indices = torch.ones(n_samples, n_max, nc, dtype=torch.long, device=device) * transformer.mask_token_id #20,20,1
        sos_tokens = torch.ones(n_samples, 1, nc, dtype=torch.long, device=device) * transformer.sos_token #20,1,1
        cur_ids = torch.cat((sos_tokens, indices), dim=1) #20,21,1
        
        # 记录初始mask数量
        unknown_number = torch.sum(cur_ids == transformer.mask_token_id, dim=1) #bs,20,1
        gamma = transformer.gamma_func("cosine")
        
        predicted_indices = []
        for t in range(T):
            # 获取所有位置的预测
            #todo: 不能预测sos1025、mask1026，只能预测0-1024（pad）
            logits = transformer(cur_ids) #([20, 21, 1, 259]) torch.Size([20, 21, 2, 19]) (n_samples, max_node_num, nc, cb_size+3)
            
            # 处理特殊token
            mask_unvalid = torch.ones_like(logits) * float('-inf') #无穷小mask
            mask_unvalid[..., :n_embeddings+1] = 0 #只允许预测0-1024（pad） 
            logits = logits + mask_unvalid #0-1024（pad）位置为logits，1025-1026位置为无穷小
            
            # 采样
            sampled_ids = Categorical(logits=logits).sample()
            
            # 只更新mask位置
            unknown_map = (cur_ids == transformer.mask_token_id) #当前mask位置  bs,21,nc
            sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids) #bs,21,nc,mask位置为此轮预测的id，非mask位置为原id
            
            # 计算当前迭代的mask比例
            ratio = (t + 1) / T
            mask_ratio = gamma(ratio)
            
            # 基于概率选择mask位置
            logits_valid = logits.clone()
            logits_valid[..., n_embeddings+1:] = float('-inf') #1025-1026位置为无穷小 
            probs = F.softmax(logits_valid, dim=-1) #20,21,2,19
            # probs = F.softmax(logits, dim=-1) #20,21,2,19
            selected_probs = torch.take_along_dim(probs, sampled_ids.unsqueeze(-1), -1).squeeze(-1) #20,21,2
            selected_probs = torch.where(unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS.to(device))#每个位置的mask位置为logits，非mask位置为inf
            
            # 计算需要mask的数量
            mask_len = torch.floor(unknown_number * mask_ratio)  # [bs, nc]
            mask_len = mask_len.unsqueeze(1)  # [bs, 1, nc]
            # 当前mask数量
            current_mask_len = unknown_map.sum(dim=1, keepdim=True)  # [bs, 1, nc]
            mask_len = torch.maximum(
                torch.zeros_like(mask_len),
                torch.minimum(current_mask_len-1, mask_len)
            )  # [bs, 1, nc]
            # 选择confidence较低的位置继续mask
            masking = mask_by_random_topk(mask_len, selected_probs, temperature=(1.0 - ratio))
            cur_ids = torch.where(masking, transformer.mask_token_id, sampled_ids) #bs,max_node_num,nc
            if cur_ids.size(2)==1:
                cur_list = cur_ids.squeeze(-1) # 需要赋值回去，否则squeeze不会改变原tensor
            cur_res = 'iter' + str(t) + ": " + str(cur_list[0].cpu().numpy()) 
            predicted_indices.append(cur_res)
            
        # 处理结果
        indices = cur_ids[:, 1:]  # 移除sos token (bs, max_node_num, nc)
        zq = quantizer.indices_to_zq(indices, padded=True) #(bs, max_node_num, nc, cb_size)
        mask = (indices[:, :, 0] != n_embeddings).unsqueeze(-1) #(bs, max_node_num, 1) 根据第一通道的预测mask掉pad位置
        
        # 将predicted id写入文件方便观察
        file_path = os.path.join("models_own", "baseline_cb256_1-mlm/community_sample", "indices.txt")
        write_into_files(predicted_indices, file_path)
        return zq, mask, indices

def write_into_files(predicted_list, file_path):
    # 设置 numpy 打印选项，确保完整输出数组
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in predicted_list:
            f.write(item + '\n')
    # 重置 numpy 打印选项为默认值
    np.set_printoptions(threshold=1000)
    print(f"predicted ids has been written in {file_path}")