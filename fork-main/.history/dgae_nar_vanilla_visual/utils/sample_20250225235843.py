import torch
from torch.distributions.categorical import Categorical
from utils.func import get_edge_masks, discretize
import time
import torch.nn.functional as F
import numpy as np

def sample_batch(n_samples, transformer, quantizer, decoder, max_node_num=None, mask_func=None):
    with torch.no_grad():
        start = time.time()
        transformer.init_sampler(n_samples)
        # zq, masks, indices = sample_prior(n_samples, transformer, quantizer, T=max_node_num//2)
        zq, node_masks, indices = sample_prior_nar_vanilla(n_samples, transformer, quantizer, decoder, T=max_node_num//2, mask_func=mask_func)
        # node_masks, edge_masks = get_mask_from_indices(indices, quantizer.n_embeddings)
        edge_masks = get_edge_masks(node_masks)
        zq = quantizer.indices_to_zq(indices.long(), padded=True)
        masks = node_masks.unsqueeze(-1)
        zq = zq.flatten(2) * masks
        annots_recon, adjs_recon = decoder(zq.flatten(2), mask=masks.squeeze())
        adjs_recon = (adjs_recon.transpose(1, 2) + adjs_recon) * .5

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
        print(f'Time to generate {n_samples}: {time.time()-start:.4f} sec. average: {(time.time()-start)/n_samples:.4f} sec.') #sample时间包括：transformer生成token，decoder重构图
        return annots_recon, adjs_recon, indices

def sample(n_sample, transformer, quantizer, decoder, max_node_num=None, mask_func=None):
    if n_sample <= 1000:
        return sample_batch(n_sample, transformer, quantizer, decoder, max_node_num, mask_func)
    else:
        K = (n_sample // 1000)
        mod = n_sample % 1000
        annots, adjs = sample_batch(n_sample // K, transformer, quantizer, decoder, max_node_num, mask_func)
        for k in range(K-1):
            ann, adj = sample_batch(n_sample // K, transformer, quantizer, decoder)
            annots = torch.cat((annots, ann), dim=0)
            adjs = torch.cat((adjs, adj), dim=0)
        if mod != 0:
            ann, adj = sample_batch(mod, transformer, quantizer, decoder)
            annots = torch.cat((annots, ann), dim=0)
            adjs = torch.cat((adjs, adj), dim=0)
    return annots, adjs

# 
def get_mask_from_indices(indices, mask_indice):
    bs, n_max, nc = indices.shape
    device = indices.device
    node_masks = (indices != mask_indice).int().to(device)
    node_masks = node_masks.reshape(-1, n_max, nc)
    node_masks = node_masks[:, :, 0]  #bs,n_max
    return node_masks, get_edge_masks(node_masks)

#找到最后一个pad/eos token，其前面的mask为1，后面的都为0
def get_mask_from_indices(indices, mask_indice):
    """
    Args:
        indices: (bs, n_max) 或 (bs, n_max, nc)
        mask_indice: 标记位置的值
    """
    device = indices.device
    # 1. 处理输入维度
    if indices.dim() == 2:
        # 如果是2D tensor，增加一个维度
        indices = indices.unsqueeze(-1)  # (bs, n_max, 1)
    bs, n_max, nc = indices.shape
    # 2. 找第一个mask_indice的位置
    first_mask = (indices[:,:,0] == mask_indice).float()  # (bs, n_max)
    first_mask_pos = first_mask.argmax(dim=1)  # (bs,)
    # 3. 检查是否存在mask_indice
    has_mask = first_mask.sum(dim=1) > 0  # (bs,)
    # 4. 创建位置索引
    pos_idx = torch.arange(n_max, device=device).unsqueeze(0)  # (1, n_max)
    # 5. 生成mask
    mask = pos_idx < first_mask_pos.unsqueeze(1)  # (bs, n_max)
    # 6. 处理没有mask_indice的情况
    mask = torch.where(has_mask.unsqueeze(1), mask, torch.ones_like(mask))
    return mask, get_edge_masks(mask)


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
        # transformer.eval()
        transformer.train()
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
      
        # 处理结果
        indices = cur_ids[:, 1:]  # 移除sos token (bs, max_node_num, nc)
        zq = quantizer.indices_to_zq(indices, padded=True) #(bs, max_node_num, nc, cb_size)
        mask = (indices[:, :, 0] != n_embeddings).unsqueeze(-1) #(bs, max_node_num, 1) 根据第一通道的预测mask掉pad位置
        return zq, mask, indices

# maskgit + predict eos
def sample_prior_nar_vanilla(n_samples, transformer, quantizer, decoder, T=None, mask_func=None, visual_step=False):
    visual_step = True
    
    # 创建保存可视化结果的目录
    import os
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    save_dir = f'./visualization/sample_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        # transformer.eval()
        transformer.train()
        n_embeddings = quantizer.n_embeddings
        n_max = transformer.n_max #max_node_num
        nc = transformer.nc
        embeddings = quantizer.embedding 
        device = embeddings.device
        cb_size = transformer.cb_size
        n_cat = transformer.out_dim #codebook dim
        
        # 初始化全mask序列
        indices = torch.ones(n_samples, n_max, nc, dtype=torch.long, device=device) * transformer.mask_token_id #20,20,1
        sos_tokens = torch.ones(n_samples, 1, nc, dtype=torch.long, device=device) * transformer.sos_token #20,1,1
        cur_ids = torch.cat((sos_tokens, indices), dim=1) #20,21,1
        
        # 记录初始mask数量
        unknown_number = torch.sum(cur_ids == transformer.mask_token_id, dim=1) #bs,20,1
        gamma = transformer.gamma_func(mask_func)

        # 创建一个mask矩阵，将所有位置初始化为负无穷
        #! 禁止 pad,mask token
        # mask_unvalid = torch.full((n_samples, n_max+1, nc, cb_size + 4), float('-inf'), device=device)
        # mask_unvalid[..., :n_embeddings] = 0
        # mask_unvalid[..., transformer.eos_token] = 0
        # mask_unvalid[...,transformer.sos_token] = 0

        #! 禁止mask token
        mask_unvalid = torch.zeros((n_samples, n_max+1, nc, cb_size + 4), device=device)
        mask_unvalid[..., transformer.mask_token_id] = float('-inf') #mask不能预测
        
        T=7
        # T=9
        # T=11
        for t in range(T if T is not None else n_max):
            # 获取所有位置的预测
            logits = transformer(cur_ids) #([20, 21, 1, 259]) torch.Size([20, 21, 2, 20]) (n_samples, max_node_num, nc, cb_size+4)
            logits = logits + mask_unvalid #0-1024（pad）位置为logits，1025-1026位置为无穷小
            
            #! 采样
            sampled_ids = Categorical(logits=logits).sample()
            #! greedy search
            # sampled_ids = torch.argmax(logits, dim=-1)
            
            # 只更新mask位置
            unknown_map = (cur_ids == transformer.mask_token_id) #当前mask位置  bs,21,nc
            sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids) #bs,21,nc,mask位置为此轮预测的id，非mask位置为原id
            
            # 计算当前迭代的mask比例
            ratio = (t + 1) / T
            mask_ratio = gamma(ratio)
            
            
            #! 如果需要可视化当前步骤
            if visual_step and t % 1 == 0:
                # 1. 获取当前indices (移除sos token)
                current_indices = cur_ids[:, 1:]
                
                # 2. 将特殊token转换为pad token (用于quantizer处理)
                indices_for_quantizer = torch.where(
                    current_indices >= n_embeddings,  # 如果是特殊token
                    torch.tensor(n_embeddings-1, device=device),  # 转换为pad token
                    current_indices
                )
                
                # 3. 获取当前mask (非mask token的位置为1)
                current_masks = (current_indices != transformer.mask_token_id)
                
                # 4. 获取当前embedding
                current_zq = quantizer.indices_to_zq(indices_for_quantizer, padded=True)
                current_zq = current_zq.flatten(2) * current_masks
                
                # 5. 使用decoder生成图
                current_annots, current_adjs = decoder(current_zq, mask=current_masks.squeeze())
                current_adjs = (current_adjs.transpose(1, 2) + current_adjs) * .5
                
                # 6. 处理生成的图
                current_edge_masks = get_edge_masks(current_masks.squeeze())
                if current_adjs.shape[-1] == 1:
                    current_adjs = current_adjs.sigmoid().squeeze().round() * current_edge_masks.squeeze()
                    current_annots = None
                else:
                    current_adjs = discretize(current_adjs, masks=current_edge_masks)
                    current_adjs[:, :, :, 0] = current_adjs[:, :, :, 0] + (1 - current_edge_masks.squeeze())
                    current_annots = discretize(current_annots, masks=current_masks)
                                
                #2
                # save_path = os.path.join(save_dir, f'batch_2_step_{t}.png')
                # current_node_labels = current_indices[2, :, 0]  # 假设nc=1，取第一个样本
                # visualize_graph(current_adjs[2].cpu(), 
                #               current_annots[2] if current_annots is not None else None, 
                #               save_path,
                #               node_labels=current_node_labels)
                # save_mask_path = os.path.join(save_dir, f'batch_2_mask_{t}.png')
                # visualize_mask(current_masks[2], save_mask_path, step=t, ratio=mask_ratio)
                
                save_path = os.path.join(save_dir, f'batch_2_step_{t}.png')
                current_node_labels = current_indices[2, :, 0]  # 假设nc=1，取第一个样本
                save_mask_path = os.path.join(save_dir, f'batch_2_mask_{t}.png')
                mask_ratio=visualize_mask(current_masks[2], save_mask_path)
                visualize_graph(current_adjs[2].cpu(), 
                              current_annots[2] if current_annots is not None else None, 
                              save_path,
                              node_labels=current_node_labels,
                              mask_ratio=mask_ratio)
                

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
            
        # 处理结果
        indices = cur_ids[:, 1:]  # 移除sos token (bs, max_node_num, nc)
        zq = None
        
        # 根据最终的EOS位置创建mask
        final_eos_positions = (indices == transformer.eos_token).float().argmax(dim=1)[:, 0]
        final_eos_positions = torch.where(
            final_eos_positions > 0,
            final_eos_positions,
            torch.tensor(n_max, device=device)
        )
        # 将EOS token替换为pad token
        indices = torch.where(indices == transformer.eos_token, 
                            torch.tensor(n_embeddings, device=device), 
                            indices)
        
        mask = torch.arange(n_max, device=device)[None, :] < final_eos_positions[:, None]
        
        # 将indices写入文件
        # write_file(indices)
        return zq, mask, indices

# def visualize_graph(adj, annots=None, save_path=None, node_labels=None):
#     """可视化单个图"""
#     import matplotlib.pyplot as plt
#     import networkx as nx
    
#     # 创建networkx图
#     G = nx.from_numpy_array(adj.numpy())
    
#     # 获取最大连通子图
#     largest_cc = max(nx.connected_components(G), key=len)
#     G = G.subgraph(largest_cc).copy()
    
#     # # 如果提供了节点标签，创建标签字典
#     # if node_labels is not None:
#     #     labels = {i: str(label.item()) for i, label in enumerate(node_labels)}
#     # else:
#     #     labels = {i: str(i) for i in G.nodes()}
        
#         # 如果提供了节点标签，创建标签字典（只保留最大连通子图中的节点标签）
#     if node_labels is not None:
#         labels = {i: str(node_labels[i].item()) for i in G.nodes()}
#     else:
#         labels = {i: str(i) for i in G.nodes()}
        
#     # 设置绘图参数
#     plt.figure(figsize=(4, 4))  # 减小图片尺寸
#     pos = nx.spring_layout(G, k=0.5)  # 增加k值使节点分布更紧凑
    
#     # 绘制节点和边
#     nx.draw(G, pos, 
#             node_color='navy',
#             edge_color='black',
#             node_size=80,  # 减小节点大小
#             width=1,
#             # with_labels=True,
#             with_labels=False,
#             labels=labels,
#             font_size=5,  # 减小字体大小
#             font_color='white')
        
#     # 保存图像时去掉多余的白边
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.1)
#     plt.close()
    
# def visualize_graph(adj, annots=None, save_path=None, node_labels=None):
#     """可视化图，用不同颜色区分最大连通子图和其他节点"""
#     import matplotlib.pyplot as plt
#     import networkx as nx
    
#     # 创建networkx图
#     G = nx.from_numpy_array(adj.numpy())
    
#     # 获取最大连通子图的节点
#     largest_cc = max(nx.connected_components(G), key=len)
    
#     # 获取其他节点
#     other_nodes = set(G.nodes()) - set(largest_cc)
    
#     # 如果提供了节点标签，创建标签字典
#     if node_labels is not None:
#         labels = {i: str(label.item()) for i, label in enumerate(node_labels)}
#     else:
#         labels = {i: str(i) for i in G.nodes()}
        
#     # 设置绘图参数
#     # 设置绘图参数，增加图片尺寸和节点间距
#     plt.figure(figsize=(6, 6))  # 增大图片尺寸
#     pos = nx.spring_layout(G, k=1)  # 增大k值来增加节点间距
    
#     # 分别绘制最大连通子图和其他节点
#     nx.draw_networkx_nodes(G, pos, 
#                           nodelist=list(largest_cc),
#                           node_color='navy',
#                         #   node_size=80)
#                           node_size=300)
#     nx.draw_networkx_nodes(G, pos, 
#                           nodelist=list(other_nodes),
#                           node_color='lightgray',
#                         #   node_size=80)
#                           node_size=300)
#     nx.draw_networkx_edges(G, pos,
#                           width=1.5,
#                           edge_color='black',
#                           alpha=0.5,
#                           style='solid')
    
#     # 如果需要显示标签
#     with_labels = True
#     if  with_labels:
#         nx.draw_networkx_labels(G, pos,
#                               labels=labels,
#                                font_size=8,
#                               font_color='white',
#                               font_weight='bold')
        
#     # 保存图像时去掉多余的白边
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.1)
#     plt.close()


# def visualize_graph(adj, annots=None, save_path=None, node_labels=None):
#     """可视化图，用不同颜色区分最大连通子图和其他节点"""
#     import matplotlib.pyplot as plt
#     import networkx as nx
    
#     # 创建networkx图
#     G = nx.from_numpy_array(adj.numpy())
    
#     # 获取最大连通子图的节点
#     largest_cc = max(nx.connected_components(G), key=len)
    
#     # 创建标签字典并标识mask节点
#     mask_nodes = set()
#     if node_labels is not None:
#         labels = {}
#         for i, label in enumerate(node_labels):
#             if label.item() == 131:  # mask token的id
#                 labels[i] = 'M'
#                 mask_nodes.add(i)
#             else:
#                 labels[i] = str(label.item())
#     else:
#         labels = {i: str(i) for i in G.nodes()}
    
#     # 获取其他节点（非mask且不在最大连通子图中）
#     other_nodes = set(G.nodes()) - set(largest_cc) - mask_nodes
#     # 从最大连通子图中移除mask节点
#     largest_cc = set(largest_cc) - mask_nodes
        
#     # 设置绘图参数
#     plt.figure(figsize=(6, 6))
#     pos = nx.spring_layout(G, k=1)
    
#     # 分别绘制三种节点：最大连通子图、mask节点、其他节点
#     nx.draw_networkx_nodes(G, pos, 
#                           nodelist=list(largest_cc),
#                           node_color='navy',
#                           node_size=300)
#     nx.draw_networkx_nodes(G, pos, 
#                           nodelist=list(mask_nodes),
#                           node_color='lightgray',
#                           node_size=300)
#     nx.draw_networkx_nodes(G, pos, 
#                           nodelist=list(other_nodes),
#                           node_color='lightgray',
#                           node_size=300)
    
#     nx.draw_networkx_edges(G, pos,
#                           width=1.5,
#                           edge_color='black',
#                           alpha=0.5,
#                           style='solid')
    
#     # 如果需要显示标签
#     with_labels = True
#     if with_labels:
#         nx.draw_networkx_labels(G, pos,
#                               labels=labels,
#                               font_size=8,
#                               font_color='white',
#                               font_weight='bold')
        
#     # 保存图像时去掉多余的白边
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.1)
#     plt.close()
    
def visualize_graph(adj, annots=None, save_path=None, node_labels=None, mask_ratio=None):
    """可视化图，用不同颜色区分最大连通子图和其他节点"""
    import matplotlib.pyplot as plt
    import networkx as nx
    
    # 创建networkx图
    G = nx.from_numpy_array(adj.numpy())
    
    # 获取最大连通子图的节点
    largest_cc = max(nx.connected_components(G), key=len)
    
    # 创建标签字典并标识mask节点
    mask_nodes = set()
    valid_nodes = set()  # 新增：用于存储所有非mask的节点
    if node_labels is not None:
        labels = {}
        for i, label in enumerate(node_labels):
            if label.item() == 131:  # mask token的id
                labels[i] = 'M'
                mask_nodes.add(i)
            else:
                labels[i] = str(label.item())
                valid_nodes.add(i)  # 新增：将所有非mask节点加入valid_nodes
    else:
        labels = {i: str(i) for i in G.nodes()}
        valid_nodes = set(G.nodes())
    
    # 设置绘图参数
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G, k=1)
    
    # 只绘制两种节点：mask节点和非mask节点
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=list(valid_nodes),  # 所有非mask节点
                          node_color='navy',
                          node_size=300)
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=list(mask_nodes),  # mask节点
                          node_color='lightgray',
                          node_size=300)
    
    nx.draw_networkx_edges(G, pos,
                          width=1.5,
                          edge_color='black',
                          alpha=0.5,
                          style='solid')
    
    # 如果需要显示标签
    with_labels = True
    if with_labels:
        nx.draw_networkx_labels(G, pos,
                              labels=labels,
                              font_size=8,
                              font_color='white',
                              font_weight='bold')
        
    # 如果提供了mask_ratio，在左下角添加文本标注
    if mask_ratio is not None:
        ax = plt.gca()
        text = f'Mask ratio: {mask_ratio:.2f}'
        ax.text(0.02, 0.02, text,
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(
                    facecolor='white',
                    alpha=0.7,
                    edgecolor='none',
                    pad=4,
                    boxstyle='round,pad=0.5'
                ))
        
        
    # 保存图像时去掉多余的白边
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.close()

def visualize_mask(mask, save_path, step=None, ratio=None):
    """可视化mask矩阵
    Args:
        mask: shape [n_max, 1], 其中:
            1 表示非mask位置 (有效token)
            0 表示mask位置 (待预测)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 获取mask的长度
    seq_len = mask.shape[0]
    actual_ratio = 1 - mask.float().mean().item()  # 计算实际的mask比例
    
    plt.figure(figsize=(10, 0.5))  # 减小高度
    
    # 创建一个1行n列的网格
    data = mask.reshape(1, -1).bool().cpu().numpy()
    
    # 使用自定义颜色映射
    colors = ['#d3d3d3', 'navy']  # 灰色表示mask位置，navy蓝色表示非mask位置
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    ax = plt.gca()
    ax.imshow(data, cmap=cmap, aspect='auto')
    
    # 添加竖线分隔
    for x in range(seq_len + 1):
        ax.axvline(x - 0.5, color='white', linewidth=1)
    
    # # # 如果提供了step和ratio，添加文本标注
    # if step is not None:
    #     text = f'Mask ratio: {actual_ratio:.2f}'
    #     ax.text(0.98, 0.5, text,
    #             horizontalalignment='right',
    #             verticalalignment='center',
    #             transform=ax.transAxes,
    #             fontsize=8,
    #             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    # if step is not None:
    #     text = f'Mask ratio: {actual_ratio:.2f}'
    #     ax.text(0.98, 0.5, text,
    #             horizontalalignment='right',
    #             verticalalignment='center',
    #             transform=ax.transAxes,
    #             fontsize=12,  # 增大字体大小
    #             bbox=dict(
    #                 facecolor='white',
    #                 alpha=0.7,
    #                 edgecolor='none',
    #                 pad=4,    # 增大文本框内边距
    #                 boxstyle='round,pad=0.5'  # 添加圆角效果
    #             ))
    #  # 显示步骤和实际的mask比例
    # if step is not None:
    #     text = f'Step: {step}\nTarget ratio: {ratio:.2f}\nActual ratio: {actual_ratio:.2f}'
    #     ax.text(0.98, 0.5, text,
    #             horizontalalignment='right',
    #             verticalalignment='center',
    #             transform=ax.transAxes,
    #             fontsize=8,
    #             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    # 移除所有刻度和标签
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 移除坐标轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.close()
    
    return actual_ratio


def write_file(indices):
    # 将indices写入文件
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d-%H')
    save_path = f'./indices_{timestamp}.txt'
    with open(save_path, 'a') as f:
        f.write(f'Time: {datetime.now().strftime("%M-%S")}:\n')
        # indices是tensor,需要转换为字符串才能写入
        f.write(str(indices[0].cpu().tolist())) #只看第一个batch
        f.write('\n')
        # for batch_idx in range(indices.shape[0]):
        #     f.write(f'Batch {batch_idx}:\n')
        #     for node_idx in range(indices.shape[1]):
        #         node_indices = indices[batch_idx, node_idx].cpu().tolist()
        #         f.write(f'Node {node_idx}: {node_indices}\n')
        #     f.write('\n')