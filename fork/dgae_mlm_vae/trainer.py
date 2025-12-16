import math

import numpy as np
import utils.func
import wandb
from typing import Tuple
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.data import Data
import torch
import time
from utils.func import plot_graphs, get_edge_masks, init_model
from utils.sample import sample
from utils.eval import get_mol_metric, init_autoencoder_running_metrics, init_prior_running_metrics
from utils.mol_utils import gen_mol
from utils.losses import get_losses
from utils.sample import sample_batch
from utils.func import dense_zq
from graph_stats.stats import eval_torch_batch
from utils.logger import log_running_metrics, log_step_autoencoder, save_model, log_mol_metrics, \
    log_step_prior, log_mmd_metrics
from model import Encoder, Decoder, Quantizer
from tqdm import tqdm
import os  
from utils.func import discretize


class Trainer:
    def __init__(self, dataloaders, config, data_info):
        self.device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")
        print(f'Run training on : {self.device}')
        config.device = self.device

        self.train_loader, self.test_loader = dataloaders

        if hasattr(data_info, 'n_node_class') and data_info.n_node_class > 1:
            self.node_classify = True
        if hasattr(data_info, 'n_graph_class') and data_info.n_graph_class > 1:
            self.graph_classify = True
    
        model = init_model(config, data_info, self.device) 
        self.encoder, self.decoder, self.quantizer, self.transformer, self.opt, self.scheduler = model

        # Extract configuration variables
        self.epochs = config.training.epochs
        self.max_node_num = config.data.max_node_num
        self.annotated_nodes = data_info.annotated_nodes
        self.annotated_edges = data_info.annotated_edges
        self.n_node_feat = data_info.n_node_feat
        self.dataset = config.data.data
        self.decay_iteration = config.training.decay_iteration
        self.cb_size = config.model.quantizer.codebook_size
        self.gamma = config.model.gamma
        self.n_logging_steps = config.log.n_loggin_steps
        self.n_logging_epochs = config.log.n_loggin_epochs
        self.mol_data = data_info.mol
        self.init_steps = config.model.quantizer.init_steps
        # self.use_vq = not config.model.quantizer.turn_off
        # self.use_vq = True
        self.use_vq = False #
        self.use_vae = True
        # self.use_vq = config.quantization if config.quantization is not None else False #todo 在run.sh中设置false时没用
        if config.train_prior: #prior时需要排序
            self.sort_indices = utils.func.sort_indices
            
        # for mlm
        # pad_token = self.cb_size
        self.use_mask = config.use_mask
        self.sos_token = self.cb_size + 1
        self.mask_token_id = self.cb_size + 2
        self.mask_gamma = self.gamma_func("cosine")
        self.nc = config.model.quantizer.nc
        print(f"use_mask: {self.use_mask}, use vq: {self.use_vq}, sos_token: {self.sos_token}, mask_token_id: {self.mask_token_id}, cb_size: {self.cb_size}, nc: {self.nc}, wandb:{config.log.wandb}, is mol:{self.mol_data}, recon nodes:{self.annotated_nodes}")

        # Define Logger
        if config.train_prior:
            self.metrics = init_prior_running_metrics()
        else:
            self.metrics = init_autoencoder_running_metrics(self.annotated_nodes)

        if config.work_type != 'sample':
            run_id = f'{config.exp_name}_{config.run_name}_{config.work_type}'
            wandb.init(project=f'VQ-GAE_{config.dataset}', config=config, mode=config.log.wandb, 
                       name=f'{config.exp_name}_{config.run_name}_{config.work_type}', id=run_id, resume=False)
            wandb.save("*.pt")
            
            config.wandb_dir = os.path.basename(os.path.dirname(wandb.run.dir)) # 获取wandb_run_id
            
            batch = next(iter(self.test_loader))
            self.n_samples = batch.num_graphs
        else:
            self.n_samples = config.n_samples
        if config.log.wandb == 'online' or config.log.wandb == 'offline':
            self.wandb = wandb
        else:
            self.wandb = None
        self.best_run = {}
        self.config = config
    
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
        
    def recon_adj_from_decoder(self, nodes_rec, edges_rec, masks, edge_masks):
        masks = masks.to(self.device)
        edge_masks = edge_masks.to(self.device)
        adjs_recon = (edges_rec.transpose(1, 2) + edges_rec) * .5
        if adjs_recon.shape[-1] == 1:
            edge_masks = get_edge_masks(masks.squeeze())
            adjs_recon = adjs_recon.sigmoid().squeeze().round().to(self.device) * edge_masks.squeeze().to(self.device)
        else:
            adjs_recon = discretize(adjs_recon, masks=edge_masks)
            adjs_recon[:, :, :, 0] = adjs_recon[:, :, :, 0].to(self.device) + (1 - edge_masks.squeeze()).to(self.device)
            annots_recon = discretize(nodes_rec, masks=masks)
            none_type = 1 - masks.float()
            annots_recon = torch.cat((annots_recon, none_type), dim=-1).detach().cpu()
            adjs_recon = adjs_recon.permute(0, 3, 1, 2).detach().cpu()
        return adjs_recon
        
    def autoencoder(self) -> None:
        print(f'Train data contains {len(self.train_loader)} batches with size {self.train_loader.batch_size}')
        times = time.time(), time.process_time()
        step = 0
        if self.init_steps > 0:
            print('Initialize autoencoder starts...total epochs: ',self.epochs)
            self.quantizer.collect = False
            for epoch in tqdm(range(1, self.epochs)):
                for batch in self.train_loader:
                    # Train the autoencoder
                    self.fit_autoencoder(batch.to(self.device)) #打包好的图数据：DataBatch(edge_index=[2, 1902], label=[504], graph_label=[16], max_num_nodes=[16], num_nodes=504, x=[504, 15], edge_index_ext=[2, 7032], edge_attr_ext=[7032, 3], batch=[504], ptr=[17])
                    step += 1
                    if step >= self.init_steps:
                        self.quantizer.collect = True

                    if self.init_steps == 0:    
                        break
                if self.init_steps == 0:
                    break

        print('Train autoencoder starts...total epochs: ',self.epochs)
        step = 0
        for epoch in tqdm(range(1, self.epochs)):
            for batch_id, batch in enumerate(self.train_loader):
                # self.fit_autoencoder(batch.to(self.device))
                nodes_rec, edges_rec, node_masks, edge_masks = self.fit_autoencoder(batch.to(self.device)) #edges:(bs,1,n_max,n_max)
                adjs_recon = self.recon_adj_from_decoder(nodes_rec, edges_rec, node_masks.to(self.device), edge_masks.to(self.device))
                step += 1
                if step % self.n_logging_steps == 0:
                    log_running_metrics(self.metrics, self.wandb, step, key='iter', times=times)
                    if self.mol_data:
                        log_mol_metrics(nodes_rec, adjs_recon, self.dataset, key='ae train')
                    else:
                        metrics, metrics_novel_unique = log_mmd_metrics(batch, adjs_recon, key='ae train')
                        print(f"reconstruction mmd at train epoch {epoch}: {metrics}")

            # Test
            for batch_id, batch in enumerate(self.test_loader):
                with torch.no_grad():
                    annots, adjs, node_masks, edge_masks = self.fit_autoencoder(batch.to(self.device), train=False) #return nodes_rec, edges_rec

            if epoch % self.n_logging_epochs == 0:
                logged_metrics = log_running_metrics(self.metrics, self.wandb, step, key='train')
                to_save = (self.encoder, self.decoder, self.quantizer, self.opt, self.scheduler)
                self.best_run = save_model(logged_metrics['recon_loss'], self.best_run, to_save, step,
                                           prefix='train_rec_loss', minimize=True, wandb=self.wandb, save_dir=f'./models_own/{self.config.exp_name}/{self.config.dataset}_autoencoder/files')
                logged_metrics = log_running_metrics(self.metrics, self.wandb, step, key='val', times=times)
                to_save = (self.encoder, self.decoder, self.quantizer, self.opt, self.scheduler)
                self.best_run = save_model(logged_metrics['recon_loss'], self.best_run, to_save, step,
                                           prefix='val_rec_loss', minimize=True, wandb=self.wandb, save_dir=f'./models_own/{self.config.exp_name}/{self.config.dataset}_autoencoder/files')

                if self.mol_data:
                    log_mol_metrics(annots, adjs_recon, self.dataset, key='ae val)') #adjs:bs,4,9,9
                else: #! 是否计算重构mmd、绘制重构图
                    metrics,_ = log_mmd_metrics(batch, adjs_recon, key='ae val')
                    print(f"reconstruction mmd at val epoch {epoch}: {metrics}")
                    
                    adjs_recon = self.recon_adj_from_decoder(annots, adjs, node_masks.to(self.device), edge_masks.to(self.device))
                    plot_graphs(adjs_recon[:20], max_plot=20, wandb=self.wandb, title='reconstruction data', filepath=f"./plots/{self.config.exp_name}", filename=f"{self.config.dataset}_recon_graph_epoch_{epoch}")


            if step % self.decay_iteration == 0 and self.scheduler.get_last_lr()[0] > 2 * 1e-5:
                self.scheduler.step()
                print('New learning rate: ', self.scheduler.get_last_lr())
                
    def fit_autoencoder(self, batch: Data, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        # Set model mode (train/eval) based on the "train" parameter
        if train:
            self.encoder.train()
            self.quantizer.train()
            self.decoder.train()
            self.opt.zero_grad()
        else:
            self.encoder.eval()
            self.quantizer.eval()
            self.decoder.eval()
            
        #! vqgraph pipeline: gnn_layer1 -> quantizer -> 2 mlp decoder -> gnn_layer2
        ze, w0 = self.encoder(batch) #gnn encoder，ze:node_feat, w0:edge_feat
        # TODO: vae不需要init_steps去收集样本?
        self.init_steps = 0
        if self.init_steps > 0:
            zq = ze
            if self.quantizer.collect:
                collect = self.quantizer.collect_samples(zq.reshape(zq.shape[0], self.quantizer.nc,
                                                                 self.quantizer.embedding_dim).detach())
            else:
                collect = True
        else:
            if self.use_vq:
                zq, commit, codebook, perplex, indices = self.quantizer(ze)
                codebook_usage = round(perplex.item()/(self.cb_size**self.nc),3)
            elif self.use_vae:
                zq, kl_loss, vae_recon_loss, vae_total_loss, _ = self.quantizer(ze) #没有进行量化的klvae
                codebook, commit = torch.zeros(1).to(self.device), torch.zeros(1).to(self.device)
                perplex = torch.zeros(1).to(self.device)
                codebook_usage = 0
            else:
                zq = ze
                codebook, commit = torch.zeros(1).to(self.device), torch.zeros(1).to(self.device)
                perplex = torch.zeros(1).to(self.device)
                codebook_usage = 0
        zq, node_masks = to_dense_batch(zq, batch.batch, max_num_nodes=self.max_node_num)

        masks = node_masks.detach(), get_edge_masks(node_masks)
        nodes_rec, edges_rec = self.decoder(zq, mask=node_masks)
        
        # Compute partial losses and reconstruction
        recon_loss, rec_losses = get_losses(batch, nodes_rec, edges_rec, node_masks.unsqueeze(-1), self.annotated_nodes,
                                            self.annotated_edges, self. max_node_num, self.n_node_feat, masks[1],self.mol_data)
        
        #recon loss是图重构loss
        if self.init_steps>0:
            loss = recon_loss
        else:
            loss = recon_loss + (codebook + commit) * self.gamma

        if self.use_vae:
            loss = recon_loss + vae_total_loss * self.gamma #没有codebook commit loss
        
        if train:
            loss.backward()
            self.opt.step()
            
        # Evaluate and log the results metrics, batch, rec_losses, vq_losses, masks, n_node_feat, train, annotated_nodes)
        if self.init_steps > 0:
            if not collect:
                self.init_steps = 0
        else:
            vq_losses = loss, recon_loss, codebook, commit, perplex, codebook_usage
            if self.use_vae:
                vq_losses = loss, recon_loss, codebook, commit, perplex, codebook_usage, kl_loss, vae_recon_loss
            nodes_rec, edges_rec = log_step_autoencoder(self.metrics, batch, rec_losses, vq_losses, masks,
                                            self.n_node_feat, train, self.annotated_nodes,self.mol_data, self.use_vae)

        # return nodes_rec, edges_rec.permute(0, 3, 1, 2), node_masks.unsqueeze(-1), masks[1] #返回node masks, edge masks 以供log mmd
        return nodes_rec, edges_rec, node_masks.unsqueeze(-1), masks[1] #返回node masks, edge masks 以供log mmd

    def prior(self) -> None:
        print(f'Train data contains {len(self.train_loader)} batches with size {self.train_loader.batch_size}')
        print('Train prior starts...')
        times = time.time(), time.process_time()
        step = 1
        ref = torch.zeros(0, self.max_node_num, self.max_node_num).to(self.device)
        # n_sample = self.n_samples
        # print("Collecting reference graphs from training set...")
        # with torch.no_grad():
        #     for batch in self.train_loader:
        #         adj = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=self.max_node_num).to(self.device)
        #         ref = torch.cat((ref,adj), dim=0)
        #         if ref.shape[0]>=self.n_samples:
        #             ref = ref[:self.n_samples]
        #             break
        
        # # 计算第一次baseline mmd
        # gen_metrics_baseline, gen_metrics_baseline_uni = log_mmd_metrics(batch[:self.n_samples], ref)
        # gen_metrics_baseline['avg']
        # print(f'Trainset baseline metrics: {gen_metrics_baseline}')
        # print(f'Trainset baseline metrics: {gen_metrics_baseline_uni}')
        
        for epoch in tqdm(range(1, self.epochs)):
            for batch in self.train_loader:
                # Train the prior
                self.fit_prior(batch.to(self.device))
                # if ref.shape[0] < 117: #构造训练baseline
                #    adj = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=self.max_node_num).to(self.device)
                #    ref = torch.cat((ref, adj), dim=0)
                if step % self.n_logging_steps == 0:
                    metrics = log_running_metrics(self.metrics, self.wandb, step, key='iter', times=times)
                    annots, adjs = sample_batch(self.n_samples, self.transformer, self.quantizer, self.decoder) #生成样本

                    if self.mol_data:
                        gen_metrics = log_mol_metrics(annots, adjs, self.dataset, key='prior train')
                        key = 'nspdk'
                        to_save = (self.transformer, self.opt, self.scheduler)
                        self.best_run = save_model(gen_metrics[key], self.best_run, to_save, step,
                                                   prefix=f'iter_{key}', minimize=True, wandb=self.wandb, prior=True, save_dir=f'./models_own/{self.config.exp_name}/{self.config.dataset}_prior')

                    else:
                        # gen_metrics = log_mmd_metrics(annots, adjs, key='iter') #计算生成样本的mmd
                        gen_metrics, gen_metrics_novel_unique = log_mmd_metrics(batch, adjs, key='prior train') #计算生成样本的mmd
                step += 1

            # Test the prior
            for batch in self.test_loader:
                batch = batch.to(self.device)
                with torch.no_grad():
                    self.fit_prior(batch, train=False)

            # Log the results
            if epoch % self.n_logging_epochs == 0:
                log_running_metrics(self.metrics, self.wandb, step, key='train', times=times)
                log_running_metrics(self.metrics, self.wandb, step, key='val', times=times)
                annots, adjs = sample(self.n_samples, self.transformer, self.quantizer, self.decoder, T=10, use_mlm=self.use_mask)
                if self.mol_data:
                    gen_metrics = log_mol_metrics(annots, adjs, self.dataset, key='prior val')
                    key = 'nspdk'
                else:
                    gen_metrics,_ = log_mmd_metrics(batch, adjs, key='prior val')
                    key = 'avg'
                print(f'epoch {epoch} mmd: {gen_metrics}')

                # gen_metrics = log_mmd_metrics(batch, ref, key='trainset')
                # print(f'trainset mmd: {gen_metrics}')
                #ref = torch.zeros(0, 125, 125)

                to_save = (self.transformer, self.opt, self.scheduler)
                self.best_run = save_model(gen_metrics[key], self.best_run, to_save, step,
                                           prefix=f'epoch_{key}', minimize=True, wandb=self.wandb, prior=True, save_dir=f'./models_own/{self.config.exp_name}/{self.config.run_name}/{self.config.dataset}_prior')

            if step % self.decay_iteration == 0 and self.scheduler.get_last_lr()[0] > 2 * 1e-5:
                self.scheduler.step()
                print('New learning rate: ', self.scheduler.get_last_lr())

    def fit_prior(self, batch: Data, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        # Set model mode (train/eval) based on the "train" parameter
        if train:
            self.opt.zero_grad()
            self.transformer.train()
        else:
            self.transformer.eval()
        self.encoder.eval()
        self.quantizer.eval()
        self.decoder.eval()

        # Encoding and quantization
        ze, w0 = self.encoder(batch)
        zq, indices = self.quantizer.encode(ze) #indices:正确索引(?,nc)，zq:量化后的节点特征(?,nc,nz)
        zq, masks, indices = dense_zq(zq, indices, self.quantizer, batch, self.max_node_num, self.sort_indices)
        #zq:(bs,max_node_num,nc,nz),masks:(max_node_num,max_node_num),indices:(bs,max_node_num,nc)
        
        ignore_pad = False
        if self.config.use_mask and ignore_pad==True:
            # 1. 创建mask,但要避免mask pad位置
            bs, seq_len, nc = indices.shape
            # self.cb_size已经作为pad token
            padding_mask = (indices != self.cb_size) #pad token:256
            # 计算每个序列的实际长度(非pad的位置数)
            valid_lengths = padding_mask.sum(dim=1)  # (bs, nc)
            # 2. 只在非pad位置中选择mask位置
            mask = torch.zeros_like(indices, dtype=torch.bool)
            for b in range(bs):
                for c in range(nc):
                    valid_len = valid_lengths[b, c]
                    if valid_len > 0:  # 确保有非pad位置
                        # 只在非pad位置中随机选择mask位置
                        num_mask = math.floor(self.mask_gamma(np.random.uniform()) * valid_len)
                        # 随机选择mask位置（只在非pad位置中选择）
                        mask_indices = torch.randperm(valid_len.item(), device=indices.device)[:num_mask]
                        mask[b, mask_indices, c] = True #被mask的位置为True
            # 3. 应用mask，保持pad位置不变
            masked_indices = torch.where(
                padding_mask,  # 只在非pad位置进行mask
                torch.where(
                    mask,
                    self.mask_token_id,  # mask位置
                    indices  # 保持原值
                ),
                self.cb_size  # pad位置保持不变
            )
            
            sos_tokens = torch.ones(indices.shape[0], 1, self.nc,dtype=torch.long, device=indices.device) * self.sos_token #(bs,1,nc)
            
            a_indices = torch.cat((sos_tokens, masked_indices), dim=1) #(bs,max_node_num+1,nc)
            target = torch.cat((sos_tokens, indices),dim=1) #(bs,max_node_num+1,nc)
            padding_mask = (a_indices != self.mask_token_id) #(bs,max_node_num+1,nc)
            
            logits = self.transformer(self.config.use_mask, indices=a_indices, padding_mask=padding_mask) #(32,21,1,257)
            
            # 计算loss时需要忽略：
            # 1. pad位置 (256)
            # 2. 未被mask的位置（即原始token位置）
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),  # (bs * seq_len * nc, 259)
                target.reshape(-1),                    # (bs * seq_len * nc)
                ignore_index=self.cb_size,            # 忽略pad token (256)
                reduction='none'                       # 先不要reduce，方便我们mask掉非mask位置
            )

            # 重塑loss以匹配输入形状
            loss = loss.reshape(logits.shape[:-1])    # (bs, seq_len, nc)

            # 只计算被mask位置的loss
            mask = (a_indices == self.mask_token_id)  # 找出被mask的位置
            loss = loss * mask                        # 只保留mask位置的loss

            # 计算平均loss（只考虑mask位置）
            loss = loss.sum() / (mask.sum() + 1e-6)   # 加上epsilon避免除零
        
        elif self.config.use_mask and ignore_pad==False:
            bs, seq_len, nc = indices.shape
            # mask = torch.zeros_like(indices, dtype=torch.bool)
            num_mask = math.floor(self.mask_gamma(np.random.uniform()) * seq_len)
            random_indices = torch.rand(indices.shape, device=indices.device).topk(k=num_mask, dim=1).indices
            mask = torch.zeros(indices.shape,dtype=torch.bool,device=indices.device)
            mask.scatter_(dim=1, index=random_indices, value=True) #index处为True
            masked_indices = self.mask_token_id * torch.ones_like(indices, device=indices.device) #mask_token_id处为True
            
            masked_indices = mask * indices + (~mask) * masked_indices
            
            sos_tokens = torch.ones(indices.shape[0], 1, self.nc,dtype=torch.long, device=indices.device) * self.sos_token #(bs,1,nc)
            
            a_indices = torch.cat((sos_tokens, masked_indices), dim=1) #(bs,max_node_num+1,nc)
            target = torch.cat((sos_tokens, indices),dim=1) #(bs,max_node_num+1,nc)
            
            logits = self.transformer(self.config.use_mask, indices=a_indices) #(32,21,1,257)
            
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),  # (bs * seq_len * nc, 259)
                target.reshape(-1),                    # (bs * seq_len * nc)
                ignore_index=self.mask_token_id            # 忽略 mask id 
            )
            
        else:
            # causal transformer
            logits = self.transformer(self.config.use_mask, embeddings=zq, mask=masks) #teacher forcing, auto-regressive training with casual mask
            
            # Apply masks to the output if needed
            if self.sort_indices:
                mask = get_output_mask(logits.shape[0], indices, self.quantizer.n_embeddings, self.device, logits.shape[2])
                logits = logits + mask #todo:RuntimeError: The size of tensor a (259) must match the size of tensor b (257) at non-singleton dimension 3

            # Calculate the loss
            loss = F.cross_entropy(logits.permute(0, 3, 1, 2), indices)
            
        if train:
            loss.backward()
            self.opt.step()
        log_step_prior(self.metrics, loss, train)

        return zq, logits

    def sample(self) -> None:
        with torch.no_grad():
            self.transformer.eval()
            self.quantizer.eval()
            self.decoder.eval()
            if self.mol_data:
                print('Generation mol graph starts... ')
                self.n_samples = 1000
                start_time = time.time()
                start = time.time()
                annots, adjs = sample(self.n_samples, self.transformer, self.quantizer, self.decoder, T=30, use_mlm=self.use_mask)
                print(time.time() - start)
                gen_mols, num_no_correct = gen_mol(annots.cpu(), adjs.cpu(), self.dataset)
                stop = time.time()
                print(stop - start)
                metrics = get_mol_metric(gen_mols, self.dataset, num_no_correct)
                print(metrics)

            else:
                print('Generation graph starts... ')
                for batch in self.test_loader:
                    annots, adjs = sample(self.n_samples, self.transformer, self.quantizer, self.decoder, T=10, use_mlm=self.use_mask)
                    ref = to_dense_adj(batch.edge_index, batch=batch.batch, max_num_nodes=self.max_node_num)

                    adjs_trainset = torch.zeros(0, self.max_node_num, self.max_node_num) #用于计算baseline
                    for train_batch in self.train_loader:
                        adj_trainset = to_dense_adj(train_batch.edge_index, batch=train_batch.batch, max_num_nodes=self.max_node_num)
                        adjs_trainset = torch.cat((adjs_trainset, adj_trainset), dim=0)
                        if adjs_trainset.shape[0] >= 117:
                            break
                    adjs = adjs[:117] #取前117个
                    adjs_trainset = adjs_trainset[:117]
                    metrics, metrics_novel_unique = eval_torch_batch(ref_batch=ref, pred_batch=adjs, methods=None)
                    metrics_baseline, metrics_novel_unique_baseline = eval_torch_batch(ref_batch=ref, pred_batch=adj_trainset, methods=None)
                    metrics['avg'] = round(sum(metrics.values()) / 3, 4) #计算平均值并保留6位小数
                    print("sample : ",metrics)
                    metrics_baseline['avg'] = round(sum(metrics_baseline.values()) / 3, 4)
                    print("baseline : ", metrics_baseline)
                    
                    plot_graphs(adjs[:20], max_plot=20, wandb=None, title='generated data', filepath=f"./plots/{self.config.exp_name}", filename=f"{self.config.dataset}_sample_graph")
                    
        return sample_batch


def get_output_mask(bs: int, indices: torch.Tensor, cb_size: int, device: torch.device,
                    output_shape: int) -> torch.Tensor:
    # causal mask
    tril = torch.tril(torch.full((cb_size + 1, cb_size + 1), float('-inf')), diagonal=-1).to(device)
    idx = torch.cat((torch.zeros(bs, 1).to(device).long(), indices[:, :-1, 0]), dim=1)
    mask0 = tril[idx]

    if output_shape == 1:
        mask = mask0.unsqueeze(2)
    else:
        mask = mask0.unsqueeze(-2)
        for c in range(indices.shape[-1]-1):
            mask1 = torch.zeros(mask0.shape).to(device)
            mask1[indices[..., 0] == cb_size] = tril[cb_size]
            mask = torch.cat((mask, mask1.unsqueeze(-2)), dim=-2)
    return mask