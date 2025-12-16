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
        
        # Data info
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
        #self.quantization = not config.model.quantizer.turn_off
        self.quantization = True
        if config.train_prior: #prior时需要排序
            self.sort_indices = utils.func.sort_indices

        # Define Logger
        if config.train_prior:
            self.metrics = init_prior_running_metrics()
        else:
            self.metrics = init_autoencoder_running_metrics(self.annotated_nodes)

        if config.work_type != 'sample':
            # wandb.init(project=f'VQ-GAE_{config.dataset}_{config.work_type}', config=config, mode=config.log.wandb, name=f'{config.exp_name}')
            # wandb.save("*.pt")
            
            run_id = f'{config.exp_name}_{config.work_type}'
            wandb.init(project=f'VQ-GAE_{config.dataset}', config=config, mode=config.log.wandb, 
                       name=f'{config.exp_name}_{config.work_type}', id=run_id, resume=False)
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
        ini_step = 0
        if self.init_steps > 0:
            print('Initialize autoencoder starts...total epochs: ',self.epochs)
            self.quantizer.collect = False
            for epoch in tqdm(range(1, self.epochs)):
                for batch in self.train_loader:
                    # Train the autoencoder
                    self.fit_autoencoder(batch.to(self.device)) #打包好的图数据：DataBatch(edge_index=[2, 1902], label=[504], graph_label=[16], max_num_nodes=[16], num_nodes=504, x=[504, 15], edge_index_ext=[2, 7032], edge_attr_ext=[7032, 3], batch=[504], ptr=[17])
                    ini_step += 1
                    if ini_step >= self.init_steps:
                        self.quantizer.collect = True

                    if self.init_steps == 0:    
                        break
                if self.init_steps == 0:
                    break

        print('Train autoencoder starts...total epochs: ',self.epochs)
        step = 0
        for epoch in tqdm(range(1, self.epochs)):
            # # 用于累积一个epoch的重构结果
            # train_recon_adjs = []
            # train_orig_adjs = []
            
            for batch in self.train_loader:
                # Train the autoencoder
                nodes_rec, edges_rec, node_masks, edge_masks = self.fit_autoencoder(batch.to(self.device))
                adjs_recon = self.recon_adj_from_decoder(nodes_rec, edges_rec, node_masks.to(self.device), edge_masks.to(self.device))
                
                # 收集重构结果和原始图
                # if epoch % self.n_logging_epochs == 0:  # 只在需要评估的epoch收集
                    # train_recon_adjs.append(adjs_recon)
                    # orig_adj = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=self.max_node_num)
                    # train_orig_adjs.append(orig_adj)
                
                step += 1
                if step % self.n_logging_steps == 0:
                    log_running_metrics(self.metrics, self.wandb, step, key='iter', times=times)
                        
            # Test
            # test_recon_adjs = []
            # test_orig_adjs = []
            for batch_id, batch in enumerate(self.test_loader):
                with torch.no_grad():
                    # annots, adjs = self.fit_autoencoder(batch.to(self.device), train=False) #return nodes_rec, edges_rec
                    annots, adjs, node_masks, edge_masks = self.fit_autoencoder(batch.to(self.device), train=False) #return nodes_rec, edges_rec
                    adjs_recon = self.recon_adj_from_decoder(annots, adjs, node_masks, edge_masks)
                    
                    # test_recon_adjs.append(adjs_recon)
                    # 收集原始邻接矩阵
                    # orig_adj = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=self.max_node_num)
                    # test_orig_adjs.append(orig_adj)
                    
                    # 绘制阶段重构图
                    if batch_id == len(self.test_loader)-2 and epoch%10==0: 
                        # adjs_recon = edges_rec.squeeze(dim=1) #(bs,n_max,n_max) this is wrong
                        if not self.mol_data:
                            plot_graphs(adjs_recon[:20], max_plot=20, wandb=None, title='reconstruction data', 
                                  filepath=f"./plots/{self.config.exp_name}", 
                                  filename=f"{self.config.dataset}_recon_graph_test_{epoch}")
            
            if epoch % self.n_logging_epochs == 0:
                #! 每n_logging_epochs个epoch计算一次训练集和测试集的重构MMD
                # if not self.mol_data:
                    # # 合并所有原始图
                    # all_train_orig = torch.cat(train_orig_adjs, dim=0)
                    # all_test_orig = torch.cat(test_orig_adjs, dim=0)
                    # all_train_recon = torch.cat(train_recon_adjs, dim=0)
                    # all_test_recon = torch.cat(test_recon_adjs, dim=0)
                    
                    # # 1. 计算原始训练集自身的MMD
                    # n_train = all_train_orig.size(0)
                    # split_idx = n_train // 2
                    # train_orig_metrics,_ = eval_torch_batch(
                    #     all_train_orig[:split_idx], 
                    #     all_train_orig[split_idx:],
                    #     methods=None
                    # )
                    # train_orig_metrics['avg'] = round(sum(train_orig_metrics.values())/3,6)
                    # print(f"epoch {epoch} train set self mmd: {train_orig_metrics}")
                    
                    # # 2. 计算原始测试集自身的MMD
                    # n_test = all_test_orig.size(0)
                    # split_idx = n_test // 2
                    # test_orig_metrics,_ = eval_torch_batch(
                    #     all_test_orig[:split_idx],
                    #     all_test_orig[split_idx:],
                    #     methods=None
                    # )
                    # test_orig_metrics['avg'] = round(sum(test_orig_metrics.values())/3,6)
                    # print(f"epoch {epoch} test set self mmd: {test_orig_metrics}")
                    
                    # # 3. 计算训练集重构MMD
                    # train_recon_metrics,_ = eval_torch_batch(all_train_orig, all_train_recon)
                    # train_recon_metrics['avg'] = round(sum(train_recon_metrics.values())/3,6)
                    # print(f"epoch {epoch} train reconstruction mmd: {train_recon_metrics}")
                    
                    # # 4. 计算测试集重构MMD
                    # test_recon_metrics,_ = eval_torch_batch(all_test_orig, all_test_recon)
                    # test_recon_metrics['avg'] = round(sum(test_recon_metrics.values())/3,6)
                    # print(f"epoch {epoch} test reconstruction mmd: {test_recon_metrics}")
                    
                    # if self.wandb is not None:
                    #     self.wandb.log({
                    #     # 原始训练集自身MMD
                    #     'train_self_mmd_degree': train_orig_metrics['degree'],
                    #     'train_self_mmd_cluster': train_orig_metrics['cluster'],
                    #     'train_self_mmd_orbit': train_orig_metrics['orbit'],
                    #     'train_self_mmd_avg': train_orig_metrics['avg'],
                        
                    #     # 原始测试集自身MMD
                    #     'test_self_mmd_degree': test_orig_metrics['degree'],
                    #     'test_self_mmd_cluster': test_orig_metrics['cluster'],
                    #     'test_self_mmd_orbit': test_orig_metrics['orbit'],
                    #     'test_self_mmd_avg': test_orig_metrics['avg'],
                        
                    #     # 训练集重构MMD
                    #     'train_recon_mmd_degree': train_recon_metrics['degree'],
                    #     'train_recon_mmd_cluster': train_recon_metrics['cluster'],
                    #     'train_recon_mmd_orbit': train_recon_metrics['orbit'],
                    #     'train_recon_mmd_avg': train_recon_metrics['avg'],
                        
                    #     # 测试集重构MMD
                    #     'test_recon_mmd_degree': test_recon_metrics['degree'],
                    #     'test_recon_mmd_cluster': test_recon_metrics['cluster'],
                    #     'test_recon_mmd_orbit': test_recon_metrics['orbit'],
                    #     'test_recon_mmd_avg': test_recon_metrics['avg']
                    # }, step=step)
                        
                logged_metrics = log_running_metrics(self.metrics, self.wandb, step, key='train')
                to_save = (self.encoder, self.decoder, self.quantizer, self.opt, self.scheduler)
                self.best_run = save_model(logged_metrics['recon_loss'], self.best_run, to_save, step,
                                           prefix='train_rec_loss', minimize=True, wandb=self.wandb, save_dir=f'./models_own/{self.config.exp_name}/{self.config.dataset}_autoencoder/files')
                logged_metrics = log_running_metrics(self.metrics, self.wandb, step, key='val', times=times)
                to_save = (self.encoder, self.decoder, self.quantizer, self.opt, self.scheduler)
                self.best_run = save_model(logged_metrics['recon_loss'], self.best_run, to_save, step,
                                           prefix='val_rec_loss', minimize=True, wandb=self.wandb, save_dir=f'./models_own/{self.config.exp_name}/{self.config.dataset}_autoencoder/files')

                if self.mol_data:
                    log_mol_metrics(annots, adjs, self.dataset, key='epoch')

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
        
        if self.init_steps > 0:
            zq = ze
            if self.quantizer.collect:
                collect = self.quantizer.collect_samples(zq.reshape(zq.shape[0], self.quantizer.nc,
                                                                 self.quantizer.embedding_dim).detach())
            else:
                collect = True
        else:
            if self.quantization:
                zq, commit, codebook, perplex, indices = self.quantizer(ze)
            else:
                zq = ze
                codebook, commit = torch.zeros(1).to(self.device), torch.zeros(1).to(self.device)
                perplex = torch.zeros(1).to(self.device)

        zq, node_masks = to_dense_batch(zq, batch.batch, max_num_nodes=self.max_node_num)

        masks = node_masks.detach(), get_edge_masks(node_masks)
        nodes_rec, edges_rec = self.decoder(zq, mask=node_masks)
        
        # Compute partial losses and reconstruction
        recon_loss, rec_losses = get_losses(batch, nodes_rec, edges_rec, node_masks.unsqueeze(-1), self.annotated_nodes,
                                            self.annotated_edges, self. max_node_num, self.n_node_feat, masks[1])
        if self.init_steps>0:
            loss = recon_loss
        else:
            loss = recon_loss + (codebook + commit) * self.gamma

        if train:
            loss.backward()
            self.opt.step()
            
        # Evaluate and log the results metrics, batch, rec_losses, vq_losses, masks, n_node_feat, train, annotated_nodes)
        if self.init_steps > 0:
            if not collect:
                self.init_steps = 0
        else:
            vq_losses = loss, recon_loss, codebook, commit, perplex
            nodes_rec, edges_rec = log_step_autoencoder(self.metrics, batch, rec_losses, vq_losses, masks,
                                            self.n_node_feat, train, self.annotated_nodes)

        # return nodes_rec, edges_rec.permute(0, 3, 1, 2)
        return nodes_rec, edges_rec, node_masks.unsqueeze(-1), masks[1] #返回node masks, edge masks 以供log mmd

    def prior(self) -> None:
        print(f'Train data contains {len(self.train_loader)} batches with size {self.train_loader.batch_size}')
        print('Train prior starts...')
        times = time.time(), time.process_time()
        step = 1
        ref = torch.zeros(0, 125, 125).to(self.device)
        for epoch in tqdm(range(1, self.epochs)):
            print(f'Train prior epoch {epoch} starts...')
            for batch in self.train_loader:
                # Train the prior
                self.fit_prior(batch.to(self.device))
                #if ref.shape[0] < 117:
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
                        gen_metrics = log_mmd_metrics(batch, adjs, key='prior train') #计算生成样本的mmd
                    print(f"train iter mmd: {gen_metrics}")
                    #save_prior(metrics, self.transformer, self.opt, self.scheduler, step, name='nspdk')
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
                annots, adjs = sample(self.n_samples, self.transformer, self.quantizer, self.decoder)
                if self.mol_data:
                    gen_metrics = log_mol_metrics(annots, adjs, self.dataset, key='prior val')
                    key = 'nspdk'
                else:
                    gen_metrics = log_mmd_metrics(batch, adjs, key='prior val')
                    key = 'avg'
                print(f'epoch {epoch} mmd: {gen_metrics}')

                # gen_metrics = log_mmd_metrics(batch, ref, key='trainset')
                # print(f'trainset mmd: {gen_metrics}')
                #ref = torch.zeros(0, 125, 125)

                to_save = (self.transformer, self.opt, self.scheduler)
                self.best_run = save_model(gen_metrics[key], self.best_run, to_save, step,
                                           prefix=f'epoch_{key}', minimize=True, wandb=self.wandb, prior=True, save_dir=f'./models_own/{self.config.exp_name}/{self.config.dataset}_prior')

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
        zq, indices = self.quantizer.encode(ze) #indices:正确索引，zq:量化后的节点特征
        zq, masks, indices = dense_zq(zq, indices, self.quantizer, batch, self.max_node_num, self.sort_indices)

        output = self.transformer(zq, mask=masks) #teacher forcing, auto regressive training
        # Apply masks to the output if needed
        if self.sort_indices:
            mask = get_output_mask(output.shape[0], indices, self.quantizer.n_embeddings, self.device, output.shape[2])
            output = output + mask

        # Calculate the loss
        loss = F.cross_entropy(output.permute(0, 3, 1, 2), indices)
        if train:
            loss.backward()
            self.opt.step()
        log_step_prior(self.metrics, loss, train)

        return zq, output

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
                annots, adjs = sample(self.n_samples, self.transformer, self.quantizer, self.decoder)
                print(time.time() - start)
                gen_mols, num_no_correct = gen_mol(annots.cpu(), adjs.cpu(), self.dataset)
                stop = time.time()
                print(stop - start)
                metrics = get_mol_metric(gen_mols, self.dataset, num_no_correct)
                print(metrics)

            else:
                print('Generation graph starts... ')
                for batch in self.test_loader:
                    annots, adjs = sample(self.n_samples, self.transformer, self.quantizer, self.decoder)
                    ref = to_dense_adj(batch.edge_index, batch=batch.batch, max_num_nodes=self.max_node_num)

                    # 收集训练数据测试baseline
                    adjs_trainset = torch.zeros(0, self.max_node_num, self.max_node_num)
                    for train_batch in self.train_loader:
                        adj = to_dense_adj(train_batch.edge_index, batch=train_batch.batch, max_num_nodes=self.max_node_num)
                        adjs_trainset = torch.cat((adjs_trainset, adj), dim=0)
                        if adjs_trainset.shape[0] >= 117:
                            break

                    adjs = adjs[:117] #取前117个
                    adjs_trainset = adjs_trainset[:117]
                    
                    metrics, metrics_novel_unique = eval_torch_batch(ref_batch=ref, pred_batch=adjs, methods=None)
                    metrics_trainset, metrics_novel_unique_trainset = eval_torch_batch(ref_batch=ref, pred_batch=adjs_trainset, methods=None)
                    plot_graphs(adjs[:20], max_plot=20, wandb=None, title='generated data', filepath=f"./plots/{self.config.exp_name}", filename=f"{self.config.dataset}_sample_graph")
                    plot_graphs(adjs_trainset[:20], max_plot=20, wandb=None, title='trainset data', filepath=f"./plots/{self.config.exp_name}", filename=f"{self.config.dataset}_trainset")
                    metrics['avg'] = round(sum(metrics.values()) / 3, 6) #计算平均值并保留6位小数
                    metrics_trainset['avg'] = round(sum(metrics_trainset.values()) / 3, 6) #计算平均值并保留6位小数
                    print(f"sample: {metrics}, {metrics_novel_unique}")
                    print(f"sample_trainset: {metrics_trainset}, {metrics_novel_unique_trainset}")
                    


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