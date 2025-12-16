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
        self.nc = config.model.quantizer.nc
        #self.quantization = not config.model.quantizer.turn_off
        self.quantization = True
        if config.train_prior:
            self.sort_indices = utils.func.sort_indices

        # Define Logger
        if config.train_prior:
            self.metrics = init_prior_running_metrics()
        else:
            self.metrics = init_autoencoder_running_metrics(self.annotated_nodes)

        ### wandb init ###
        if config.work_type != 'sample':
            project = f'VQ-GAE_{config.dataset}'
            run_id = f'{config.exp_name}_{config.run_name}_{config.work_type}'
            wandb.init(project=project, config=config, mode=config.log.wandb, 
                       name=run_id, id=run_id, resume=False)
            # wandb.save("*.pt") #todo: 先不在wandb保存pt
            # config.wandb_dir = os.path.basename(os.path.dirname(wandb.run.dir)) # 获取wandb_run_id
            
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
            # annots_recon = discretize(nodes_rec, masks=masks)
            none_type = 1 - masks.float()
            # annots_recon = torch.cat((annots_recon, none_type), dim=-1).detach().cpu()
            adjs_recon = adjs_recon.permute(0, 3, 1, 2).detach().cpu()
        return adjs_recon
    
    def compute_recon_mmd(self):
        print("Autoencoder has trained, start to compute mmd for reconstructed graph...")
        self.encoder.eval()
        self.decoder.eval()
        self.quantizer.eval()
        n_runs = 5
        base_seed = 42
        all_metrics = []
        for run in range(n_runs):
            # 为每次运行设置不同的随机种子
            current_seed = base_seed + run
            torch.manual_seed(current_seed)
            torch.cuda.manual_seed(current_seed)
            np.random.seed(current_seed)
            print(f'Running reconstruction iteration {run+1}/{n_runs} with seed {current_seed}')
            
            # 收集所有测试集的重构结果
            all_true_adjs = []
            all_recon_adjs = []
            with torch.no_grad():
                for batch in tqdm(self.test_loader, desc='Computing Recon MMD'):
                    batch = batch.to(self.device)
                    # 获取重构结果
                    annots, adjs, node_masks, edge_masks = self.fit_autoencoder(batch, train=False)
                    # 获取真实图的邻接矩阵
                    true_adj = to_dense_adj(batch.edge_index, batch=batch.batch, 
                                        max_num_nodes=self.max_node_num)
                    # 获取重构图的邻接矩阵
                    recon_adj = self.recon_adj_from_decoder(annots, adjs, 
                                                        node_masks.to(self.device), 
                                                        edge_masks.to(self.device))
                    
                    all_true_adjs.append(true_adj)
                    all_recon_adjs.append(recon_adj)
                
                # 合并所有batch的结果
                all_true_adjs = torch.cat(all_true_adjs, dim=0)
                all_recon_adjs = torch.cat(all_recon_adjs, dim=0)
            
                # 计算MMD指标
                metrics, metrics_novel_unique = eval_torch_batch(
                    ref_batch=all_true_adjs, 
                    pred_batch=all_recon_adjs,
                    methods=None
                )
                
                # 可选：绘制一些重构图的示例
                if run==0:
                    plot_graphs(all_recon_adjs[:20], max_plot=20, 
                            wandb=self.wandb, 
                            title='Final reconstruction samples',
                            filepath=f"./plots/{self.config.exp_name}/{self.config.run_name}",
                            filename=f"{self.config.dataset}_recon_run0")
                
                metrics['avg'] = round(sum(metrics.values()) / 3, 6)
                all_metrics.append(metrics)
                print(f"Run {run+1} (seed {current_seed}) recon:", metrics)
        
         # 计算平均指标和标准差
        avg_metrics = {}
        std_metrics = {}
        
        # 计算所有指标的平均值和标准差
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = round(np.mean(values), 6)
            std_metrics[key] = round(np.std(values), 6)
        
        print("\nAverage over 5 runs:")
        print("Recon average metrics:", avg_metrics)
        print("Recon std metrics:", std_metrics)
        
         # 将结果写入文件
        result_path = f'./models_own/{self.config.exp_name}/{self.config.run_name}/{self.config.dataset}_sample/recon_metrics.txt'
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Recon Results for {self.config.dataset}\n")
            
            # 合并平均值和标准差为 value±std 格式
            metrics_with_std = {}
            for key in avg_metrics.keys():
                metrics_with_std[key] = f"{avg_metrics[key]:.6f}±{std_metrics[key]:.6f}"
            
            f.write(f"Recon metrics (mean±std):\n")
            for key, value in metrics_with_std.items():
                f.write(f"  {key}: {value}\n")
            
            f.write(f"\nIndividual run results:\n")
            for i, m in enumerate(all_metrics):
                seed = base_seed + i
                f.write(f"Run {i+1} (seed {seed}):\n")
                f.write(f"  Recon: {m}\n")

    def node_visualize(self):
        print("Visualizing node embedding...")
        self.encoder.eval()
        self.quantizer.eval()
        
        # 收集所有测试集的node embeddings和labels
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Collecting embeddings'):
                batch = batch.to(self.device)
                ze, _ = self.encoder(batch)
                if self.quantization:
                    zq, _, _, _, _= self.quantizer(ze)
                else:
                    zq = ze
                all_embeddings.append(zq.cpu())
                all_labels.append(batch.label.cpu())
        
        # 合并所有batch的结果
        all_embeddings = torch.cat(all_embeddings, dim=0) #n_node, dim
        all_labels = torch.cat(all_labels, dim=0) #n_node
        
         # 使用t-SNE降维
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(all_embeddings.numpy()) # n_node, 2
        
        # 绘制散点图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=all_labels.numpy(), cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('t-SNE visualization of node embeddings')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        
        # 保存图片
        save_path = f"./plots/{self.config.exp_name}/{self.config.run_name}"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{self.config.dataset}_tsne.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.wandb:
            self.wandb.log({"node_embeddings_tsne": wandb.Image(f"{save_path}/{self.config.dataset}_tsne.png")})

    ### train autoencoder ###
    def autoencoder(self) -> None:
        print(f'Train data contains {len(self.train_loader)} batches with size {self.train_loader.batch_size}')
        times = time.time(), time.process_time()
        step = 0
        if self.init_steps > 0:
            print('Initialize autoencoder starts...init epochs: ',self.init_steps)
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
            for batch in self.train_loader:
                # Train the autoencoder
                self.fit_autoencoder(batch.to(self.device))
                step += 1
                if step % self.n_logging_steps == 0:
                    log_running_metrics(self.metrics, self.wandb, step, key='iter', times=times)


            # TEST
            for batch in self.test_loader:
                with torch.no_grad():
                    annots, adjs, node_masks, edge_masks = self.fit_autoencoder(batch.to(self.device), train=False)

            if epoch % self.n_logging_epochs == 0:
                logged_metrics = log_running_metrics(self.metrics, self.wandb, step, key='train')
                to_save = (self.encoder, self.decoder, self.quantizer, self.opt, self.scheduler)
                self.best_run = save_model(logged_metrics['recon_loss'], self.best_run, to_save, step,
                                        prefix='train_rec_loss', minimize=True, wandb=self.wandb, save_dir=f'./models_own/{self.config.exp_name}/{self.config.run_name}/{self.config.dataset}_autoencoder')
                logged_metrics = log_running_metrics(self.metrics, self.wandb, step, key='val', times=times)
                to_save = (self.encoder, self.decoder, self.quantizer, self.opt, self.scheduler)
                self.best_run = save_model(logged_metrics['recon_loss'], self.best_run, to_save, step,
                                        prefix='val_rec_loss', minimize=True, wandb=self.wandb, save_dir=f'./models_own/{self.config.exp_name}/{self.config.run_name}/{self.config.dataset}_autoencoder')

                if self.mol_data:
                    log_mol_metrics(annots, adjs, self.dataset, key='epoch') #adjs:bs,4,9,9
                # else: #! 是否计算重构mmd、绘制重构图
                    # metrics,_ = log_mmd_metrics(batch, adjs_recon, key='ae val')
                #     # print(f"reconstruction mmd at val epoch {epoch}: {metrics}")
                    
                #     adjs_recon = self.recon_adj_from_decoder(annots, adjs, node_masks.to(self.device), edge_masks.to(self.device))
                #     plot_graphs(adjs_recon[:20], max_plot=20, wandb=self.wandb, title='reconstruction data', filepath=f"./plots/{self.config.exp_name}", filename=f"{self.config.dataset}_recon_graph_epoch_{epoch}")


            if step % self.decay_iteration == 0 and self.scheduler.get_last_lr()[0] > 2 * 1e-5:
                self.scheduler.step()
                print('New learning rate: ', self.scheduler.get_last_lr())

        ### 重构mmd ###
        self.compute_recon_mmd()
        # 可视化node embedding
        # self.node_visualize()
            
    def fit_autoencoder(self, batch: Data, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        if train:
            self.encoder.train()
            self.quantizer.train()
            self.decoder.train()
            self.opt.zero_grad()
        else:
            self.encoder.eval()
            self.quantizer.eval()
            self.decoder.eval()
        
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
                perplex = round(perplex.item()/self.cb_size**self.nc, 3)
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

        # return nodes_rec, edges_rec.permute(0, 3, 1, 2), node_masks.unsqueeze(-1), masks[1]
        return nodes_rec, edges_rec, node_masks.unsqueeze(-1), masks[1]

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
                    annots, adjs,_ = sample_batch(self.n_samples, self.transformer, self.quantizer, self.decoder)

                    if self.mol_data:
                        gen_metrics = log_mol_metrics(annots, adjs, self.dataset, key='iter')
                        key = 'nspdk'
                    else:
                        gen_metrics = log_mmd_metrics(batch, adjs, key='iter')
                        key = 'avg'
                        
                    to_save = (self.transformer, self.opt, self.scheduler)
                    self.best_run = save_model(gen_metrics[key], self.best_run, to_save, step,
                                           prefix=f'iter_{key}', minimize=True, wandb=self.wandb, 
                                           prior=True, save_dir=f'./models_own/{self.config.exp_name}/{self.config.run_name}/{self.config.dataset}_prior')
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
                annots, adjs, indices = sample(self.n_samples, self.transformer, self.quantizer, self.decoder)
                if self.mol_data:
                    gen_metrics = log_mol_metrics(annots, adjs, self.dataset, key='epoch')
                    key = 'nspdk'
                else:
                    gen_metrics = log_mmd_metrics(batch, adjs, key='epoch')
                    key = 'avg'
                print(f'epoch {epoch} mmd: {gen_metrics}')

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
        zq, indices = self.quantizer.encode(ze)
        zq, masks, indices = dense_zq(zq, indices, self.quantizer, batch, self.max_node_num, self.sort_indices)
        # Prepare indices and node masks
        # import ipdb; ipdb.set_trace()
        output = self.transformer(zq, mask=masks)
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
                n_runs = 5
                base_seed = 42
                all_metrics = []
                # all_sample_time = []
                for run in range(n_runs):
                    start_time = time.time()
                    start = time.time()
                    annots, adjs,indices = sample(self.n_samples, self.transformer, self.quantizer, self.decoder)
                    sample_time = time.time() - start
                    print(f"run {run} sample total time: {sample_time}")
                    # all_sample_time.append(sample_time)
                    gen_mols, num_no_correct = gen_mol(annots.cpu(), adjs.cpu(), self.dataset)
                    stop = time.time()
                    print(f"run {run} gen mol total time: {stop - start}")
                    metrics = get_mol_metric(gen_mols, self.dataset, num_no_correct)
                    print(metrics)
                    all_metrics.append(metrics)
                
                avg_metrics = {}
                std_metrics = {}
                # 计算所有指标的平均值和标准差
                for key in all_metrics[0].keys():
                    values = [m[key] for m in all_metrics]
                    avg_metrics[key] = round(np.mean(values), 6)
                    std_metrics[key] = round(np.std(values), 6)
                
                print("\nAverage over 5 runs:")
                print("Sample average metrics:", avg_metrics)
                print("Sample std metrics:", std_metrics)
                
                
            else:
                print('Generation graph starts... ')
                self.n_samples = 1000
                # self.n_samples = 117
                n_runs = 5 
                base_seed = 42
                all_metrics = []
                all_metrics_trainset = []
                all_metrics_novel = []
                all_metrics_novel_trainset = []
                
                with torch.cuda.amp.autocast():  # 使用混合精度
                    for run in range(n_runs):
                        # 为每次运行设置不同的随机种子
                        current_seed = base_seed + run
                        torch.manual_seed(current_seed)
                        torch.cuda.manual_seed(current_seed)
                        np.random.seed(current_seed)
                        print(f'Running sample iteration {run+1}/{n_runs} with seed {current_seed}')
                                        
                        for batch in tqdm(self.test_loader, desc='Computing Sample MMD'):
                            annots, adjs, indices = sample(self.n_samples, self.transformer, self.quantizer, self.decoder)
                            ref = to_dense_adj(batch.edge_index, batch=batch.batch, max_num_nodes=self.max_node_num)

                            # 收集训练数据测试baseline
                            adjs_trainset = torch.zeros(0, self.max_node_num, self.max_node_num)
                            for train_batch in self.train_loader:
                                adj = to_dense_adj(train_batch.edge_index, batch=train_batch.batch, max_num_nodes=self.max_node_num)
                                adjs_trainset = torch.cat((adjs_trainset, adj), dim=0)
                                if adjs_trainset.shape[0] >= self.n_samples:
                                    break

                            adjs = adjs[:self.n_samples]
                            adjs_trainset = adjs_trainset[:self.n_samples]
                                            
                            metrics, metrics_novel_unique = eval_torch_batch(ref_batch=ref, pred_batch=adjs, methods=None)
                            metrics_trainset, metrics_novel_unique_trainset = eval_torch_batch(ref_batch=ref, pred_batch=adjs_trainset, methods=None)
                            # 只在第一次运行时绘图
                            if run == 0:
                                plot_graphs(adjs[:20], max_plot=20, wandb=None, title='generated data', 
                                        filepath=f"./plots/{self.config.exp_name}/{self.config.run_name}", 
                                        filename=f"{self.config.dataset}_sample_graph", indices=indices)
                                plot_graphs(adjs_trainset[:20], max_plot=20, wandb=None, title='trainset data', 
                                        filepath=f"./plots/{self.config.exp_name}/{self.config.run_name}", 
                                        filename=f"{self.config.dataset}_trainset")
                            metrics['avg'] = round(sum(metrics.values()) / 3, 6) #计算平均值并保留6位小数
                            metrics_trainset['avg'] = round(sum(metrics_trainset.values()) / 3, 6) #计算平均值并保留6位小数
                            all_metrics.append(metrics)
                            all_metrics_trainset.append(metrics_trainset)
                            all_metrics_novel.append(metrics_novel_unique)
                            all_metrics_novel_trainset.append(metrics_novel_unique_trainset)
                                            
                            print(f"Run {run+1} (seed {current_seed}) sample:", metrics)
                            print(f"Run {run+1} (seed {current_seed}) sample novel:", metrics_novel_unique)
                            print(f"Run {run+1} (seed {current_seed}) trainset:", metrics_trainset)
                            print(f"Run {run+1} (seed {current_seed}) trainset novel:", metrics_novel_unique_trainset)
                    
                    
def compute_avg_write_files(all_metrics,all_metrics_trainset,all_metrics_novel):
    # 计算平均指标
    avg_metrics = {}
    avg_metrics_trainset = {}
    avg_metrics_novel = {}
    avg_metrics_novel_trainset = {}
    std_metrics = {}
    std_metrics_trainset = {}
    std_metrics_novel = {}
    std_metrics_novel_trainset = {}
                    
    # 计算普通指标的平均值和标准差
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        values_trainset = [m[key] for m in all_metrics_trainset]
        avg_metrics[key] = round(np.mean(values), 6)
        avg_metrics_trainset[key] = round(np.mean(values_trainset), 6)
        std_metrics[key] = round(np.std(values), 6)
        std_metrics_trainset[key] = round(np.std(values_trainset), 6)

    # 分别计算uniqueness和novelty的平均值和标准差
    for key in ['uniqueness', 'novelty']:
        values_novel = [m[key] for m in all_metrics_novel]
        values_novel_trainset = [m[key] for m in all_metrics_novel_trainset]
        avg_metrics_novel[key] = round(np.mean(values_novel), 6)
        avg_metrics_novel_trainset[key] = round(np.mean(values_novel_trainset), 6)
        std_metrics_novel[key] = round(np.std(values_novel), 6)
        std_metrics_novel_trainset[key] = round(np.std(values_novel_trainset), 6)
                    
    print("\nAverage over 5 runs:")
    print("Sample average metrics:", avg_metrics)
    print("Sample std metrics:", std_metrics)
    print("Sample novel metrics:", avg_metrics_novel)
    print("Sample novel std metrics:", std_metrics_novel)
    print("Trainset average metrics:", avg_metrics_trainset)
    print("Trainset std metrics:", std_metrics_trainset)
    print("Trainset novel metrics:", avg_metrics_novel_trainset)
    print("Trainset novel std metrics:", std_metrics_novel_trainset)
                    
                    
    # 将结果写入文件
    result_path = f'./models_own/{self.config.exp_name}/{self.config.run_name}/{self.config.dataset}_sample/metrics.txt'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
                    
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Results for {self.config.dataset}\n")
                        
        # 合并平均值和标准差为 value±std 格式
        metrics_with_std = {}
        trainset_metrics_with_std = {}
        novel_metrics_with_std = {}
        novel_trainset_metrics_with_std = {}
        
        for key in avg_metrics.keys():
            metrics_with_std[key] = f"{avg_metrics[key]:.6f}±{std_metrics[key]:.6f}"
            trainset_metrics_with_std[key] = f"{avg_metrics_trainset[key]:.6f}±{std_metrics_trainset[key]:.6f}"
        
        for key in ['uniqueness', 'novelty']:
            novel_metrics_with_std[key] = f"{avg_metrics_novel[key]:.6f}±{std_metrics_novel[key]:.6f}"
            novel_trainset_metrics_with_std[key] = f"{avg_metrics_novel_trainset[key]:.6f}±{std_metrics_novel_trainset[key]:.6f}"
                        
        f.write(f"Sample metrics (mean±std):\n")
        for key, value in metrics_with_std.items():
            f.write(f"  {key}: {value}\n")
        
        f.write(f"\nSample novel metrics (mean±std):\n")
        for key, value in novel_metrics_with_std.items():
            f.write(f"  {key}: {value}\n")
                            
        f.write(f"\nTrainset metrics (mean±std):\n")
        for key, value in trainset_metrics_with_std.items():
            f.write(f"  {key}: {value}\n")
        
        f.write(f"\nTrainset novel metrics (mean±std):\n")
        for key, value in novel_trainset_metrics_with_std.items():
            f.write(f"  {key}: {value}\n")
                        
        f.write(f"\nIndividual run results:\n")
        for i, (m1, m2, m3, m4) in enumerate(zip(all_metrics, all_metrics_trainset, all_metrics_novel, all_metrics_novel_trainset)):
            seed = base_seed + i
            f.write(f"Run {i+1} (seed {seed}):\n")
            f.write(f"  Sample: {m1}\n")
            f.write(f"  Sample novel: {m3}\n")
            f.write(f"  Trainset: {m2}\n")
            f.write(f"  Trainset novel: {m4}\n")
                                                


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