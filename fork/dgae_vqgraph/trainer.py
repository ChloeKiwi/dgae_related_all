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
    
        model = init_model(config, data_info, self.device) #! 加载模型
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
            wandb.init(project=f'VQ-GAE_{config.dataset}_{config.work_type}', config=config, mode=config.log.wandb, name=f'{config.exp_name}')
            wandb.save("*.pt")
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
            for batch in self.train_loader:
                # Train the autoencoder
                self.fit_autoencoder(batch.to(self.device))
                step += 1
                if step % self.n_logging_steps == 0:
                    log_running_metrics(self.metrics, self.wandb, step, key='iter', times=times)

            # Test
            for batch in self.test_loader:
                with torch.no_grad():
                    annots, adjs = self.fit_autoencoder(batch.to(self.device), train=False) #return nodes_rec, edges_rec

            # Log autoencoder results
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
                collect = self.quantizer.collect_samples(zq.reshape(zq.shape[0], self.quantizer.embedding_dim).detach()) #TODO:这里已经删掉了nc
            else:
                collect = True
        else: 
            if self.quantization:
                zq, h_list, codebook, commit, vq_indices, perplex = self.quantizer(batch, ze) #only quantizer, no decoder
            else:
                zq = ze 
                codebook, commit = torch.zeros(1).to(self.device), torch.zeros(1).to(self.device) #全0
                perplex = torch.zeros(1).to(self.device) #全0
        
        zq, node_masks = to_dense_batch(zq, batch.batch, max_num_nodes=self.max_node_num) #!True是valid node
        masks = node_masks.detach(), get_edge_masks(node_masks)
        # zq_adj = self.process_adj_batch(adj_quantized, batch, zq.size(0), self.max_node_num) #16*125*125 vqgraph返回了adj才能用
        
        quantized_node, quantized_edge = self.decoder(zq)
        edges_rec = torch.matmul(quantized_edge, quantized_edge.permute(0, 2, 1))
        edges_rec = (edges_rec - edges_rec.min()) / (edges_rec.max() - edges_rec.min())
        
        # reconstruct nodes and edges
        # nodes_rec, edges_rec = self.decoder(zq, mask=node_masks) #decoder:gnn, 输出：重构的node_feat, edge_feat
        nodes_rec, edges_rec = quantized_node, edges_rec #TODO: use nodes_rec ; nodes_rec require shape [batch_size, max_num_nodes, n_node_feat], edges_rec require shape [batch_size, max_num_nodes, max_num_nodes]

        # recon loss
        recon_loss, recon_info = get_losses(batch, nodes_rec, edges_rec, node_masks.unsqueeze(-1), self.annotated_nodes, 
                                            self.annotated_edges, self.max_node_num, self.n_node_feat, masks[1])
        
        #TODO: graph_layer_2 
        use_graph_layer_2 = False
        if use_graph_layer_2:
            batch = batch.to(quantized_edge.device)
            h_structure, _ = self.decoder.graph_layer_2(batch, quantized_edge) #origin g + recon g 
        
        else:
            h_structure = quantized_edge
        
        ce_loss = 0
        add_ce_loss = False
        if add_ce_loss and hasattr(self, 'node_classify') and self.node_classify:
            h_classify_node = self.decoder.node_classify_linear(h_structure)
            logits = h_classify_node.log_softmax(dim=-1)
            labels = batch.label
            ce_loss = F.cross_entropy(logits,labels)
            pred = logits.argmax(dim=-1)
            acc = (pred == labels).float().mean()
            print(f"node classify, ce_loss: {ce_loss}, acc: {acc}")
        elif hasattr(self, 'graph_classify') and self.graph_classify:
            h_pooled = global_mean_pool(h_structure)
            h_classify_graph = self.decoder.graph_classify_linear(h_pooled)
            logits = h_classify_graph.log_softmax(dim=-1)
            labels = batch.graph_label
            ce_loss = F.cross_entropy(logits,labels)
            pred = logits.argmax(dim=-1)
            acc = (pred == labels).float().mean()
            print("graph classify, ce_loss: {ce_loss}, acc: {acc}")
        
        if self.init_steps>0:
            # loss = recon_loss 
            loss = recon_loss + ce_loss
        else:
            # loss = recon_loss + (codebook + commit) * self.gamma 
            loss = recon_loss + (codebook + commit) * self.gamma + ce_loss
            
        if train:
            loss.backward()
            self.opt.step()
            
        # Evaluate and log the results metrics, batch, rec_losses, vq_losses, masks, n_node_feat, train, annotated_nodes)
        if self.init_steps > 0:
            if not collect:
                self.init_steps = 0
                print("stop collecting samples")
        else:
            vq_losses = loss, recon_loss, codebook, commit, perplex
            nodes_rec, edges_rec = log_step_autoencoder(self.metrics, batch, recon_info, vq_losses, masks,
                                            self.n_node_feat, train, self.annotated_nodes)

        if edges_rec.ndim==4:
            return nodes_rec, edges_rec.permute(0, 3, 1, 2)
        return nodes_rec, edges_rec
    
    def process_adj_batch(self, adj_quantized, batch, batch_size=16, max_node_num=125):
        # 重塑adj_quantized
        adj_quantized_list = [] 
        start_idx = 0 
        # 按照batch.ptr切分adj_quantized
        for i in range(batch_size):
            # 获取当前图的节点数
            if i == batch_size - 1:
                end_idx = adj_quantized.size(0)
            else:
                end_idx = batch.ptr[i + 1]
            # 切分当前图的邻接矩阵
            curr_adj = adj_quantized[start_idx:end_idx, start_idx:end_idx]
            # 填充到最大节点数
            padded_adj = torch.zeros(max_node_num, max_node_num, device=curr_adj.device)
            n_nodes = end_idx - start_idx
            padded_adj[:n_nodes, :n_nodes] = curr_adj
            
            adj_quantized_list.append(padded_adj)
            start_idx = end_idx

        # 堆叠成batch形式
        adj_quantized_dense = torch.stack(adj_quantized_list, dim=0)  # [batch_size, max_nodes, max_nodes]
        return adj_quantized_dense

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
                    annots, adjs = sample_batch(self.n_samples, self.transformer, self.quantizer, self.decoder) #TODO:bug可能在这里

                    if self.mol_data:
                        gen_metrics = log_mol_metrics(annots, adjs, self.dataset, key='iter')
                        key = 'nspdk'
                        to_save = (self.transformer, self.opt, self.scheduler)
                        self.best_run = save_model(gen_metrics[key], self.best_run, to_save, step,
                                                   prefix=f'iter_{key}', minimize=True, wandb=self.wandb, prior=True, save_dir=f'./models_own/{self.config.exp_name}/{self.config.dataset}_prior/files')

                    else:
                        gen_metrics = log_mmd_metrics(annots, adjs, key='iter')
                    print(gen_metrics)
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
                    gen_metrics = log_mol_metrics(annots, adjs, self.dataset, key='epoch')
                    key = 'nspdk'
                else:
                    gen_metrics = log_mmd_metrics(batch, adjs, key='epoch')
                    key = 'avg'
                print(gen_metrics)

                # ori_data_metrics = log_mmd_metrics(batch, ref, key='trainset')
                # print(ori_data_metrics)
                # ref = torch.zeros(0, 125, 125)

                to_save = (self.transformer, self.opt, self.scheduler)
                self.best_run = save_model(gen_metrics[key], self.best_run, to_save, step,
                                           prefix=f'epoch_{key}', minimize=True, wandb=self.wandb, prior=True, save_dir=f'./models_own/{self.config.exp_name}/{self.config.dataset}_prior/files')

            if step % self.decay_iteration == 0 and self.scheduler.get_last_lr()[0] > 2 * 1e-5:
                self.scheduler.step()
                print('New learning rate: ', self.scheduler.get_last_lr())



    def fit_prior(self, batch: Data, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        # Set model mode (train/eval) based on the "train" parameter
        if train:
            self.opt.zero_grad()
            self.transformer.train()
        else:
            self.transformer.eval() #prior时只训练transformer
        self.encoder.eval() 
        self.quantizer.eval()
        self.decoder.eval()

        # Encoding and quantization
        ze, w0 = self.encoder(batch)
        zq, indices, codebook = self.quantizer.encode(ze, batch) #TODO:?
        zq, masks, indices = dense_zq(zq, indices, self.quantizer, batch, self.max_node_num, self.sort_indices, codebook)

        # learn the distribution of vq_indices autoregressively
        output = self.transformer(zq, mask=masks)
        # Apply masks to the output if needed
        if self.sort_indices:  
            mask = get_output_mask(output.shape[0], indices, self.quantizer.n_embeddings, self.device, output.shape[2])
            output = output + mask

        # Calculate the loss
        loss = F.cross_entropy(output.permute(0, 3, 1, 2), indices) #output有4维度
        # loss = F.cross_entropy(output.permute(0,2,1), indices) #1d
        if train:
            loss.backward()
            self.opt.step()
        log_step_prior(self.metrics, loss, train)

        return zq, output

    def sample(self) -> None:
        with torch.no_grad():
            # import ipdb; ipdb.set_trace()
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
                    # import ipdb; ipdb.set_trace()
                    annots, adjs = sample(self.n_samples, self.transformer, self.quantizer, self.decoder)
                    
                    ref = to_dense_adj(batch.edge_index, batch=batch.batch, max_num_nodes=self.max_node_num) #也是117个
                    # adjs = torch.zeros(0, self.max_node_num, self.max_node_num) #初始化adjs,shape:0,125,125
                    
                    # for train_batch in self.train_loader:
                    #     adj = to_dense_adj(train_batch.edge_index, batch=train_batch.batch, max_num_nodes=self.max_node_num)
                    #     adjs = torch.cat((adjs, adj), dim=0) #生成的图+训练集的图
                    #     if adjs.shape[0] >= 117:
                    #         break

                    adjs = adjs[:117] #取前117个
                    metrics = eval_torch_batch(ref_batch=ref, pred_batch=adjs, methods=None)
                    plot_graphs(adjs[:20], max_plot=20, wandb=None, title=None, filename="enzy_sample.svg")
                    metrics['avg'] = round(sum(metrics.values()) / 3, 6) #计算平均值并保留6位小数
                    print(metrics)
    
    # def benchmark(self) -> None:
    #     # 1 原数据集在gnn上训练+验证+测试后的性能
    #     # 2 使用sample生成新数据集
    #     # 3 使用新数据集在gnn上训练+验证+测试后的性能
    #     # 4 比较新数据集和原数据集的性能
    #     # if train:
    #     #     self.encoder.train()
    #     #     self.quantizer.train()
    #     #     self.decoder.train() #decoder也需要train
    #     #     self.opt.zero_grad()
    #     # else:
    #     #     self.encoder.eval()
    #     #     self.quantizer.eval()
    #     #     self.decoder.eval()
        
    #     for epoch in range(1, self.epochs):
    #         for batch in self.train_loader:
    #             import ipdb; ipdb.set_trace()
    #             node_feat, _, cls_out = self.encoder(batch)
                
                
    #         for batch in self.test_loader:
    #             pass
            
    #     for batch in self.test_loader:
    #         with torch.no_grad():
    #             pass                


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

# 1d
def get_output_mask(bs: int, indices: torch.Tensor, cb_size: int, device: torch.device,
                    output_shape: int) -> torch.Tensor:
    """
    生成因果mask，确保每个位置只能看到之前的token
    Args:
        bs: batch size
        indices: token indices, shape [bs, seq_len]
        cb_size: codebook size
        device: torch device
    Returns:
        mask: shape [bs, seq_len, cb_size+1]
    """
    # 创建下三角矩阵，对角线以上为-inf
    tril = torch.tril(
        torch.full((cb_size + 1, cb_size + 1), float('-inf')), 
        diagonal=-1
    ).to(device)
    
    # 为每个位置获取前一个token的index
    idx = torch.cat(
        (torch.zeros(bs, 1).to(device).long(),  # 第一个位置用0填充
         indices[:, :-1]),                       # 其他位置用前一个token
        dim=1
    )
    
    # 根据前一个token的index获取对应的mask
    mask = tril[idx]  # shape: [bs, seq_len, cb_size+1]
    
    return mask