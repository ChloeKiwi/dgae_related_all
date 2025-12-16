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


class Trainer:
    def __init__(self, dataloaders, config, data_info):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Run training on : {self.device}')

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
        #self.quantization = not config.model.quantizer.turn_off
        self.quantization = True
        print(self.quantization)
        if config.train_prior:
            self.sort_indices = utils.func.sort_indices

        # Define Logger
        if config.train_prior:
            self.metrics = init_prior_running_metrics()
        else:
            self.metrics = init_autoencoder_running_metrics(self.annotated_nodes)

        if config.work_type != 'sample':
            wandb.init(project=f'VQ-GAE_{config.dataset}_{config.work_type}', config=config, mode=config.log.wandb)
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


    def autoencoder(self) -> None:
        print(f'Train data contains {len(self.train_loader)} batches with size {self.train_loader.batch_size}')
        times = time.time(), time.process_time()
        step = 0
        if self.init_steps > 0:

            print('Initialization starts...')
            self.quantizer.collect = False
            for epoch in range(1, self.epochs):
                for batch in self.train_loader:
                    # Train the autoencoder
                    self.fit_autoencoder(batch.to(self.device))
                    print(step)
                    step += 1
                    if step >= self.init_steps:
                        self.quantizer.collect = True

                    if self.init_steps == 0:
                        break
                if self.init_steps == 0:
                    break


        print('Training starts...')
        step = 0
        for epoch in range(1, self.epochs):
            for batch in self.train_loader:
                # Train the autoencoder
                self.fit_autoencoder(batch.to(self.device))
                step += 1
                if step % self.n_logging_steps == 0:
                    log_running_metrics(self.metrics, self.wandb, step, key='iter', times=times)


            # TEST
            for batch in self.test_loader:
                with torch.no_grad():
                    annots, adjs = self.fit_autoencoder(batch.to(self.device), train=False)

            # Log autoencoder results
            if epoch % self.n_logging_epochs == 0:
                logged_metrics = log_running_metrics(self.metrics, self.wandb, step, key='train')
                to_save = (self.encoder, self.decoder, self.quantizer, self.opt, self.scheduler)
                self.best_run = save_model(logged_metrics['recon_loss'], self.best_run, to_save, step,
                                           prefix='train_rec_loss', minimize=True, wandb=self.wandb)
                logged_metrics = log_running_metrics(self.metrics, self.wandb, step, key='val', times=times)
                to_save = (self.encoder, self.decoder, self.quantizer, self.opt, self.scheduler)
                self.best_run = save_model(logged_metrics['recon_loss'], self.best_run, to_save, step,
                                           prefix='val_rec_loss', minimize=True, wandb=self.wandb)

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

        # Encoding, quantization, and decoding
        ze, w0 = self.encoder(batch)

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

        return nodes_rec, edges_rec.permute(0, 3, 1, 2)

    def prior(self) -> None:
        print(f'Train data contains {len(self.train_loader)} batches with size {self.train_loader.batch_size}')
        print('Training starts...')
        times = time.time(), time.process_time()
        step = 1
        ref = torch.zeros(0, 125, 125).to(self.device)
        for epoch in range(1, self.epochs):
            print(f'Epoch {epoch} starts...')
            for batch in self.train_loader:
                # Train the prior
                self.fit_prior(batch.to(self.device))
                #if ref.shape[0] < 117:
                #    adj = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=self.max_node_num).to(self.device)
                #    ref = torch.cat((ref, adj), dim=0)
                if step % self.n_logging_steps == 0:
                    metrics = log_running_metrics(self.metrics, self.wandb, step, key='iter', times=times)
                    annots, adjs = sample_batch(self.n_samples, self.transformer, self.quantizer, self.decoder)

                    if self.mol_data:
                        gen_metrics = log_mol_metrics(annots, adjs, self.dataset, key='iter')
                        key = 'nspdk'
                        to_save = (self.transformer, self.opt, self.scheduler)
                        self.best_run = save_model(gen_metrics[key], self.best_run, to_save, step,
                                                   prefix=f'iter_{key}', minimize=True, wandb=self.wandb, prior=True)

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

                #gen_metrics = log_mmd_metrics(batch, ref, key='trainset')
                #print(gen_metrics)
                #ref = torch.zeros(0, 125, 125)

                to_save = (self.transformer, self.opt, self.scheduler)
                self.best_run = save_model(gen_metrics[key], self.best_run, to_save, step,
                                           prefix=f'epoch_{key}', minimize=True, wandb=self.wandb, prior=True)

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
                print('Generation starts... ')
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
                for batch in self.test_loader:
                    annots, adjs = sample(self.n_samples, self.transformer, self.quantizer, self.decoder)
                    ref = to_dense_adj(batch.edge_index, batch=batch.batch, max_num_nodes=self.max_node_num)

                    
                    adjs = torch.zeros(0, self.max_node_num, self.max_node_num)
                    for train_batch in self.train_loader:
                        adj = to_dense_adj(train_batch.edge_index, batch=train_batch.batch, max_num_nodes=self.max_node_num)
                        adjs = torch.cat((adjs, adj), dim=0)
                        if adjs.shape[0] >= 117:
                            break

                    adjs = adjs[:117]
                    metrics = eval_torch_batch(ref_batch=ref, pred_batch=adjs, methods=None)
                    plot_graphs(adjs[:20], max_plot=20, wandb=None, title=None)
                    metrics['avg'] = sum(metrics.values()) / 3
                    print(metrics)


def get_output_mask(bs: int, indices: torch.Tensor, cb_size: int, device: torch.device,
                    output_shape: int) -> torch.Tensor:

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
