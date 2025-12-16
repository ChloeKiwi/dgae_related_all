import time
import torch
import os
import wandb
from torch_geometric.utils import to_dense_adj
from graph_stats.stats import eval_torch_batch
from utils.eval import reconstruction_stats, get_mol_metric
from utils.mol_utils import gen_mol
from datetime import datetime

# def log_running_metrics(metrics, wandb, step, key, times=None):
#     metrics = metrics[key]
#     log_metrics = {name: metric.compute() for name, metric in metrics.items()}
#     if times is not None:
#         clock_time = time.time()-times[0]
#         process_time = time.process_time()-times[1]
#         log_metrics['clock_time'] = clock_time
#         log_metrics['process_time'] = process_time
#         print(f'Running metrics for {key} afer {step} steps/epochs and {clock_time} seconds')
#     else:
#         print(f'Running metrics for {key} afer {step} steps/epochs')
#     print(log_metrics)
#     for metric in metrics.values():
#         metric.reset()
#     if wandb is not None:
#         wandb.log({key: log_metrics}, step=step)
#     return log_metrics

def log_running_metrics(metrics, wandb, step, key, times=None):
    metrics = metrics[key]
    log_metrics = {f"{key}/{name}": value for name, value in 
                  {name: metric.compute() for name, metric in metrics.items()}.items()}
    
    if times is not None:
        clock_time = time.time()-times[0]
        process_time = time.process_time()-times[1]
        log_metrics[f'{key}/clock_time'] = clock_time
        log_metrics[f'{key}/process_time'] = process_time
        print(f'Running metrics for {key} after {step} steps/epochs and {clock_time} seconds')
    else:
        print(f'Running metrics for {key} after {step} steps/epochs')
    
    print("log_metrics:", log_metrics)
    for metric in metrics.values():
        metric.reset()
        
    if wandb is not None:
        wandb.log(log_metrics, step=step)  # 直接记录扁平化的指标
    return log_metrics

def log_step_autoencoder(metrics, batch, rec_losses, vq_losses, masks, n_node_feat, train, annotated_nodes,mol_data=True):
    train_metrics, iter_metrics, val_metrics = metrics.values()
    node_loss, edge_loss, nodes_rec, edges_rec = rec_losses
    loss, recon_loss, codebook, commit, perplex, codebook_usage = vq_losses
    node_masks, edge_masks = masks
    '''
        nc = indices.shape[-1]
        joint_indices = 0
        for i in range(nc):
            joint_indices += indices[:, i] * quantizer.n_embeddings ** (nc - (i + 1))
        unique = len(joint_indices.unique())
        dist = torch.bincount(joint_indices, minlength=quantizer.n_embeddings ** nc)/indices.shape[0]
        alpha = 0.1
        temp_logger['dist_ema'] = dist*alpha + (1-alpha)*temp_logger['dist_ema']
        temp_logger['n_vq_used'] = (((dist + temp_logger['dist_ema'])*1_000_000).round()>0).sum()
        H = torch.distributions.Categorical(probs=temp_logger['dist_ema']).entropy()
    '''
    stats = reconstruction_stats(batch, edges_rec, nodes_rec, node_masks, edge_masks, n_node_feat, mol_data)
    err_edges, edge_acc, err_nodes, node_acc, n_graph_rec, graph_rec = stats

    metric_values = (loss.item(), edge_loss.item(), edge_acc.item(), n_graph_rec.item(),
                     codebook.item(), commit.item(), recon_loss.item(),perplex.item(), codebook_usage)
    if annotated_nodes:
        metric_values += (node_loss.item(), node_acc.item())
        nodes_rec = nodes_rec.detach().cpu()
    edges_rec = edges_rec.detach().cpu()
    if train:
        for metric, values in zip(train_metrics.values(), metric_values):
            metric.update(values)
        for metric, values in zip(iter_metrics.values(), metric_values):
            metric.update(values)
    else:
        for metric, values in zip(val_metrics.values(), metric_values):
            metric.update(values)
    return nodes_rec, edges_rec

def log_step_prior(metrics, loss, train):
    if train:
        metrics['train']['loss'].update(loss.item())
        metrics['iter']['loss'].update(loss.item())
    else:
        metrics['val']['loss'].update(loss.item())

def save_model(metric, best_metrics, to_save, step, prefix, minimize=True, wandb=None, prior=False, save_dir='./models_own/no_name'):
    if prior:
        transformer, opt, scheduler = to_save
    else:
        encoder, decoder, quantizer, opt, scheduler = to_save
    if prefix not in best_metrics:
        best_metrics[prefix] = metric

    if minimize:
        condition = metric <= best_metrics[prefix] #没有= ego可能有问题
    else:
        condition = metric >= best_metrics[prefix]
    if condition:
        best_metrics[prefix] = metric

        if wandb is not None:
            save_dir = wandb.run.dir
        else:
            save_dir = save_dir
            is_folder = os.path.exists(save_dir)
            if not is_folder:
                # Create a new directory because it does not exist
                os.makedirs(save_dir)
        # save_dir = save_dir

        if prior:
            torch.save({
                'iteration': step,
                'transformer': transformer.state_dict(),
                'optimizer': opt.state_dict(),
                'scheduler': scheduler.state_dict(),
                'metric': best_metrics[prefix]},
                os.path.join(save_dir, f'best_run_{prefix}.pt'))
        else:
            torch.save({
                'iteration': step,
                'encoder': encoder.state_dict(),
                'quantizer': quantizer.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': opt.state_dict(),
                'scheduler': scheduler.state_dict(),
                'metric': best_metrics[prefix]},
                os.path.join(save_dir, f'best_run_{prefix}.pt'))

        if wandb is not None:
            wandb.save("*.pt")
    return best_metrics

def log_mol_metrics(annots, adjs, dataset, key=None):
    # import ipdb; ipdb.set_trace()
    gen_mols, num_no_correct = gen_mol(annots, adjs, dataset)
    from rdkit.Chem import Draw
    img = Draw.MolsToGridImage(gen_mols[:16])
    img.save(f'./plots/{dataset}_mol_eval_{key}.png')
    metrics = get_mol_metric(gen_mols, dataset, num_no_correct)
    if key is not None:
        wandb.log({f'Mol eval {key}': metrics})
    return metrics

def log_mmd_metrics(batch, adjs, key=None):
    # import ipdb; ipdb.set_trace()
    refs = to_dense_adj(batch.edge_index, batch.batch)
    #from utils.func import plot_graphs
    #plot_graphs(adjs, max_plot=16, wandb=None, title=None)
    metrics, metrics_novel_unique = eval_torch_batch(refs, adjs)
    metrics['avg'] = sum(metrics.values()) / 3
    # wandb.log({f'MMD eval {key}': metrics})
    if key is not None:
        wandb.log({f'MMD {key}': metrics})
    return metrics, metrics_novel_unique
