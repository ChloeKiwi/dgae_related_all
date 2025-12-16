from utils.chem_func import mols_to_smiles
from graph_stats.stats import eval_graph_list
from utils.mol_utils import mols_to_nx, load_smiles, gen_mol
from utils.sample import sample
from utils.func import get_edge_target
from fcd_torch import FCD
import rdkit.Chem as Chem
from rdkit import rdBase
import pickle
import torch
from torch_geometric.utils import to_dense_batch
from torchmetrics import MeanMetric
import random

def init_autoencoder_running_metrics(annotated_nodes):
    metric_names = ['loss', 'edge_loss', 'edge_acc', 'edge_acc',
                    'graph_rec', 'commit', 'codebook', 'perplexity', 'recon_loss']
    if annotated_nodes:
        metric_names += ['node_loss', 'node_acc']

    metrics = {}
    metrics['train'] = {}
    metrics['iter'] = {}
    metrics['val'] = {}
    for metric_step in metrics:
        for metric in metric_names:
            metrics[metric_step][metric] = MeanMetric()
    return metrics

def init_prior_running_metrics():
    metric_names = ['loss']
    metrics = {}
    metrics['train'] = {}
    metrics['iter'] = {}
    metrics['val'] = {}
    for metric_step in metrics:
        for metric in metric_names:
            metrics[metric_step][metric] = MeanMetric()
    return metrics

def eval_sample(transformer, quantizer, decoder, dataset):
    print(dataset)
    if dataset == 'zinc' or dataset == 'qm9':
        N_SAMPLE = 1000
        annots, adjs = sample(N_SAMPLE, transformer, quantizer, decoder)
        gen_mols, num_no_correct = gen_mol(annots, adjs, dataset)
        metrics = get_mol_metric(gen_mols, dataset, num_no_correct)
        log_iter.log(metrics)
        log_iter.save_prior(metrics['nspdk'], transformer, opt, scheduler, step,
                                 name='nspdk')
    else:
        N_SAMPLE = 117
        annots, adjs = sample(N_SAMPLE, transformer, quantizer, decoder)
        ref = to_dense_adj(batch.edge_index, batch=batch.batch, max_num_nodes=max_node_num)
        metrics = eval_torch_batch(ref_batch=ref, pred_batch=adjs, methods=None)
        metrics['avg'] = sum(metrics.values()) / 3
        log_iter.log(metrics)
        log_iter.save_prior(metrics['avg'], transformer, opt, scheduler, step,
                                 name='Avg')


def reconstruction_stats(batch, edges_rec, nodes_rec, masks_nodes, masks_edges, n_node_feat):
    '''
    Args:
        edges_rec (tensor): The binary dense tensors of the reconstructed edge types (batch size x n x n x #edge_types)
        edges_rec (tensor): The dense tensors of the true edge types (batch size x n x n)
        edges_rec (tensor): The binary dense tensor of the true edge types (n x n x #edge_types)
        masks (tensor, bool): The dense tensor masking where there is possible edges.
    Return:
        tensor: the number of edges correctly reconstructed by graph.
        tensor: the rate of reconstruction by graph.
        int: the number of graphs, for which the edges are reconstructed totally correctly
    '''

    edges_true = get_edge_target(batch)
    if nodes_rec is not None:
        max_node_num = nodes_rec.shape[1]
        dense_nodes, _ = to_dense_batch(batch.x, batch.batch, max_num_nodes=max_node_num)
        nodes_true = dense_nodes[:, :, :n_node_feat].argmax(-1)
        nodes_rec = nodes_rec.argmax(-1)
        nodes_rec[(1 - masks_nodes.int()).bool()] = 0
        correct_nodes = nodes_rec == nodes_true
        n_nodes_corr = correct_nodes.sum(-1)
        all_nodes_corr = n_nodes_corr == nodes_rec.shape[-1]

        n_potential_nodes = nodes_rec.shape[0]*nodes_rec.shape[1]
        n_nodes = masks_nodes.sum()

        err_nodes = n_potential_nodes - n_nodes_corr.sum()
        acc_nodes = 1-err_nodes/n_nodes

    if edges_rec.shape[-1] > 1:
        edges_rec = edges_rec.argmax(-1)
    else:
        edges_rec = edges_rec.round()
    edges_rec[(1-masks_edges).squeeze().bool()] = 0

    edge_rec = edges_rec.squeeze() == edges_true

    n_edges_corr = edge_rec.sum([1, 2])
    n_potential_edges = edge_rec.shape[0] * edges_rec.shape[1] * edges_rec.shape[2]
    n_edges = masks_edges.sum()
    err_edges = n_potential_edges - n_edges_corr.sum()

    acc_edges = 1 - err_edges / n_edges

    all_edges_corr = (n_edges_corr == edges_rec.shape[1] ** 2)
    if nodes_rec is not None:
        graphs_corr = (all_nodes_corr*all_edges_corr)
    else:
        graphs_corr = all_edges_corr.sum()
        err_nodes, acc_nodes = None, None
    n_graphs_corr = graphs_corr.sum()
    return err_edges, acc_edges, err_nodes, acc_nodes, n_graphs_corr, graphs_corr

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def fraction_unique(gen, k=None, check_validity=True):
    """
    Computes a number of unique molecules
    Parameters:
        gen: list of SMILES
        k: compute unique@k
        n_jobs: number of threads for calculation
        check_validity: raises ValueError if invalid molecules are present
    """
    if k is not None:
        if len(gen) < k:
            raise ValueError(f"Can't compute unique@{k} gen contains only {len(gen)} molecules")
        gen = gen[:k]
    canonic = set(map(canonic_smiles, gen))
    if None in canonic and check_validity:
        raise ValueError("Invalid molecule passed to unique@k")
    return len(canonic) / len(gen)

def remove_invalid(gen, canonize=True):
    """
    Removes invalid molecules from the dataset
    """

    if not canonize:
        mols = get_mol(gen)
        return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    return [x for x in map(canonic_smiles, gen) if x is not None]

def fraction_valid(gen):
    """
    Computes a number of valid molecules
    Parameters:
        gen: list of SMILES
    """
    gen = [mol for mol in map(get_mol, gen)]
    return 1 - gen.count(None) / len(gen)

def novelty(gen, train):
    gen_smiles = []
    for smiles in gen:
        gen_smiles.append(canonic_smiles(smiles))
    gen_smiles_set = set(gen_smiles) - {None}
    train_set = set(train)
    return len(gen_smiles_set - train_set) / len(gen_smiles_set)

def get_mol_metric(gen_mols, dataset, num_no_correct, train_smiles=None):
    '''
    Args:
        - graphs(list of torch_geometric.Data)
        - train_smiles (list of smiles from the training set)
    Return:
        - Dict with key valid, unique, novel nspdk
    '''
    metrics = {}
    rdBase.DisableLog('rdApp.*')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_smiles, test_smiles = load_smiles(dataset=dataset)
    gen_smiles = mols_to_smiles(gen_mols)
    metrics['valid'] = num_no_correct
    gen_valid = remove_invalid(gen_smiles)
    metrics['unique'] = fraction_unique(gen_valid, k=None, check_validity=True)
    if train_smiles is not None:
        metrics['novel'] = novelty(gen_valid, train_smiles)
    else:
        metrics['novel'] = None

    with open(f'./data/{dataset.lower()}_test_nx.pkl', 'rb') as f:
        test_graph_list = pickle.load(f)
        random.Random(42).shuffle(test_graph_list)
    metrics['nspdk'] = eval_graph_list(test_graph_list[:1000], mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']
    # TODO: Make the computation of fcd more efficient
    # print(gen_smiles)
    metrics['fcd'] = FCD(n_jobs=0, device=device)(ref=test_smiles, gen=gen_smiles)
    metrics['valid_with_corr'] = len(gen_valid)
    return metrics

