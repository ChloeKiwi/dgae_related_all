import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import torch
from operator import itemgetter
from datetime import datetime
from time import perf_counter

from args_parse import parse_args as get_args
from task.utils.dataset import Dataset, collate
from task.utils.utils import load_graph, split_ids

from task.aggregation.gcn import run as aggregation
from trainer import Trainer
from data.dataset import get_dataset, get_benchmark_dataset
import yaml
from easydict import EasyDict as edict
from torch_geometric.utils import to_dense_batch, to_dense_adj
from config.config import get_config, get_prior_config, get_sample_config


def evaluate(args, train_loader, val_loader, test_loader):
    """
    Evaluate the performance of GNNs on the given dataset
    Args:
        args: arguments
        train_set: training set
        val_set: validation set
        test_set: test set
    Returns:
        acc_mic: micro-F1 score
        acc_mac: macro-F1 score
    """
    acc_mic = np.zeros(len(args.model_list)) #each model's micro-F1 score
    acc_mac = np.zeros(len(args.model_list)) #each model's macro-F1 score

    # params = {'batch_size': args.batch_size,
    #           'num_workers': args.num_workers,
    #           'prefetch_factor': args.prefetch_factor,
    #           'collate_fn': collate,
    #           'shuffle': True,
    #           'drop_last': True}
    # train_loader = torch.utils.data.DataLoader(train_set, **params)
    # val_loader = torch.utils.data.DataLoader(val_set, **params)
    # test_loader = torch.utils.data.DataLoader(test_set, **params)

    for i, model_name in enumerate(args.model_list):
        if args.task_name == "aggregation":
            # import ipdb; ipdb.set_trace()
            acc_mic[i], acc_mac[i] = aggregation(args, model_name, train_loader, val_loader, test_loader)
    return acc_mic, acc_mac

def main():
    args = get_args()
    # args.gpt_train_name = args.task_name + '_' + args.dataset + datetime.now().strftime("_%Y%m%d_%H%M%S")
    # Load the original graph datasets
    # import ipdb; ipdb.set_trace()
    # adj, feat, label, feat_size, label_size = load_graph(args)
    
    args.dataset = 'ego' #! 指定数据集  
    args.exp_name = 'baseline-cb8_2-mlm'
    config = get_sample_config(args) #!准备好某exp训练好的autoencoder+prior文件
    config.sample = True
    config.dataset = args.dataset
    args.max_node_num = config.data.max_node_num

    print(f"exp_name: {config.exp_name}") #从prior config.yaml中读取    
    print(f"dataset: {args.dataset}")
    loaders, config, data_info = get_dataset(args.dataset, config)
    train_loader, test_loader = loaders
    
    args.feat_size = data_info.n_node_feat
    args.label_size = data_info.n_node_class
    args.device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")
    trials = 1 #实验次数
    acc_mic_list = np.zeros((2, len(args.model_list), trials)) #2: original, synthetic; len(args.model_list): model num; trials: experiment num
    acc_mac_list = np.zeros((2, len(args.model_list), trials))
    
    for t in range(trials):
        # import ipdb; ipdb.set_trace()
        # Prepare duplicate-encoded computation graphs
        # train_set = Dataset(args, "train", adj, feat, label, ids) #feat:3312,3703, adj:3312,3312
        # val_set = Dataset(args, "val", adj, feat, label, ids)
        # test_set = Dataset(args, "test", adj, feat, label, ids)
        # initial dataset
        start_time = perf_counter()
        acc_mic, acc_mac = evaluate(args, train_loader, test_loader, test_loader) #!原数据集性能
        acc_mic_list[0, : , t] = acc_mic
        acc_mac_list[0, : , t] = acc_mac
        print('Original evaluation time: {:.3f}, acc: {}'.format(perf_counter() - start_time, acc_mic)) 
        
        ## Train GPT to generate new graphs
        start_time = perf_counter()
        # gen_train_set, gen_val_set, gen_test_set = gpt.run(args, adj, feat, label, ids) #!生成图
        import ipdb; ipdb.set_trace()
        trainer = Trainer(loaders, config, data_info)
        sample_batch = trainer.sample()
        print('DGAE-MASK training/generation total time: {:.3f}'.format(perf_counter() - start_time))

        # 生成数据集
        loaders, config, data_info = get_benchmark_dataset(sample_batch, args.dataset, config)
        train_loader, test_loader = loaders
        
        ## Check GNN performance on the generated dataset
        start_time = perf_counter()
        import ipdb; ipdb.set_trace()
        acc_mic, acc_mac = evaluate(args, train_loader, test_loader, test_loader) #! 生成的数据集性能
        acc_mic_list[1, : , t] = acc_mic
        acc_mac_list[1, : , t] = acc_mac
        print('Synthetic evaluation time: {:.3f}, acc: {}'.format(perf_counter() - start_time, acc_mic))
    
    test_acc_avg = np.average(acc_mic_list, axis=2)
    test_acc_std = np.std(acc_mic_list, axis=2)

    print('Task: ' + args.task_name + ', Dataset: ' + args.dataset)
    for model_name in args.model_list:
        print(model_name, end=', ')
    print()
    for model_id in range(len(args.model_list)):
        print("ORG: {:.2f} {:.3f}, GEN: {:.2f} {:.3f}".format(test_acc_avg[0][model_id], test_acc_std[0][model_id],\
                                                            test_acc_avg[1][model_id], test_acc_std[1][model_id]))


if __name__ == "__main__":
    main()