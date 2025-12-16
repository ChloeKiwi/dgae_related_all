import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import gc
import numpy as np
from tqdm import tqdm
from time import perf_counter, sleep

from ..utils.utils import calc_f1, calc_loss
from torch_geometric.utils import to_dense_batch, to_dense_adj

# GCN template
class GCN(nn.Module):
    """
    Graph Convolutional Networks (GCN)
    Args:
        model_name: model name ('gcn', 'gin', 'sgc', 'gat')
        input_dim: input dimension
        output_dim: output dimension
        hidden_dim: hidden dimension
        step_num: number of propagation steps
        output_layer: whether to use the output layer
    """

    def __init__(self, model_name, input_dim, output_dim, hidden_dim, step_num, output_layer=True, graph_task=False):
        super(GCN, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.step_num = step_num

        self.output_layer = output_layer
        
        self.graph_task = graph_task

        self.W =  nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=False)])
        for _ in range(step_num-1):
            self.W.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        for w in self.W:
            nn.init.xavier_uniform_(w.weight)
        self.outputW = nn.Linear(hidden_dim, output_dim, bias=False)
        nn.init.xavier_uniform_(self.outputW.weight)

        # 不同的模型有不同的pooling方式(gin,gat,sgc)
        self.pooling = 'sum' if model_name in ('gin', 'gat') else 'avg'
        self.nonlinear = (model_name != 'sgc')
        self.attention = (model_name == 'gat')
        if self.attention:
            self.attentionW = nn.Parameter(torch.empty(size=(2*hidden_dim, 1))) #parameter that initialized by 0
            nn.init.xavier_uniform_(self.attentionW.data) #initialize the parameter
            self.leakyReLU = nn.LeakyReLU(0.2)
            self.softmax = nn.Softmax(dim=1)

    def get_parameters(self):
        ml = list()
        for w in self.W:
            ml.append({'params': w.parameters()})
        ml.append({'params': self.outputW.parameters()})
        if self.attention:
            ml.append({'params': self.attentionW})
        return ml

    def forward(self, feat, raw_adj):
        """
        Args:
            feat: feature matrix (batch_num * node_num, input_dim)
            raw_adj: adjacency matrix (batch_num, node_num, node_num)
        Returns:
            output: feature matrix of target nodes (batch_num, output_dim)
        """
        feat = feat.float() #从float64转换为float32
        batch_num, batch_size = raw_adj.shape[0], raw_adj.shape[1]
        ids = torch.range(0, (batch_num - 1) * batch_size, batch_size, dtype=torch.long).to(raw_adj.device) #torch.tensor([0, 64, 128, ..., 1984])，each element is the start index of each batch
        adj = torch.block_diag(*raw_adj).to(raw_adj.device) #torch.Size([3648, 3648])
        if self.pooling == 'avg':
            adj = self.avg_pooling(feat, adj)
        X = feat
        for w in self.W:
            Z = w(X) #特征变换
            if self.attention:
                adj = self.compute_attention(adj, Z) #计算attention for each node
            X = torch.spmm(adj, Z) #邻接矩阵乘以特征矩阵，消息传递
            if self.nonlinear: #是否使用非线性激活函数
                X = F.relu(X)
        if self.output_layer: #是否使用输出层
            X = self.outputW(X) #X is the first embedding
        
        if self.graph_task:
            return X[ids] #return the first embedding of each batch
        else:
            return X

    def avg_pooling(self, feats, adj):
        """
        Args:
            feats: feature matrix (batch_num * node_num, input_dim)
            adj: adjacency matrix (batch_num * node_num, batch_num * node_num)
        Returns:
            adj: adjacency matrix (batch_num * node_num, batch_num * node_num)
        """
        # import ipdb; ipdb.set_trace()
        nonzeros = torch.nonzero(torch.norm(feats, dim=1), as_tuple=True)[0]
        nonzero_adj = adj[:, nonzeros] #adj的非零列（度数不为0的节点）
        row_sum = torch.sum(nonzero_adj, dim=1) #sum of each row
        row_sum = row_sum.masked_fill_(row_sum == 0, 1.) #if row_sum == 0, row_sum = 1，mask the row 
        row_sum = torch.diag(1/row_sum).to(adj.device) #计算每行的归一化系数
        adj = torch.spmm(row_sum, adj) #row_sum * adj 对每一行进行归一化
        return adj

    def compute_attention(self, adj, X):
        Wh1 = torch.matmul(X, self.attentionW[:self.hidden_dim, :]) # X*W1
        Wh2 = torch.matmul(X, self.attentionW[self.hidden_dim:, :]) # X*W2
        e = Wh1 + Wh2.T # X*W1 + X*W2.T
        e= self.leakyReLU(e) 

        zero_vec = -9e15 * torch.ones_like(e) #torch.tensor(-9e15).to(self.device)
        # import ipdb; ipdb.set_trace()
        attention = torch.where(adj > 0, e, zero_vec) #if adj > 0, attention = e, else attention = zero_vec
        attention = F.softmax(attention, dim=1) #softmax
        return attention


def run(args, model_name, train_loader, val_loader, test_loader):
    """
    Evaluate GNN performance
    Args:
        args: arguments
        model_name: model name ('gcn', 'gin', 'sgc', 'gat')
        train_loader: training data loader
        val_loader: validation data loader
        test_loader: test data loader
    Returns:
        acc_mic: micro-F1 score
        acc_mac: macro-F1 score
    """

    device = args.device
    model = GCN(model_name, args.feat_size, args.label_size, args.hidden_dim, args.step_num, graph_task=False)
    # model = nn.DataParallel(model).to(device) #!分布式
    model = model.to(device)

    # Test GCN models
    def test_model(args, model, data_loader, split='val'):
        start_time = perf_counter()
        stack_output = []
        stack_label = []
        model.eval()
        with tqdm(data_loader, unit="batch") as t_data_loader:
            for batch in t_data_loader:
                batch["adj"] = to_dense_adj(batch["edge_index"])
                feats, adjs, labels = batch["feature"].to(device), batch["adj"].to(device), batch["label"].to(device)
                outputs = model(feats, adjs)
                loss = calc_loss(outputs, labels)
                stack_output.append(outputs.detach().cpu())
                stack_label.append(labels.cpu())
                t_data_loader.set_description(f"{split}")
                t_data_loader.set_postfix(loss=loss.item())
                sleep(0.1)
        stack_output = torch.cat(stack_output, dim=0)
        stack_label = torch.cat(stack_label, dim=0)
        loss = calc_loss(stack_output, stack_label)
        acc_mic, acc_mac = calc_f1(stack_output, stack_label)
        return loss, acc_mic, acc_mac

    # 获取模型参数的正确方式
    ml = list()
    if hasattr(model, 'module'):  # 检查是否是DataParallel/DistributedDataParallel模型
        ml.extend(model.module.get_parameters())
    else:
        ml.extend(model.get_parameters())
    optimizer = optim.Adam(ml, lr=args.lr)

    patient = 0
    min_loss = np.inf
    for epoch in range(args.epochs):
        with tqdm(train_loader, unit="batch") as t_train_loader:
            for batch in t_train_loader:
                batch["adj"] = to_dense_adj(batch["edge_index"]) #1,num_nodes, num_nodes
                feats, adjs, labels = batch["feature"].to(device), batch["adj"].to(device), batch["label"].to(device)
                # feats:1984,3703 adjs:64,31,31 labels:64 ; 64个子图，每个子图31个节点，3703维特征, 64个标签
                model.train()
                optimizer.zero_grad()
                outputs = model(feats, adjs) #64,6
                loss = calc_loss(outputs, labels) #分类损失(仅train_loader计算)
                loss.backward() #反向传播
                optimizer.step() #更新参数

                t_train_loader.set_description(f"Epoch {epoch}")
                t_train_loader.set_postfix(loss=loss.item())
                sleep(0.1) 

        with torch.no_grad(): #t_train_loader全部完成后
            new_loss, acc_mic, acc_mac = test_model(args, model, val_loader, 'val')
            if new_loss >= min_loss: #loss无限大时
                patient = patient + 1
            else:
                min_loss = new_loss
                patient = 0

        if patient == args.early_stopping:
            break

    _, acc_mic, acc_mac = test_model(args, model, test_loader, 'test') #测试

    # import ipdb; ipdb.set_trace()
    del model #释放模型
    return acc_mic, acc_mac

