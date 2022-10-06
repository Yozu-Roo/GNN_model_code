
import torch
import torch.nn.functional as F

# The PyG built-in GCNConv
from torch_geometric.nn import GCNConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
import copy

class GCN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
        '''
        模型的初始化设置
        :param input_dim: int，输入维度，即节点的特征维度
        :param hidden_dim: int，隐藏层维度，自主设置
        :param output_dim: int，输出维度，节点所属的类别个数
        :param num_layers: int，层数
        :param dropout: float，若用到dropout方法，设置的随机暂退概率值
        :param return_embeds: 是否需要返回节点嵌入向量
        '''
        super(GCN, self).__init__()

        self.convs = None

        self.bns = None

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(in_channels=input_dim, out_channels=hidden_dim))
            input_dim = hidden_dim
        self.convs.append(GCNConv(in_channels=hidden_dim, out_channels=output_dim))

        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features=hidden_dim) for i in range(num_layers-1)])

        self.softmax = torch.nn.LogSoftmax()

        self.dropout = dropout
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        '''
        前向传播过程
        :param x: 节点的特征向量，Tensor
        :param adj_t: 邻接矩阵，或者传入为edge
        :return: 分类结果或者节点嵌入表示
        '''
        out = None
        for layer in range(len(self.convs)-1):
            x = self.convs[layer](x, adj_t)
            x = self.bns[layer](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, self.training)

        out = self.convs[-1](x, adj_t)
        if not self.return_embeds:
            out = self.softmax(out)

        return out


def train(model, data, train_idx, optimizer, loss_fn):
    '''
    训练过程
    :param model: 设置好的模型
    :param data: 节点特征向量
    :param train_idx: 划分为训练集的数据id
    :param optimizer: 优化器
    :param loss_fn: 损失函数
    :return: 训练一次后得到的损失值
    '''
    model.train()
    loss = 0

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    train_output = out[train_idx]
    train_label = data.y[train_idx, 0]
    loss = loss_fn(train_output, train_label)

    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, data, split_idx, evaluator):
    '''
    测试过程
    :param model: 设置好的模型
    :param data: 节点的特征向量
    :param split_idx: 划分的数据的id
    :param evaluator: 评估器
    :return: 返回在训练集、验证集和测试集上模型的准确率
    '''
    model.eval()

    out = None

    out = model(data.x, data.adj_t)

    y_pred = out.argmax(dim=-1, keepdim=True)  # 将嵌入向量转换为对应的预测结果，并对维度进行调整

    #  使用评估器来进行acc计算
    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

args = {
    'num_layers': 3,
    'hidden_dim': 256,
    'dropout': 0.5,
    'lr': 0.01,
    'epochs': 100,
}

if __name__ == '__main__':
    #  调用OGB库中的数据集，本次实验为节点预测，选择数据集为ogbn-arxiv
    dataset_name = 'ogbn-arxiv'
    dataset = PygNodePropPredDataset(dataset_name, transform=T.ToSparseTensor())  # 将数据转换为稀疏矩阵存储

    #  从数据集对象中提取数据部分，并设置邻接矩阵（对称矩阵）
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()

    #  数据的划分：训练、验证和测试
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']

    model = GCN(data.num_features, args['hidden_dim'],
                dataset.num_classes, args['num_layers'],
                args['dropout'])
    evaluator = Evaluator(name='ogbn-arxiv')

    model.reset_parameters()  # 参数的随机初始化

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    loss_fn = F.nll_loss  # 损失函数选择nll

    best_model = None
    best_valid_acc = 0

    for epoch in range(1, 1 + args['epochs']):
        loss = train(model, data, train_idx, optimizer, loss_fn)
        result = test(model, data, split_idx, evaluator)
        train_acc, valid_acc, test_acc = result
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(model)
        print(f'Epoch:{epoch:02d},'
              f'Loss:{loss:.4f},'
              f'Train:{100 * train_acc:.2f}%,'
              f'Valid:{100 * valid_acc:.2f}%,'
              f'Test:{100 * test_acc:.2f}%')

