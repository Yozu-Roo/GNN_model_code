'''
GNN Stack Module
'''
import torch
from torch import nn
from torch.nn import functional as F
from GNN_model_code.GNN_layers.GraphSage_layer import GraphSage
from GNN_model_code.GNN_layers.GAT_layer import GAT
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader


class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, emb=False):
        '''
        模型结构的初始化
        :param input_dim: 输入特征维度
        :param hidden_dim: 隐藏层的维度
        :param output_dim: 输出层的维度
        :param args: 其他参数
        :param emb: 是否返回嵌入，默认false
        '''
        super(GNNStack, self).__init__()

        #  将GraphSage模型或GAT模型引入
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        assert (args.num_layers >= 1), 'Number of layers is not >= 1'

        #  选择多个模型进行连接
        for l in range(args.num_layers - 1):
            self.convs.append(conv_model(args.heads * hidden_dim, hidden_dim))

        #  后处理模型
        self.post_mp = nn.Sequential(
            nn.Linear(args.heads * hidden_dim, hidden_dim),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.dropout = args.dropout
        self.nums_layers = args.num_layers
        self.emb = emb

    def build_conv_model(self, model_type):#  根据传入参数的类型不同，返回不同的基础模型

        if model_type == 'GraphSage':
            return GraphSage
        elif model_type == 'GAT':
            return GAT

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.nums_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.post_mp(x)

        #  emb为true则返回节点嵌入表示，否则返回节点预测分类结果
        if self.emb == True:
            return x

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


if __name__ == '__main__':
    # https://blog.csdn.net/PolarisRisingWar/article/details/116399648
    dataset = Planetoid(root='/tmp/cora', name='Cora')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    args = {'model_type': 'GraphSage', 'dataset': 'cora', 'num_layers': 2, 'heads': 1, 'batch_size': 32,
            'hidden_dim': 32, 'dropout': 0.5, 'epochs': 500, 'opt': 'adam', 'opt_scheduler': 'none',
            'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.01}
    args = objectview(args)

    model = GNNStack(dataset.num_features, 32, dataset.num_classes, args)

    for data in loader:
        output = model(data)
        print(output)
