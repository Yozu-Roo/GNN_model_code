'''
利用PyG实现消息传播机制，并手工搭建一个GAT类
GAT的结构和模型可见readme.md文档
'''
import torch
import torch_scatter
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import Planetoid


class GAT(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=2, negative_slope=0.2, dropout=0., **kwargs):
        '''
        模型的初始化
        :param in_channels: 输入特征维度
        :param out_channels: 输出特征维度
        :param heads: 多头注意力机制中，头的个数，默认为2
        :param negative_slope: 默认为0.2
        :param dropout: 隐退的概率，默认为0.
        :param kwargs: 其他参数
        '''
        super(GAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        #  权重的初始化(W_l,W_r)，同时也对特征维度进行了增强。
        self.lin_l = Linear(in_channels, heads * out_channels)
        self.lin_r = self.lin_l

        #  注意力分数的初始化，维度为[1, head, output_dim]
        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))  # [1, H, C]
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size=None):
        '''
        前馈传播过程
        :param x: 节点的特征信息，[节点的个数，节点特征维度]
        :param edge_index: 边集
        :param size:
        :return:
        '''

        H, C = self.heads, self.out_channels

        #  权重的初始化，W_l,W_r
        x_l = self.lin_l(x)  # [N,H*C]
        x_r = self.lin_r(x)  # [N,H*C]

        #  维度变换
        x_l = x_l.view(-1, H, C)  # [N,H,C]
        x_r = x_r.view(-1, H, C)

        #  先进行内积，然后按照特征的维度进行求和，可参照attension机制中a的算法：每个key的value计算出后需要将所有的value进行累加求和
        alpha_l = (x_l * self.att_l).sum(axis=-1)  # [N,H]
        alpha_r = (x_r * self.att_r).sum(axis=-1)

        out = self.propagate(edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r), size=size)

        out = out.view(-1, H * C)

        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        # alpha:[E,H]
        # x:[N,H,C]
        # 其余参数为softmax所需参数https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch-geometric-utils

        # 步骤：
        # 在message而非aggregate函数中应用attention
        # attention coefficient=LeakyReLU(alpha_i+alpha_j)
        # attention weight=softmax(attention coefficient)（就这两步都是alpha，就在代码里没区分e和alpha）
        # embeddings * attention weights

        alpha = alpha_i + alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training).unsqueeze(-1)
        out = x_j * alpha  # [E,H,C]

        return out

    def aggregate(self, inputs, index, dim_size=None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
        return out


if __name__ == '__main__':
    # https://blog.csdn.net/PolarisRisingWar/article/details/116399648
    dataset = Planetoid(root='/tmp/cora', name='Cora')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = GAT(dataset.num_features, dataset.num_classes)

    for data in loader:
        output = model(data.x, data.edge_index)
        print(output)
