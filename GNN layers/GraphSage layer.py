'''
利用PyG实现消息传播机制，并手工搭建一个GraphSAGE层
GraphSage的结构和模型可见readme.md文档
'''
import torch_scatter
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing, Linear
import torch.nn.functional as F


class GraphSage(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=False, **kwargs):
        '''
        Graph层的初始化
        :param in_channels: 输入特征维度
        :param out_channels: 输入特征维度
        :param normalize: 是否规则化
        :param bias: 是否偏置
        :param kwargs: 其他参数
        '''
        super(GraphSage, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        #  定义两个权重W_l 和 W_r
        self.lin_l = Linear(in_channels, out_channels)
        self.lin_r = Linear(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size=None):
        '''
        前馈传播过程
        :param x: 节点的特征 tensor-[节点个数,特征维度]
        :param edge_index: 边集 tensor-[2,边的个数]
        :param size: 邻接矩阵的大小 turple-(N,N)
        :return: 节点的嵌入表示
        '''

        #  propagate函数会自动顺序地调用message函数和aggregate函数
        out = self.propagate(edge_index, x=(x, x), size=size)

        x = self.lin_l(x)

        out = self.lin_r(out)

        out = out + x

        if self.normalize:
            out = F.normalize(out)

        return out

    def message(self, x_j):
        '''
        该函数是继承MessagePassing基类中的函数，并对函数的功能进行了重写。
        参数是在调用基类中的propagate方法自动完成构建
        :param x_j: 邻居节点特征信息
        :return: 邻居节点信息
        '''
        out = x_j
        return out

    def aggregate(self, inputs, index, dim_size=None):
        '''
        信息聚合
        参数是在调用基类中的propagate方法自动完成构建。具体的每个参数的意义可参见链接：
        https://blog.csdn.net/PolarisRisingWar/article/details/118545695
        :param inputs: 节点特征
        :param index: 所有
        :param dim_size: 维度
        :return:
        '''
        node_dim = self.node_dim
        # scatter函数的意义与用法可参见：https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html
        out = torch_scatter.scatter(inputs, index, dim=node_dim, dim_size=dim_size, reduce='mean')
        return out


if __name__ == '__main__':
    dataset = Planetoid(root='/tmp/cora', name='Cora')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = GraphSage(dataset.num_features, dataset.num_classes)

    for data in loader:
        output = model(data.x, data.edge_index)
        print(output)
