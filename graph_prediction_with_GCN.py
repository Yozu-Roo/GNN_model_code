'''
图预测任务：基于GCN模型，在化学分子结构图数据集中进行训练，并对图像进行二分预测
'''

import copy
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import global_mean_pool
import torch
from GNN_Learning_PyG.cab2_1 import GCN
from operator import itemgetter

#  本次用到的数据集是化学分子结构图，每张图展示一个化学分子结构，每个原子都有不同的特征，原子之间用化学健链接
dataset = PygGraphPropPredDataset(name='ogbg-molhiv')

split_idx = dataset.get_idx_split()

# 下面的代码直接运行可能会产生错误，因计算机而异吧，报错原因尚未调查清楚，如果能正常运行下面的代码建议使用这些，参考的来源：https://ogb.stanford.edu/docs/home/
# train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True, num_workers=0)
# valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=32, shuffle=False)
# test_loader = DataLoader(dataset[split_idx['test']], batch_size=32, shuffle=False)

#  若上面注释的代码无法正常运行，可以使用下面的代码来实现同样的功能，经过测试，最终的结果并无异样
#  将整个数据集划分完毕后，按照32张图一组打包成一个batch
train_loader = DataLoader(itemgetter(*split_idx['train'])(dataset), batch_size=32, shuffle=True, num_workers=0)
valid_loader = DataLoader(itemgetter(*split_idx['valid'])(dataset), batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(itemgetter(*split_idx['test'])(dataset), batch_size=32, shuffle=False, num_workers=0)



class GCN_Graph(torch.nn.Module):

    def __init__(self, hidden_dim, output_dim, num_layers, dropout):
        '''
        模型构建
        :param hidden_dim: 隐层节点维度，256
        :param output_dim: 输出层维度，1
        :param num_layers: 层数，5
        :param dropout: 若采用dropout方法，随机隐退的概率，0.5
        '''
        super(GCN_Graph, self).__init__()

        #  为了编码原始的节点特征，使用AtomEncoder方法，对这些节点的特征进行嵌入，即原始特征——>嵌入的特征
        self.node_encoder = AtomEncoder(hidden_dim)

        #  这里使用了上一个文件中构建的GCN模型，以此来实现对节点的嵌入表示，嵌入特征+图结构特征——>节点完整的嵌入表示
        self.gnn_node = GCN(hidden_dim, hidden_dim,
                            hidden_dim, num_layers, dropout, return_embeds=True)

        self.pool = None

        #  全局平均池化，将每个batch中每个图中的节点信息取平均进行池化，最后再以batch打包成向量返回
        #  https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.global_mean_pool
        self.pool = global_mean_pool

        #  输出层，将数据转换为分类结果
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def reset_parameters(self):
        self.gnn_node.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, batched_data):
        '''
        模型的前向传播过程
        :param batched_data: 一个batch中的所有信息
        :return: 一个batch中每张图的嵌入表示
        '''
        #  batch_data.x：batch中所有节点的嵌入，Tensor类型。
        #  batch_data.edge_index：batch中每条边的连接情况
        #  batch_data.batch：一个batch中每个节点归属的图标号
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        embed = self.node_encoder(x)

        out = None

        out = self.gnn_node(embed, edge_index)
        out = self.pool(out, batch)
        out = self.linear(out)

        return out


def train(model, data_loader, optimizer, loss_fn):
    model.train()
    loss = 0

    for batch in data_loader:
        #  如果batch中无节点，或者batch中的节点只有1种则跳过
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            is_labeled = batch.y == batch.y  # 将标签不为nan的数据索引保留

            optimizer.zero_grad()
            op = model(batch)
            train_op = op[is_labeled]  # 将真实标签不为nan的数据的预测结果保留
            train_labels = batch.y[is_labeled].view(-1)
            loss = loss_fn(train_op.float(), train_labels.float())

            loss.backward()
            optimizer.step()

    return loss.item()

def eval(model, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        if batch.x.shape[0] == 1:  # 如果batch中无节点则跳过
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true":y_true, "y_pred":y_pred}

    return evaluator.eval(input_dict)

if __name__ == '__main__':

    args = {
        'num_layers': 5,
        'hidden_dim': 256,
        'dropout': 0.5,
        'lr': 0.001,
        'epoch': 30
    }

    model = GCN_Graph(args['hidden_dim'],
                dataset.num_tasks, args['num_layers'],
                args['dropout'])
    evaluator = Evaluator(name='ogbg-molhiv')

    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_model = None
    best_valid_acc = 0

    for epoch in range(1, 1+args["epoch"]):
        print("Training...")
        loss = train(model, train_loader, optimizer, loss_fn)

        print('Evaluating...')
        train_result = eval(model, train_loader, evaluator)
        val_result = eval(model, valid_loader, evaluator)
        test_result = eval(model, test_loader, evaluator)

        train_acc, valid_acc, test_acc = train_result[dataset.eval_metric], val_result[dataset.eval_metric], test_result[dataset.eval_metric]

        if valid_acc > best_valid_acc:
          best_valid_acc = valid_acc
          best_model = copy.deepcopy(model)
        print(f'Epoch: {epoch:02d}, '
            f'Loss: {loss:.4f}, '
            f'Train: {100 * train_acc:.2f}%, '
            f'Valid: {100 * valid_acc:.2f}% '
            f'Test: {100 * test_acc:.2f}%')

