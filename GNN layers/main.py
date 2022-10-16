'''
    1、优化器的构建
    2、训练
    3、测试
'''
import torch
import torch.optim as optim
import numpy as np
from torch_geometric.data import DataLoader
from GNN_Learning_PyG.cab3_3 import GNNStack
from torch_geometric.datasets import Planetoid


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


def build_optimizer(args, params):
    '''
    创建优化器，包括学习率调整器
    :param args: 其他参数
    :param params: 模型的参数
    :return: 优化器和学习率调整器
    '''

    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)  # 过滤掉不需要梯度的参数，只将需要计算梯度的参数传入各优化器

    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)

    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)

    return scheduler, optimizer

def test(loader, model, is_validation=True):

    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
            label = data.y

        mask = data.val_mask if is_validation else data.test_mask

        pred = pred[mask]
        label = data.y[mask]

        correct += pred.eq(label).sum().item()

    total = 0
    for data in loader.dataset:
        total += torch.sum(data.val_mask if is_validation else data.test_mask).item()

    return correct / total


def train(dataset, args):
    '''
    训练。注意：数据集已经划分了哪些节点是训练集、验证集和测试集。
    :param dataset: 数据集
    :param args: 其他参数
    :return: 损失值
    '''
    print("Node task, test set size:", np.sum(dataset[0]['train_mask'].numpy()))

    test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes, args)
    scheduler, opt = build_optimizer(args, model.parameters())

    losses = []
    test_accs = []

    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            label = batch.y

            #  将参与训练的节点的标签和预测结果提取出来计算损失值
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
            loss = model.loss(pred, label)

            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs

        if scheduler != None:
            scheduler.step()

        total_loss /= len(loader.dataset)
        losses.append(total_loss)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            test_accs.append(test_acc)
        else:
            test_accs.append(test_accs[-1])

    return test_accs, losses


if __name__ == '__main__':

    for args in [{'model_type': 'GraphSage', 'dataset': 'cora', 'num_layers': 2, 'heads': 1, 'batch_size': 32,
                  'hidden_dim': 32, 'dropout': 0.5, 'epochs': 500, 'opt': 'adam', 'opt_scheduler': 'none',
                  'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.01}]:
        args = objectview(args)

        for model in ['GraphSage', 'GAT']:
            args.model_type = model
            if model == 'GAT':
                args.heads = 2
            else:
                args.heads = 1
            if args.dataset == 'cora':
                dataset = Planetoid(root='/tmp/cora', name='Cora')

            else:
                raise NotImplementedError("Unkown dataset")

            test_acc, losses = train(dataset, args)

            print("Maximum accuracy: {0}".format(max(test_acc)))
            print("Minimum loss: {0}".format(min(losses)))
