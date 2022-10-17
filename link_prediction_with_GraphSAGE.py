import copy
import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch import nn
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset
from torch_geometric.datasets import Planetoid
from sklearn.metrics import *
from torch.utils.data import DataLoader


class LinkPredModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(LinkPredModel, self).__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, batch):
        node_feature, edge_index, edge_label_index = batch.node_feature, batch.edge_index, batch.edge_label_index
        pred = self.conv1(node_feature, edge_index)
        pred = F.relu(pred)
        pred = F.dropout(pred, self.dropout, training=self.training)
        pred = self.conv2(pred, edge_index)  # [节点数,num_classes]

        sp_edges = pred[edge_label_index]  # [2,supervision边数,num_classes]
        source_nodes = sp_edges[0]  # [supervision边数,num_classes]
        destination_nodes = sp_edges[1]
        #  点积计算相似性，就相似性越高认为两点之间越有边
        pred = (source_nodes * destination_nodes).sum(axis=1)

        return pred

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)


def train(model, dataloaders, optimizer, args):
    val_max = 0
    best_model = model

    for epoch in range(1, args['epoch']):
        for i, batch in enumerate(dataloaders['train']):
            model.train()
            optimizer.zero_grad()
            p = model(batch)
            loss = model.loss(p, batch.edge_label.float())
            loss.backward()
            optimizer.step()

            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Loss: {}'
            score_train = test(model, dataloaders['train'], args)
            score_val = test(model, dataloaders['val'], args)
            score_test = test(model, dataloaders['test'], args)

            print(log.format(epoch, score_train, score_val, score_test, loss))
            if val_max < score_val:
                val_max = score_val
                best_model = copy.deepcopy(model)
    return best_model

def test(model, dataloader, args):
    model.eval()
    score = 0
    for batch in dataloader:
        p = model(batch)
        p = torch.sigmoid(p)

        # 将tensor转换成array格式
        p = p.cpu().detach().numpy()
        label = batch.edge_label.cpu().detach().numpy()

        score += roc_auc_score(label, p)

    score = score/len(dataloader)
    return score

if __name__ == '__main__':
    args = {
        "hidden_dim": 128,
        "epoch": 200
    }
    pyg_dataset = Planetoid('/tmp/cora', 'Cora')
    graphs = GraphDataset.pyg_to_graphs(pyg_dataset)
    dataset = GraphDataset(graphs, task='link_pred', edge_train_mode='disjoint')

    datasets = {}
    datasets['train'], datasets['val'], datasets['test'] = dataset.split(transductive=True, split_ratio=[0.85, 0.05, 0.1])

    input_dim = datasets['train'].num_node_features
    num_class = datasets['train'].num_edge_labels

    model = LinkPredModel(input_dim, args['hidden_dim'], num_class)
    model.reset_parameters()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    dataloaders = {split: DataLoader(ds, collate_fn=Batch.collate([]), batch_size=1,
                                    shuffle=(split == 'train'))
                  for split, ds in datasets.items()}

    best_model = train(model, dataloaders, optimizer, args)

    log = "Train: {:.4f}, Val: {:.4f}, Test: {:.4f}"
    best_train_roc = test(best_model, dataloaders['train'], args)
    best_val_roc = test(best_model, dataloaders['val'], args)
    best_test_roc = test(best_model, dataloaders['test'], args)
    print(log.format(best_train_roc, best_val_roc, best_test_roc))
