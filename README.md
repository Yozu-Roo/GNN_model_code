# GNN-model-code

[TOC]

*更新中……*

该项目主要记录使用PyG、OGB等框架对GNN模型进行构建的过程，其中的资源和编码的方法和思路参照了斯坦福大学的公开课《图机器学习》和一些大佬的博客，后续分章节提供相关的引文链接。

[PyG官方文档](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.GCN)

[斯坦福大学CS224w官网](http://web.stanford.edu/class/cs224w/)

[GitHub上关于该课程实验的参考代码](https://github.com/PolarisRisingWar/cs224w-2021-winter-colab)

## 1. node_prediction_with_GCN.py

该文件主要目的是利用GCN实现对节点类别的预测，用到的数据集为[ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv)，GCN模型的原理与结构可参考[【斯坦福大学公开课CS224W——图机器学习】六、图神经网络1：GNN模型](https://blog.csdn.net/qq_45955883/article/details/127135419)

所参考使用的GCN结构图 [![image](https://user-images.githubusercontent.com/114124424/193591834-97b77245-8cd2-4a56-8ae8-7a7840232559.png)](https://user-images.githubusercontent.com/114124424/193591834-97b77245-8cd2-4a56-8ae8-7a7840232559.png)

代码的运行结果：

```python
Epoch: 01, Loss: 4.0492, Train: 25.35%, Valid: 28.71% Test: 25.71%
Epoch: 02, Loss: 2.3172, Train: 26.54%, Valid: 25.99% Test: 31.52%
Epoch: 03, Loss: 1.9148, Train: 28.85%, Valid: 23.71% Test: 27.98%
Epoch: 04, Loss: 1.7875, Train: 38.76%, Valid: 34.36% Test: 38.54%
Epoch: 05, Loss: 1.6523, Train: 44.26%, Valid: 41.42% Test: 39.92%
Epoch: 06, Loss: 1.5905, Train: 42.94%, Valid: 37.80% Test: 36.83%
Epoch: 07, Loss: 1.5237, Train: 38.57%, Valid: 30.34% Test: 32.19%
Epoch: 08, Loss: 1.4630, Train: 35.33%, Valid: 26.59% Test: 29.29%
Epoch: 09, Loss: 1.4224, Train: 33.98%, Valid: 26.00% Test: 29.01%
Epoch: 10, Loss: 1.3806, Train: 32.60%, Valid: 23.77% Test: 26.16%
Epoch: 11, Loss: 1.3508, Train: 33.04%, Valid: 24.98% Test: 26.93%
Epoch: 12, Loss: 1.3235, Train: 35.74%, Valid: 30.87% Test: 32.98%
Epoch: 13, Loss: 1.2994, Train: 39.26%, Valid: 35.40% Test: 38.59%
Epoch: 14, Loss: 1.2766, Train: 42.55%, Valid: 38.34% Test: 42.86%
Epoch: 15, Loss: 1.2579, Train: 46.46%, Valid: 42.82% Test: 47.82%
Epoch: 16, Loss: 1.2423, Train: 50.38%, Valid: 47.91% Test: 52.64%
Epoch: 17, Loss: 1.2272, Train: 54.46%, Valid: 53.63% Test: 57.32%
Epoch: 18, Loss: 1.2086, Train: 57.20%, Valid: 57.74% Test: 60.28%
Epoch: 19, Loss: 1.1971, Train: 58.74%, Valid: 59.79% Test: 61.64%
Epoch: 20, Loss: 1.1854, Train: 59.59%, Valid: 60.64% Test: 62.07%
Epoch: 21, Loss: 1.1732, Train: 60.31%, Valid: 61.18% Test: 62.34%
Epoch: 22, Loss: 1.1665, Train: 60.55%, Valid: 61.09% Test: 62.17%
Epoch: 23, Loss: 1.1569, Train: 60.76%, Valid: 61.12% Test: 62.10%
Epoch: 24, Loss: 1.1494, Train: 61.55%, Valid: 61.65% Test: 62.50%
Epoch: 25, Loss: 1.1322, Train: 62.90%, Valid: 63.09% Test: 63.70%
Epoch: 26, Loss: 1.1262, Train: 64.00%, Valid: 64.11% Test: 65.01%
Epoch: 27, Loss: 1.1184, Train: 64.68%, Valid: 64.75% Test: 65.66%
Epoch: 28, Loss: 1.1145, Train: 65.20%, Valid: 64.86% Test: 65.83%
Epoch: 29, Loss: 1.1042, Train: 65.63%, Valid: 65.07% Test: 65.96%
Epoch: 30, Loss: 1.0977, Train: 65.80%, Valid: 65.28% Test: 66.03%
Epoch: 31, Loss: 1.0915, Train: 66.15%, Valid: 65.58% Test: 66.41%
Epoch: 32, Loss: 1.0855, Train: 66.42%, Valid: 66.01% Test: 66.88%
Epoch: 33, Loss: 1.0797, Train: 66.93%, Valid: 66.58% Test: 67.31%
Epoch: 34, Loss: 1.0748, Train: 67.44%, Valid: 67.21% Test: 67.74%
Epoch: 35, Loss: 1.0710, Train: 67.99%, Valid: 67.92% Test: 68.17%
Epoch: 36, Loss: 1.0658, Train: 68.62%, Valid: 68.59% Test: 68.32%
Epoch: 37, Loss: 1.0624, Train: 69.06%, Valid: 68.80% Test: 68.39%
Epoch: 38, Loss: 1.0585, Train: 69.19%, Valid: 69.07% Test: 68.44%
Epoch: 39, Loss: 1.0525, Train: 69.41%, Valid: 69.42% Test: 68.87%
Epoch: 40, Loss: 1.0507, Train: 69.70%, Valid: 69.65% Test: 69.32%
Epoch: 41, Loss: 1.0435, Train: 69.93%, Valid: 69.78% Test: 69.64%
Epoch: 42, Loss: 1.0412, Train: 70.27%, Valid: 70.00% Test: 69.61%
Epoch: 43, Loss: 1.0392, Train: 70.46%, Valid: 70.18% Test: 69.69%
Epoch: 44, Loss: 1.0333, Train: 70.59%, Valid: 70.13% Test: 69.82%
Epoch: 45, Loss: 1.0307, Train: 70.59%, Valid: 70.07% Test: 69.97%
Epoch: 46, Loss: 1.0276, Train: 70.72%, Valid: 70.00% Test: 69.94%
Epoch: 47, Loss: 1.0243, Train: 70.80%, Valid: 70.11% Test: 69.78%
Epoch: 48, Loss: 1.0219, Train: 70.85%, Valid: 70.06% Test: 69.66%
Epoch: 49, Loss: 1.0178, Train: 71.00%, Valid: 70.14% Test: 69.66%
Epoch: 50, Loss: 1.0142, Train: 71.01%, Valid: 70.26% Test: 69.60%
Epoch: 51, Loss: 1.0094, Train: 71.10%, Valid: 70.46% Test: 69.63%
Epoch: 52, Loss: 1.0076, Train: 71.29%, Valid: 70.58% Test: 69.63%
Epoch: 53, Loss: 1.0048, Train: 71.49%, Valid: 70.71% Test: 69.50%
Epoch: 54, Loss: 1.0041, Train: 71.55%, Valid: 70.54% Test: 69.41%
Epoch: 55, Loss: 0.9992, Train: 71.59%, Valid: 70.51% Test: 69.44%
Epoch: 56, Loss: 0.9976, Train: 71.64%, Valid: 70.63% Test: 69.61%
Epoch: 57, Loss: 0.9940, Train: 71.68%, Valid: 70.61% Test: 69.62%
Epoch: 58, Loss: 0.9913, Train: 71.76%, Valid: 70.56% Test: 69.31%
Epoch: 59, Loss: 0.9947, Train: 71.87%, Valid: 70.68% Test: 69.53%
Epoch: 60, Loss: 0.9858, Train: 71.95%, Valid: 70.87% Test: 69.82%
Epoch: 61, Loss: 0.9847, Train: 71.90%, Valid: 70.95% Test: 70.15%
Epoch: 62, Loss: 0.9830, Train: 71.92%, Valid: 70.92% Test: 70.16%
Epoch: 63, Loss: 0.9800, Train: 71.99%, Valid: 71.01% Test: 70.24%
Epoch: 64, Loss: 0.9785, Train: 72.09%, Valid: 71.10% Test: 70.17%
Epoch: 65, Loss: 0.9766, Train: 72.17%, Valid: 71.17% Test: 70.13%
Epoch: 66, Loss: 0.9722, Train: 72.20%, Valid: 71.18% Test: 70.19%
Epoch: 67, Loss: 0.9742, Train: 72.16%, Valid: 71.15% Test: 70.63%
Epoch: 68, Loss: 0.9688, Train: 72.10%, Valid: 71.04% Test: 70.69%
Epoch: 69, Loss: 0.9681, Train: 72.27%, Valid: 71.23% Test: 70.66%
Epoch: 70, Loss: 0.9608, Train: 72.45%, Valid: 71.23% Test: 70.30%
Epoch: 71, Loss: 0.9639, Train: 72.57%, Valid: 71.17% Test: 70.01%
Epoch: 72, Loss: 0.9603, Train: 72.57%, Valid: 71.31% Test: 70.30%
Epoch: 73, Loss: 0.9572, Train: 72.52%, Valid: 71.45% Test: 70.81%
Epoch: 74, Loss: 0.9572, Train: 72.51%, Valid: 71.52% Test: 70.92%
Epoch: 75, Loss: 0.9554, Train: 72.62%, Valid: 71.51% Test: 70.69%
Epoch: 76, Loss: 0.9542, Train: 72.76%, Valid: 71.41% Test: 70.15%
Epoch: 77, Loss: 0.9537, Train: 72.87%, Valid: 71.55% Test: 70.44%
Epoch: 78, Loss: 0.9498, Train: 72.85%, Valid: 71.53% Test: 70.85%
Epoch: 79, Loss: 0.9486, Train: 72.73%, Valid: 71.41% Test: 71.09%
Epoch: 80, Loss: 0.9472, Train: 72.76%, Valid: 71.46% Test: 71.09%
Epoch: 81, Loss: 0.9465, Train: 73.03%, Valid: 71.60% Test: 70.77%
Epoch: 82, Loss: 0.9410, Train: 73.12%, Valid: 71.51% Test: 70.52%
Epoch: 83, Loss: 0.9386, Train: 73.05%, Valid: 71.57% Test: 70.52%
Epoch: 84, Loss: 0.9383, Train: 73.01%, Valid: 71.67% Test: 70.83%
Epoch: 85, Loss: 0.9423, Train: 73.11%, Valid: 71.69% Test: 70.79%
Epoch: 86, Loss: 0.9351, Train: 73.11%, Valid: 71.53% Test: 70.54%
Epoch: 87, Loss: 0.9336, Train: 73.20%, Valid: 71.34% Test: 70.33%
Epoch: 88, Loss: 0.9324, Train: 73.30%, Valid: 71.51% Test: 70.51%
Epoch: 89, Loss: 0.9315, Train: 73.39%, Valid: 71.86% Test: 70.98%
Epoch: 90, Loss: 0.9284, Train: 73.41%, Valid: 71.92% Test: 70.98%
Epoch: 91, Loss: 0.9278, Train: 73.38%, Valid: 71.85% Test: 70.85%
Epoch: 92, Loss: 0.9248, Train: 73.28%, Valid: 71.77% Test: 71.09%
Epoch: 93, Loss: 0.9242, Train: 73.42%, Valid: 71.64% Test: 70.69%
Epoch: 94, Loss: 0.9197, Train: 73.46%, Valid: 71.46% Test: 70.07%
Epoch: 95, Loss: 0.9235, Train: 73.65%, Valid: 71.56% Test: 70.26%
Epoch: 96, Loss: 0.9211, Train: 73.65%, Valid: 71.84% Test: 70.87%
Epoch: 97, Loss: 0.9153, Train: 73.45%, Valid: 71.69% Test: 71.03%
Epoch: 98, Loss: 0.9190, Train: 73.57%, Valid: 71.85% Test: 71.05%
Epoch: 99, Loss: 0.9167, Train: 73.65%, Valid: 71.84% Test: 71.05%
Epoch: 100, Loss: 0.9143, Train: 73.72%, Valid: 71.84% Test: 71.04%
```

## 2.graph_predition_with_GNN.py

该模型调用了上面的节点分类文件，将嵌入后的节点表示进行全局平均池化，作为图的嵌入表示，再进行训练。

## 3. GNN layers

GNN单层PyG创建，具体的技术实现原理可细节可参考[cs224w（图机器学习）2021冬季课程学习笔记13 Colab 3](https://blog.csdn.net/PolarisRisingWar/article/details/118545695)

本次实验手动创建了GraphSAGE和GAT层，并通过GNN Stack进行调用。实验的数据集是Cora，Cora数据集包含2708篇科学出版物， 5429条边，总共7种类别。数据集中的每个出版物都由一个 0/1 值的词向量描述，表示字典中相应词的缺失/存在。 该词典由 1433 个独特的词组成。意思就是说每一个出版物都由1433个特征构成，每个特征仅由0/1表示。

**注意：Cora数据集中每个节点已经被划分好，哪些是训练、验证和测试。**

数据集包含以下文件：

- ind.cora.x : 训练集节点特征向量，保存对象为：scipy.sparse.csr.csr_matrix，实际展开后大小为： (140, 1433)
- ind.cora.tx : 测试集节点特征向量，保存对象为：scipy.sparse.csr.csr_matrix，实际展开后大小为： (1000, 1433)
- ind.cora.allx : 包含有标签和无标签的训练节点特征向量，保存对象为：scipy.sparse.csr.csr_matrix，实际展开后大小为：(1708, 1433)，可以理解为除测试集以外的其他节点特征集合，训练集是它的子集
- ind.cora.y : one-hot表示的训练节点的标签，保存对象为：numpy.ndarray
- ind.cora.ty : one-hot表示的测试节点的标签，保存对象为：numpy.ndarray
- ind.cora.ally : one-hot表示的ind.cora.allx对应的标签，保存对象为：numpy.ndarray
- ind.cora.graph : 保存节点之间边的信息，保存格式为：{ index : [ index_of_neighbor_nodes ] }
- ind.cora.test.index : 保存测试集节点的索引，保存对象为：List，用于后面的归纳学习设置。

[PyG的Planetoid无法直接下载Cora等数据集的3个解决方式](https://blog.csdn.net/PolarisRisingWar/article/details/116399648)

> 模型的训练表现：
>
> Node task. test set size: 140
> Maximum accuracy: 0.766
> Minimum loss: 0.10853853076696396
> Node task. test set size: 140
> Maximum accuracy: 0.782
> Minimum loss: 0.028875034302473068

## 4.link_prediction_with_GraphSAGE.py

本次实验是基于Deepsnap Basic的链接预测任务，即预测两个节点之间是否存在边。使用的数据集是Cora数据集，将数据集采用inductive方式划分。

> 训练：用 training message edges 预测 training supervision edges
>
> 验证：用 training message edges 和 training supervision edges 预测 validation edges
>
> 测试：用 training message edges 和 training supervision edges 和 validation edges 预测 test edges

本次实验的参考博客：https://blog.csdn.net/PolarisRisingWar/article/details/118545695

针对Cora数据集划分的具体情况见表格：

|           | edge_index (message passing edges) | edge_label_index (supervision edges) | edge_label |
| :-------: | :--------------------------------: | :----------------------------------: | :--------: |
| train_set |                7176                |                 3592                 |    3592    |
| valid_set |                8972                |                 1052                 |    1052    |
| test_set  |                9498                |                 2116                 |    2116    |

其中，edge_label为数据集划分后的属性，其代表supervision edge的真实标签类型（即是否有边），supervision edges中正负样本边各占50%（负样本，即采样一些不存在边的节点对作为负样本边）。
$$
EdgeIndex_{valid}=EdgeIndex_{train}+0.5*EdgeLabelIndex_{train}
$$

$$
EdgeIndex_{test}=EdgeIndex_{valid}+0.5*EdgeLabelIndex_{valid}
$$

在训练过程中，使用message passing edges进行消息传递，即不参与预测；使用supervision edges进行预测。

最终的模型效果：

> Train: 0.8823, Val: 0.7995, Test: 0.7942
