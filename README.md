# GNN-model-code

该项目主要记录使用PyG、OGB等框架对GNN模型进行构建的过程，其中的资源和编码的方法和思路参照了斯坦福大学的公开课《图机器学习》和一些大佬的博客，后续分章节提供相关的引文链接。

[PyG官方文档](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.GCN)

[斯坦福大学CS224w官网](http://web.stanford.edu/class/cs224w/)

[GitHub上关于该课程实验的参考代码](https://github.com/PolarisRisingWar/cs224w-2021-winter-colab)

## 1. node_prediction_with_GCN.py

该文件主要目的是利用GCN实现对节点类别的预测，用到的数据集为[ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv)，所参考使用的GCN结构图 [![image](https://user-images.githubusercontent.com/114124424/193591834-97b77245-8cd2-4a56-8ae8-7a7840232559.png)](https://user-images.githubusercontent.com/114124424/193591834-97b77245-8cd2-4a56-8ae8-7a7840232559.png)
