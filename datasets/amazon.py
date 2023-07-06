import torch
from dgl.data.dgl_dataset import DGLDataset


class AmazonFromGraphSaintDataset(DGLDataset):
    def process(self):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


import dgl
import numpy as np
import scipy.sparse
import torch as th
def get_amazon():
    sparse_matrix = scipy.sparse.load_npz("amazon/adj_full.npz")
    graph = dgl.from_scipy(sparse_matrix, eweight_name="w")
    train_sparse_matrix = np.load("amazon/adj_train.npz")
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_nids = train_sparse_matrix["indices"]
    print(train_nids)

    train_nids = th.unique(th.squeeze(th.from_numpy(train_nids)))
    print(train_nids.shape)
    train_mask[train_nids] = True
    feats = np.load("amazon/feats.npy")
    labels = np.load("amazon/labels.npy")
    labels = np.argmax(labels, axis=1)
    graph.ndata["features"] = th.from_numpy(feats)
    graph.ndata["labels"] = th.squeeze(th.from_numpy(labels))
    graph.ndata["train_mask"] = train_mask
    val_mask = test_mask = ~train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask
    return graph

if __name__ == '__main__':
    graph = get_amazon()
    dgl.save_graphs("../dataset/amazon", [graph])
    graphs = dgl.load_graphs("../dataset/amazon")
