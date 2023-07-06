from contextlib import contextmanager

import dgl
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


class DistGCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers=3, dropout=0.5, lr=0.01, weight_decay=5e-4,
                 with_relu=True, with_bias=True, with_bn=False, device=None):

        super(DistGCN, self).__init__()

        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass
        self.n_hidden = nhid
        self.layers = nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(GraphConv(nfeat, nclass))
        else:
            self.layers.append(GraphConv(nfeat, nhid))
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            for i in range(nlayers - 2):
                self.layers.append(GraphConv(nhid, nhid))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConv(nhid, nclass))

        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bn = with_bn
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.multi_label = None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        # self.log_sigmoid = nn.LogSigmoid()

    def forward(self, x, blocks):
        for ix, (layer, block) in enumerate(zip(self.layers, blocks)):
            x = layer(block, x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = self.relu(x)
                x = self.dropout(x)
        if self.multi_label:
            return self.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_by_adj(self, x, g):
        x = x.to(self.device)
        g = g.to(self.device)
        for ix, layer in enumerate(self.layers):
            x = layer(g, x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = self.relu(x)
                x = self.dropout(x)
        if self.multi_label:
            return self.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of GCN.
        """
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without
        neighbor sampling).

        g : the entire graph.
        x : the input of entire node set.

        Distributed layer-wise inference.
        """
        # During inference with sampling, multi-layer blocks are very
        # inefficient because lots of computations in the first few layers
        # are repeated. Therefore, we compute the representation of all nodes
        # layer by layer.  The nodes on each layer are of course splitted in
        # batches.
        import dgl
        import numpy as np
        import torch as th
        import tqdm
        nodes = dgl.distributed.node_split(
            np.arange(g.num_nodes()),
            g.get_partition_book(),
            force_even=True,
        )
        y = dgl.distributed.DistTensor(
            (g.num_nodes(), self.n_hidden),
            th.float32,
            "h",
            persistent=True,
        )
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                y = dgl.distributed.DistTensor(
                    (g.num_nodes(), self.nclass),
                    th.float32,
                    "h_last",
                    persistent=True,
                )
            print(
                f"|V|={g.num_nodes()}, eval batch size: {batch_size}"
            )

            sampler = dgl.dataloading.NeighborSampler([-1])
            dataloader = dgl.dataloading.DistNodeDataLoader(
                g,
                nodes,
                sampler,
                batch_size=batch_size,
                num_workers=0,
                shuffle=False,
                drop_last=False,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                h = self.bns[i](h) if self.with_bn else h
                if self.with_relu:
                    h = self.relu(h)
                h = self.dropout(h)
                if i == len(self.layers) - 1:
                    if self.multi_label:
                        h = self.sigmoid(h)
                    else:
                        h = F.log_softmax(h, dim=1)
                y[output_nodes] = h.cpu()

            x = y
            g.barrier()
        return y

    @contextmanager
    def join(self):
        """dummy join for standalone"""
        yield


if __name__ == '__main__':
    gcn = DistGCN(128, 128, 40)
