from contextlib import contextmanager

import dgl
import numpy as np
import tqdm
from torch import nn
import dgl.nn.pytorch as dglnn
import torch as th


class DistSAGE(nn.Module):
    def __init__(
            self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, device='cpu'
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.nclass = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.device = device

    def forward(self, x, blocks):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[:block.number_of_dst_nodes()]
            #print(h.shape)
            #print(h_dst.shape)
            h = layer(block, (h, h_dst))
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def forward_by_adj(self, x, g):
        x = x.to(self.device)
        g = g.to(self.device)
        for ix, layer in enumerate(self.layers):
            x = layer(g, x)
            if ix != len(self.layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        return x

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
        """nodes = dgl.distributed.node_split(
            np.arange(g.num_nodes()),
            g.get_partition_book(),
            force_even=True,
        )"""
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

            # sampler = dgl.dataloading.NeighborSampler([-1])
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DistNodeDataLoader(
                g,
                th.arange(g.num_nodes()),
                sampler,
                batch_size=batch_size,
                num_workers=0,
                shuffle=True,
                drop_last=False
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].int().to(device)
                h = x[input_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if i != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
            g.barrier()
        return y

    @contextmanager
    def join(self):
        """dummy join for standalone"""
        yield
