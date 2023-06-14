import dgl
from dgl.distributed import DistGraph
import numpy as np
import torch


def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]


class DGL2Data(object):
    def __init__(self, dgl_dist_graph: DistGraph, args, **kwargs):
        self.syn_num = None
        self.class_dict2 = None
        self.num_class_dict = None
        self.syn_class_indices = None
        self.test_nid = None
        self.val_nid = None
        self.train_nid = None
        self.test_mask = None
        self.train_mask = None
        self.val_mask = None
        self.features_syn = None
        self.class_dict = None
        self.n_class = args.n_class
        self.args = args
        self.g = dgl_dist_graph
        self.local_g = dgl.node_subgraph(dgl_dist_graph.local_partition,
                                         dgl_dist_graph.local_partition.ndata['inner_node'].bool(),
                                         store_ids=False)
        self.local_g.ndata['labels'] = dgl_dist_graph.ndata['labels'][self.local_g.ndata[dgl.NID]]
        self.idx_full = self.local_g.ndata[dgl.NID]
        self.init_mask(dgl_dist_graph)
        self.labels_train = dgl_dist_graph.ndata["labels"][self.train_nid]
        self.labels_syn = self.generate_labels_syn(self.labels_train)
        self.init_features_syn(dgl_dist_graph.ndata["features"])
        self.sampler = None

    def init_features_syn(self, features):
        n = self.syn_num
        d = features.shape[1]
        features_syn = torch.FloatTensor(n, d)
        self.class_dict = {}
        for c in self.num_class_dict.keys():
            # 存储的是bool，不是id.
            self.class_dict[c] = (self.labels_train == c)
        for c in self.num_class_dict.keys():
            num = self.num_class_dict[c]
            idx = self.retrieve_class(c, num)
            features_syn[self.syn_class_indices[c][0]:self.syn_class_indices[c][1]] = \
                features[idx][:num]
        self.features_syn = features_syn

    def init_mask(self, g: DistGraph):
        self.train_mask = g.ndata['train_mask']
        self.val_mask = g.ndata['val_mask']
        self.test_mask = g.ndata['test_mask']
        pb = g.get_partition_book()
        self.train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"],
            pb,
            force_even=True
        )
        self.val_nid = dgl.distributed.node_split(
            g.ndata["val_mask"],
            pb,
            force_even=True
        )
        self.test_nid = dgl.distributed.node_split(
            g.ndata["test_mask"],
            pb,
            force_even=True
        )

    def generate_labels_syn(self, labels_train):
        from collections import Counter
        labels_train = labels_train.tolist()
        counter = Counter(labels_train)
        num_class_dict = {}
        n = len(labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x: x[1])

        labels_syn = []
        self.syn_class_indices = {}
        # syn_class_indices:合成图中class对应的index范围，闭区间
        # labels_syn:合成图中的label列表。每个长度都是n*reduction_rate。最后一个截断
        # num_class_dict:每个class在合成图中的数量
        self.syn_num = 0
        for ix, (c, num) in enumerate(sorted_counter):
            num_class_dict[c] = max(int(num * self.args.reduction_rate), 2)
            self.syn_num += num_class_dict[c]
            self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
            labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return torch.LongTensor(labels_syn)

    def retrieve_class(self, c, num):
        idx = self.class_dict[c]
        idx = self.get_global_train_idx(idx)
        return np.random.permutation(idx)[:num]

    def retrieve_class_sampler(self, c, num=None, args=None):
        self.class_dict2 = self.class_dict
        """sizes = [15, 10, 5]
        if args.nlayers == 1:
            sizes = [15]
        if args.nlayers == 2:
            sizes = [10, 5]
            # sizes = [-1, -1]
        if args.nlayers == 3:
            sizes = [15, 10, 5]
        if args.nlayers == 4:
            sizes = [15, 10, 5, 5]
        if args.nlayers == 5:
            sizes = [15, 10, 5, 5, 5]"""
        sizes = [int(fanout) for fanout in args.fan_out.split(",")]
        from dgl.dataloading import NeighborSampler
        node_idx = self.class_dict2[c]
        node_idx = self.get_global_train_idx(node_idx)
        if self.sampler is None:
            self.sampler = NeighborSampler(sizes)
        if num is not None:
            batch = np.random.permutation(self.class_dict2[c])[:num]
        else:
            batch = node_idx
        out = self.sampler.sample(self.g, batch)
        return out

    def retrieve_local_class_sampler(self, c, num=None, args=None):
        """sizes = [15, 10, 5]
        if args.nlayers == 1:
            sizes = [15]
        if args.nlayers == 2:
            sizes = [10, 5]
            # sizes = [-1, -1]
        if args.nlayers == 3:
            sizes = [15, 10, 5]
        if args.nlayers == 4:
            sizes = [15, 10, 5, 5]
        if args.nlayers == 5:
            sizes = [15, 10, 5, 5, 5]"""
        sizes = [int(fanout) for fanout in args.fan_out.split(",")]
        seeds = torch.nonzero(self.g.ndata["labels"].local_partition == c)
        if num is not None:
            seeds = seeds[:num]
        from dgl.dataloading import NeighborSampler
        if self.sampler is None:
            self.sampler = NeighborSampler(sizes)
        out = self.sampler.sample(self.local_g, seeds)
        return out

    def load_local_sub_tensor(self, output_nodes, input_nodes, device, load_labels=False):
        batch_inputs = self.g.ndata["features"].local_partition[input_nodes].to(device)
        batch_labels = self.g.ndata["labels"].local_partition[output_nodes].to(device) if load_labels else None
        return batch_inputs, batch_labels

    def get_global_idx(self, idx):
        return self.idx_full[idx]

    def get_global_train_idx(self, idx):
        return self.train_nid[idx]
