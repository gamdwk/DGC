import dgl
from dgl.distributed import DistGraph
import numpy as np
import torch
from dgl.distributed import load_partition
from ipc import write_syn_label_indices


def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]


class DGL2Data(object):
    def __init__(self, part_json, rank, args, **kwargs):
        self.sorted_counter = None
        self.part_json = part_json
        self.rank = rank
        print("load_data")
        g, ndata, edata, pb, g_name, node_types, edge_types = load_partition(part_json, rank)
        print("load_data成功")
        self.syn_num = None
        # self.class_dict2 = None
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
        # self.g = dgl_dist_graph
        self.local_g = dgl.node_subgraph(g,
                                         g.ndata['inner_node'].bool(),
                                         store_ids=False)
        self.local_g.ndata['labels'] = ndata['_N/labels']
        self.local_g.ndata['features'] = ndata['_N/features']
        self.local_g.ndata['train_mask'] = ndata['_N/train_mask']
        self.local_g.ndata['test_mask'] = ndata['_N/test_mask']
        self.local_g.ndata['val_mask'] = ndata['_N/val_mask']
        self.init_mask()
        self.labels_syn = self.generate_labels_syn(self.local_g.ndata['labels'])
        self.init_features_syn(self.local_g.ndata['features'])
        self.sampler = None
        write_syn_label_indices(self.syn_class_indices)
        print("初始化成功")

    def init_features_syn(self, features):
        n = self.syn_num
        d = features.shape[1]
        features_syn = torch.FloatTensor(n, d)
        self.class_dict = {}
        for c in self.num_class_dict.keys():
            # 存储的是bool，不是id.
            self.class_dict[c] = (self.local_g.ndata['labels'] == c)

        for c in self.num_class_dict.keys():
            num = self.num_class_dict[c]
            idx = self.retrieve_class(c, num)
            if idx.shape[0] < 2:
                temp = np.zeros((2,),dtype=int)
                temp[0] = idx[0]
                temp[1] = idx[0]
                idx = temp
                #idx = torch.cat(torch.from_numpy([idx, idx]))
            idx = torch.from_numpy(np.squeeze(idx))
            # print(idx)

            features_syn[self.syn_class_indices[c][0]:self.syn_class_indices[c][1]] = \
                features[idx][:num]
        self.features_syn = features_syn

    def init_mask(self):
        self.train_mask = self.local_g.ndata['train_mask']
        self.val_mask = self.local_g.ndata['val_mask']
        self.test_mask = self.local_g.ndata['test_mask']
        self.train_nid = mask_to_index(self.train_mask, len(self.train_mask))
        self.val_nid = mask_to_index(self.val_mask, len(self.val_mask))
        self.test_nid = mask_to_index(self.test_mask, len(self.test_mask))

    def generate_labels_syn(self, labels_train):
        from collections import Counter
        labels_train = labels_train.tolist()
        counter = Counter(labels_train)
        num_class_dict = {}
        n = len(labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x: x[1])
        self.sorted_counter = sorted_counter
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
        idx = torch.nonzero(idx)
        # idx = self.get_global_train_idx(idx)
        return np.random.permutation(idx)[:num]

    def retrieve_local_class_sampler(self, c, num=None, args=None):
        sizes = [int(fanout) for fanout in args.fan_out.split(",")]

        seeds = torch.nonzero(self.local_g.ndata["labels"] == c)
        seeds = np.squeeze(seeds)
        if num is not None:
            seeds = seeds[:num]
        from dgl.dataloading import NeighborSampler
        if self.sampler is None:
            self.sampler = NeighborSampler(sizes)
        out = self.sampler.sample(self.local_g, seeds)
        return out

    def load_local_sub_tensor(self, output_nodes, input_nodes, device, load_labels=False):
        batch_inputs = self.local_g.ndata["features"][input_nodes].to(device)
        batch_labels = self.local_g.ndata["labels"][output_nodes].to(device) if load_labels else None
        return batch_inputs, batch_labels


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--reduction_rate', type=float, default=0.0005)
    parser.add_argument('--keep_ratio', type=float, default=1.0)  # buzhid
    parser.add_argument('--inner', type=int, default=0)
    parser.add_argument('--outer', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--num_gpus', type=int, default=2, help='num_gpus')
    parser.add_argument('--dataset', type=str, default='ogb-product')
    parser.add_argument('--ip_config', type=str, default='ip_config.txt')
    parser.add_argument('--nlayers', type=int, default=3)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--lr_adj', type=float, default=0.01)
    parser.add_argument('--lr_feat', type=float, default=0.01)
    parser.add_argument('--lr_model', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--n_class', type=int, default=47)
    parser.add_argument(
        "--local-rank", type=int, default=0, help="get rank of the process"
    )
    parser.add_argument(
        "--standalone", type=bool, default=False, help="standalone"
    )
    parser.add_argument(
        "--pad-data",
        default=False,
        action="store_true",
        help="Pad train nid to the same length across machine, to ensure num "
             "of batches to be the same.",
    )
    parser.add_argument(
        "--net_type",
        type=str,
        default="socket",
        help="backend net type, 'socket' or 'tensorpipe'",
    )
    parser.add_argument("--fan_out", type=str, default="10,25,10")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=5)
    args = parser.parse_args()
    """data = DGL2Data("data/ogbn-arxiv.json", 0, args=args)"""
    """x = data.retrieve_local_class_sampler(c=1,args=args)"""
    from train_dgc_tranductive import condense
    from models.dist_gcn import DistGCN
    from ipc import write_model
    model = DistGCN(100, args.hidden, 47, nlayers=3,
                    dropout=args.dropout)
    write_model(model)
    condense(args, 1)
