import dgl
import torch
import torch.nn as nn
from dgl.distributed import DistGraph
from torch.optim import Adam

import data
from ipc import write_syn_feat, write_syn_label_indices
from mmd import mmd_rbf
from models.dist_gcn import DistGCN
from models.parametrized_adj import PGE
from sample import init_remote
from sample import get_features_remote_syn
import time


def get_loops(args):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if args.one_step:
        return 10, 0

    if args.dataset in ['ogbn-arxiv']:
        return 20, 0
    if args.dataset in ['reddit']:
        return args.outer, args.inner
    if args.dataset in ['flickr']:
        return args.outer, args.inner
        # return 10, 1
    if args.dataset in ['cora']:
        return 20, 10
    if args.dataset in ['citeseer']:
        return 20, 5  # at least 200 epochs
    else:
        return 20, 5


def load_subtensor(g, seeds, input_nodes, device, rate, load_labels=True):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = get_features_remote_syn(g, input_nodes, rate)
    batch_labels = g.ndata["labels"][seeds].to(device) if load_labels else None
    return batch_inputs, batch_labels


class DistGCDM(object):
    def __init__(self, g: DistGraph, syn_data: data.DGL2Data, args, evaluator, device='cuda', **kwargs):
        self.n_class = syn_data.n_class
        self.g = g
        self.args = args
        self.data = syn_data
        if args.num_gpus == -1:
            self.device = torch.device("cpu")
        else:
            self.dev_id = g.rank() % args.num_gpus
            self.device = torch.device("cuda:" + str(self.dev_id))
        self.device_str = str(self.device)
        self.labels_syn = self.data

        self.local_g = syn_data.local_g
        self.local_g.ndata['labels'] = g.ndata['labels'][self.local_g.ndata[dgl.NID]]

        self.feat_syn = nn.Parameter(self.data.features_syn.to(self.device), requires_grad=True)

        write_syn_feat(self.feat_syn)

        write_syn_label_indices(syn_data.syn_class_indices)

        # 训练
        self.model = DistGCN(g.ndata["features"].shape[1] * 2, args.hidden, self.data.n_class, nlayers=3,
                             dropout=args.dropout)

        self.optimizer = Adam(self.model.parameters(), lr=args.lr_model, weight_decay=args.weight_decay)

        self.loss_fn = mmd_rbf
        print(1)
        self.pge = PGE(self.feat_syn.shape[1], self.feat_syn.shape[0], args=args)

        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.evaluator = evaluator

        print("DC初始化成功")

    def train(self):
        feat_syn, pge, labels_syn = self.feat_syn, self.pge, self.labels_syn

        model = nn.parallel.DistributedDataParallel(self.model,
                                                    device_ids=[self.dev_id],
                                                    output_device=self.device)
        args = self.args
        model = model.to(self.device)
        model.train()
        outer_loop, inner_loop = get_loops(args)
        self.optimizer.zero_grad()
        for epoch in range(self.args.epochs + 1):
            print("epoch{} begin".format(epoch))
            model.train()
            for ol in range(outer_loop):

                # loss = torch.tensor(0.0).to(self.device)
                for c in self.n_class:
                    # input_nodes:计算的数据（包含邻居），output_nodes：不含邻居，需要更新的数据，都是全局节点
                    # blocks：包含邻居的子图
                    adj_syn = pge(self.feat_syn)
                    input_nodes, output_nodes, blocks = self.data.retrieve_class_sampler(c)
                    batch_inputs, batch_labels = load_subtensor(self.g, blocks[0].srcdata[dgl.NID],
                                                                blocks[-1].dstdata[dgl.NID], self.device,
                                                                args.reduction_rate, False)

                    dist_full = model(batch_inputs, blocks)
                    c_syn_ins = self.data.syn_class_indices[c]
                    local_model = model.module.to(self.device)
                    dist_syn = local_model(feat_syn[c_syn_ins[0]:c_syn_ins[1]],
                                           adj_syn[c_syn_ins[0]:c_syn_ins[1], c_syn_ins[0]:c_syn_ins[1]])
                    rc = dist_full.shape[0] / self.data.syn_num
                    loss = rc * self.loss_fn(dist_full, dist_syn)
                    self.optimizer_pge.zero_grad()
                    self.optimizer_feat.zero_grad()
                    loss.backward()
                    if ol % 50 < 10:
                        self.optimizer_feat.step()
                    else:
                        self.optimizer_pge.step()
            for il in range(inner_loop):
                adj_syn = pge(self.feat_syn)
                for c in self.n_class:
                    input_nodes, output_nodes, blocks = self.data.retrieve_class_sampler(c)
                    batch_inputs, batch_labels = load_subtensor(self.g, blocks[0].srcdata[dgl.NID],
                                                                blocks[-1].dstdata[dgl.NID], self.device,
                                                                args.reduction_rate, False)

                    dist_full = model(batch_inputs, blocks)
                    c_syn_ins = self.data.syn_class_indices[c]
                    local_model = model.module.to(self.device)
                    dist_syn = local_model(feat_syn[c_syn_ins[0]:c_syn_ins[1]],
                                           adj_syn[c_syn_ins[0]:c_syn_ins[1], c_syn_ins[0]:c_syn_ins[1]])
                    rc = dist_full.shape[0] / self.data.syn_num
                    loss = rc * self.loss_fn(dist_full, dist_syn)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            write_syn_feat(self.feat_syn)

            if epoch % args.eval_every == 0 and epoch != 0:
                start = time.time()
                val_acc, test_acc = self.evaluate(
                    model.module if args.standalone else model.module,
                    self.g,
                    self.g.ndata["features"],
                    self.g.ndata["labels"],
                    self.data.val_nid,
                    self.data.test_nid,
                    args.batch_size_eval,
                    self.device,
                )
                print(
                    "Part {}, Val Acc {:.4f}, Test Acc {:.4f}, time: {:.4f}".format
                        (
                        self.g.rank(), val_acc, test_acc, time.time() - start
                    )
                )
        """A = pge.inference()
        A[A < 0.5] = 0"""

    def reset_parameters(self):
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))

    def evaluate(self, model, g, inputs, labels, val_nid, test_nid, batch_size, device):
        model.eval()
        with torch.no_grad():
            pred = model.inference(g, inputs, batch_size, device)
        model.train()
        return self.compute_acc(
            pred[val_nid], labels[val_nid]), self.compute_acc(
            pred[test_nid], labels[test_nid])

    def compute_acc(self, pred, labels):
        return self.evaluator.eval({"y_pred": pred, "y_true": labels})
