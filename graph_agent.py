import dgl
import torch
import torch.nn as nn
from dgl.distributed import DistGraph
from torch.optim import Adam

import data
from ipc import write_syn_feat, write_syn_label_indices
from mmd import compute_mmd
from models.dist_gcn import DistGCN
from models.parametrized_adj import PGE
from sample import get_features_remote_syn
import time
from nic import create_start_nic, compute_nic


def get_loops(args):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if args.dataset in ['ogbn-arxiv']:
        return 20, 10
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
    """import pdb
    p = pdb.Pdb()
    p.set_trace()"""
    batch_inputs = get_features_remote_syn(g, input_nodes, rate)
    # batch_inputs = g.ndata["features"][input_nodes].to(device)
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

        write_syn_feat(self.feat_syn.data)

        write_syn_label_indices(syn_data.syn_class_indices)

        # 训练
        self.model = DistGCN(g.ndata["features"].shape[1], args.hidden, self.data.n_class, nlayers=3,
                             dropout=args.dropout)

        self.optimizer = Adam(self.model.parameters(), lr=args.lr_model, weight_decay=args.weight_decay)

        self.loss_fn = compute_mmd
        self.cross_loss = nn.CrossEntropyLoss()
        self.pge = PGE(self.feat_syn.shape[1], self.feat_syn.shape[0], args=args)

        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)

        self.evaluator = evaluator

    def train(self):
        feat_syn, pge, labels_syn = self.feat_syn, self.pge, self.labels_syn
        pge = pge.to(self.device)
        model = self.model.to(self.device)
        if self.args.standalone is False:
            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[self.dev_id],
                                                        output_device=self.device)

        args = self.args
        model = model.to(self.device)
        model.train()
        outer_loop, inner_loop = get_loops(args)
        self.optimizer.zero_grad()
        net_interface = "eth0"
        il_iter_tput = []
        ol_iter_tput = []
        for epoch in range(self.args.epochs + 1):
            start_counters, start_time = create_start_nic(net_interface)
            # compute_nic(start_counters, start_time, net_interface)
            # start_counters, start_time = create_start_nic(net_interface)
            print("epoch{} begin".format(epoch))
            model.train()
            tic = time.time()
            with model.join():
                ol_step_time = []
                ol_sample_time = 0
                ol_forward_time = 0
                ol_backward_time = 0
                ol_update_time = 0
                ol_num_seeds = 0
                ol_num_inputs = 0
                ol_start = time.time()
                for ol in range(outer_loop):
                    tic_step = time.time()
                    # loss = torch.tensor(0.0).to(self.device)
                    for c in range(self.n_class):
                        # input_nodes:计算的数据（包含邻居），output_nodes：不含邻居，需要更新的数据，都是全局节点
                        # blocks：包含邻居的子图
                        adj_syn = pge(self.feat_syn)
                        syn_g = self.get_syn_g(adj_syn)

                        input_nodes, output_nodes, blocks = self.data.retrieve_class_sampler(c, args=self.args)
                        ol_sample_time += tic_step - ol_start
                        batch_inputs, batch_labels = load_subtensor(self.g, output_nodes,
                                                                    input_nodes, self.device,
                                                                    args.reduction_rate, True)
                        # print(batch_inputs, blocks)

                        blocks = [block.to(self.device) for block in blocks]
                        batch_inputs = batch_inputs.to(self.device)
                        ol_num_seeds += len(blocks[-1].dstdata[dgl.NID])
                        ol_num_inputs += len(blocks[0].srcdata[dgl.NID])
                        dist_full = model(batch_inputs, blocks)
                        c_syn_ins = self.data.syn_class_indices[c]
                        ol_start = time.time()
                        if self.args.standalone is False:
                            local_model = model.module.to(self.device)
                        else:
                            local_model = model
                        dist_syn = local_model.forward_by_adj(feat_syn,
                                                              syn_g)
                        forward_end = time.time()
                        rc = dist_full.shape[0] / self.data.syn_num
                        loss = rc * self.loss_fn(dist_full, dist_syn[c_syn_ins[0]:c_syn_ins[1]])

                        ol_forward_time += forward_end - ol_start

                        self.optimizer_pge.zero_grad()
                        self.optimizer_feat.zero_grad()
                        loss.backward()
                        compute_end = time.time()
                        ol_backward_time += compute_end - forward_end
                        if ol % 50 < 10:
                            self.optimizer_feat.step()
                        else:
                            self.optimizer_pge.step()
                        ol_update_time += time.time() - compute_end
                        step_t = time.time() - tic_step
                        ol_step_time.append(step_t)
                        ol_iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)
                    if ol % args.log_every == 0:
                        acc = self.compute_acc(dist_full, batch_labels)
                        import torch as th
                        import numpy as np
                        gpu_mem_alloc = (
                            th.cuda.max_memory_allocated() / 1000000
                            if th.cuda.is_available()
                            else 0
                        )
                        print(
                            "Part {} | Epoch {:05d} | ol_Step {:06d} | Loss {:.4f} | "
                            "Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU "
                            "{:.1f} MB | time {:.3f} s".format(
                                self.g.rank(),
                                epoch,
                                ol,
                                loss.item(),
                                acc.item(),
                                np.mean(ol_iter_tput[3 * self.n_class:]),
                                gpu_mem_alloc,
                                np.sum(ol_step_time[-args.log_every * self.n_class:]),
                            )
                        )
                    ol_start = time.time()
                write_syn_feat(self.feat_syn.data)
                il_step_time = []
                il_sample_time = 0
                il_forward_time = 0
                il_backward_time = 0
                il_update_time = 0
                il_num_seeds = 0
                il_num_inputs = 0
                il_start = time.time()
                for il in range(inner_loop):
                    # adj_syn = pge(self.feat_syn)
                    tic_step = time.time()
                    # syn_g = self.get_syn_g(adj_syn)
                    for c in range(self.n_class):
                        input_nodes, output_nodes, blocks = self.data.retrieve_class_sampler(c, args=self.args)
                        il_sample_time += tic_step - il_start
                        batch_inputs, batch_labels = load_subtensor(self.g, output_nodes,
                                                                    input_nodes, self.device,
                                                                    args.reduction_rate, True)
                        blocks = [block.to(self.device) for block in blocks]
                        batch_labels = batch_labels.long()
                        il_num_seeds += len(blocks[-1].dstdata[dgl.NID])
                        il_num_inputs += len(blocks[0].srcdata[dgl.NID])
                        batch_inputs = batch_inputs.to(self.device)
                        il_start = time.time()
                        dist_full = model(batch_inputs, blocks)
                        """c_syn_ins = self.data.syn_class_indices[c]
                        if self.args.standalone is False:
                            local_model = model.module.to(self.device)
                        else:
                            local_model = model
                        dist_syn = local_model.forward_by_adj(feat_syn, syn_g)
                        rc = dist_full.shape[0] / self.data.syn_num"""
                        # loss = rc * self.loss_fn(dist_full, dist_syn[c_syn_ins[0]:c_syn_ins[1]])
                        forward_end = time.time()
                        loss = self.cross_loss(dist_full, batch_labels)
                        compute_end = time.time()
                        il_forward_time += forward_end - il_start
                        self.optimizer.zero_grad()
                        loss.backward()
                        il_backward_time += compute_end - forward_end
                        self.optimizer.step()
                        il_update_time += time.time() - compute_end

                        step_t = time.time() - tic_step
                        il_step_time.append(step_t)
                        il_iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)
                    if il % args.log_every == 0:
                        acc = self.compute_acc(dist_full, batch_labels)
                        import torch as th
                        import numpy as np
                        gpu_mem_alloc = (
                            th.cuda.max_memory_allocated() / 1000000
                            if th.cuda.is_available()
                            else 0
                        )
                        print(
                            "Part {} | Epoch {:05d} | il_Step {:06d} | Loss {:.4f} | "
                            "Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU "
                            "{:.1f} MB | time {:.3f} s".format(
                                self.g.rank(),
                                epoch,
                                il,
                                loss.item(),
                                acc.item(),
                                np.mean(il_iter_tput[3 * self.n_class:]),
                                gpu_mem_alloc,
                                np.sum(il_step_time[-args.log_every * self.n_class:]),
                            )
                        )
                    il_start = time.time()
                # write_syn_feat(self.feat_syn.data)
                toc = time.time()
                print(
                    "Part {}, Epoch Time(s): {:.4f}, sample+data_copy: {:.4f}, "
                    "forward: {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, "
                    "#inputs: {}".format(
                        self.g.rank(),
                        toc - tic,
                        il_sample_time,
                        il_forward_time,
                        il_backward_time,
                        il_update_time,
                        il_num_seeds,
                        il_num_inputs,
                    )
                )
                if epoch % args.eval_every == 0 and epoch != 0:
                    il_start = time.time()

                    val_acc, test_acc = self.evaluate(
                        model if args.standalone else model.module,
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
                            self.g.rank(), val_acc, test_acc, time.time() - il_start
                        )
                    )

        """A = pge.inference()
        A[A < 0.5] = 0"""

    def reset_parameters(self):
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))

    def evaluate(self, model, g, inputs, labels, val_nid, test_nid, batch_size, device):
        model.eval()
        device = "cpu"
        model = model.to("cpu")
        with torch.no_grad():
            pred = model.inference(g, inputs, batch_size, device)
        model.train()
        model.to(self.device)
        return self.compute_acc(
            pred[val_nid], labels[val_nid]), self.compute_acc(
            pred[test_nid], labels[test_nid])

    def compute_acc(self, pred, labels):
        pred = torch.argmax(pred, dim=1)
        return self.evaluator({"y_pred": pred, "y_true": labels})

    def get_syn_g(self, adj):
        adj[adj < 0.5] = 0
        adj[adj >= 0.5] = 1
        edge_index = torch.nonzero(adj, as_tuple=False).t()
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=self.data.syn_num)
        syn_g = dgl.add_self_loop(g)
        return syn_g
