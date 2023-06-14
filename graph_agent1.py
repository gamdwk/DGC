import dgl
import torch
import torch.nn as nn
from dgl.distributed import DistGraph
from torch.optim import Adam

import data
from ipc import write_syn_feat, write_syn_label_indices, read_model, write_model
from mmd import compute_mmd
from models.dist_gcn import DistGCN
from models.parametrized_adj import PGE
from sample import get_features_remote_syn
import time
from nic import create_start_nic, compute_nic
import numpy as np
import torch as th
from dgl.dataloading import DistNodeDataLoader, NeighborSampler

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


def load_local_subtensor(g, seeds, input_nodes, device, load_labels=True):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata["features"][input_nodes].to(device)
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
        self.net_interface = "eth0"

    def train(self):
        shuffle = True
        model = self.model.to(self.device)
        if self.args.standalone is False:
            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[self.dev_id],
                                                        output_device=self.device)

        args = self.args
        model = model.to(self.device)
        model.train()
        self.optimizer.zero_grad()
        """sampler = dgl.dataloading.NeighborSampler(
            [int(fanout) for fanout in args.fan_out.split(",")]
        )
        dataloader = dgl.dataloading.DistNodeDataLoader(
            self.g,
            self.data.train_nid,
            sampler,
            batch_size=args.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )"""
        iter_tput = []
        step = 0
        for epoch in range(self.args.epochs + 1):
            start_counters, start_time = create_start_nic(self.net_interface)

            # start_counters, start_time = create_start_nic(net_interface)
            model.train()
            tic = time.time()
            with model.join():
                step_time = []
                sample_time = 0
                forward_time = 0
                backward_time = 0
                update_time = 0
                num_seeds = 0
                num_inputs = 0
                start = time.time()
                tic_step = time.time()

                for c in range(self.n_class):
                    start = time.time()
                    input_nodes, output_nodes, blocks = self.data.retrieve_class_sampler(c, args=self.args)
                    sample_time += tic_step - start

                    batch_inputs, batch_labels = load_subtensor(self.g, output_nodes,
                                                                input_nodes, self.device,
                                                                args.reduction_rate, True)
                    blocks = [block.to(self.device) for block in blocks]
                    batch_labels = batch_labels.long()

                    num_seeds += len(blocks[-1].dstdata[dgl.NID])
                    num_inputs += len(blocks[0].srcdata[dgl.NID])

                    batch_inputs = batch_inputs.to(self.device)
                    start = time.time()
                    dist_full = model(batch_inputs, blocks)

                    forward_end = time.time()
                    loss = self.cross_loss(dist_full, batch_labels)

                    self.optimizer.zero_grad()

                    loss.backward()
                    compute_end = time.time()
                    forward_time += forward_end - start
                    backward_time += compute_end - forward_end

                    self.optimizer.step()
                    update_time += time.time() - compute_end

                    step_t = time.time() - tic_step
                    step_time.append(step_t)
                    iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)
                    if step % args.log_every == 0:
                        acc = self.compute_acc(dist_full, batch_labels)

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
                                step,
                                loss.item(),
                                acc.item(),
                                np.mean(iter_tput[3 * self.n_class:]),
                                gpu_mem_alloc,
                                np.sum(step_time[-args.log_every * self.n_class:]),
                            )
                        )
                    step = step + 1

                # write_syn_feat(self.feat_syn.data)
                if args.standalone:
                    write_model(model)
                else:
                    write_model(model.module)
                elapsed_time, rx_speed, tx_speed = compute_nic(
                    start_counters, start_time, self.net_interface)
                toc = time.time()
                print(
                    "Part {}, Epoch Time(s): {:.4f}, sample+data_copy: {:.4f}, "
                    "forward: {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, "
                    "#inputs: {}, nic时间:{:.2f}秒,接收: {:.2f} bytes/秒,发送: {:.2f} bytes/秒".format(
                        self.g.rank(),
                        toc - tic,
                        sample_time,
                        forward_time,
                        backward_time,
                        update_time,
                        num_seeds,
                        num_inputs,
                        elapsed_time,
                        rx_speed,tx_speed
                    )
                )
                if epoch % args.eval_every == 0 and epoch != 0:
                    start = time.time()

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
                            (self.g.rank(), val_acc, test_acc, time.time() - start)
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


