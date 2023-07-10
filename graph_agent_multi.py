import atexit
import time

import dgl
import numpy as np
import torch
import torch as th
import torch.nn as nn
from dgl.distributed import DistGraph
from torch.optim import Adam

from ipc import write_syn_feat, read_model, write_model, del_model, del_syn
from models.dist_gcn import DistGCN
from models.parametrized_adj import PGE
from nic import create_start_nic, compute_nic


def load_subtensor(g, seeds, input_nodes, device, rate, load_labels=True):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    """import pdb
    p = pdb.Pdb()
    p.set_trace()"""
    from sample import get_features_remote_syn
    batch_inputs = get_features_remote_syn(g, input_nodes, rate)
    batch_inputs = batch_inputs.to(device)
    # batch_inputs = g.ndata["features"][input_nodes].to(device)
    batch_labels = g.ndata["labels"][seeds].to(device) if load_labels else None
    return batch_inputs, batch_labels


def raw_load_subtensor(g, seeds, input_nodes, device, rate, load_labels=True):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    """import pdb
    p = pdb.Pdb()
    p.set_trace()"""
    batch_inputs = g.ndata["features"][input_nodes].to(device)
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
    def __init__(self, g: DistGraph, args, evaluator, device, dev_id=None, **kwargs):
        self.n_class = args.n_class
        self.g = g
        self.args = args
        # self.data = syn_data
        self.init_mask(g)
        self.device = device
        self.dev_id = dev_id
        self.device_str = str(self.device)
        g.local_partition.create_formats_()
        self.local_g = dgl.node_subgraph(g.local_partition,
                                         g.local_partition.ndata['inner_node'].bool(),
                                         store_ids=False)
        self.local_g.ndata['labels'] = g.ndata['labels'][self.local_g.ndata[dgl.NID]]
        # 训练

        if args.model == "GCN":
            self.model = DistGCN(g.ndata["features"].shape[1], args.hidden, self.n_class,
                                 nlayers=args.nlayers,
                                 dropout=args.dropout, device=self.device)
        elif args.model == "GraphSage":
            from models.dist_sage import DistSAGE
            import torch.nn.functional as F
            self.model = DistSAGE(
                g.ndata["features"].shape[1],
                args.hidden,
                self.n_class,
                args.nlayers,
                F.relu,
                args.dropout,
                self.device
            )
        write_model(self.model)
        self.optimizer = None
        self.cross_loss = nn.CrossEntropyLoss()

        self.evaluator = evaluator
        if args.standalone:
            self.net_interface = "eno2"
        else:
            self.net_interface = "eth0"
        # self.net_interface = "eth0"
        atexit.register(del_model)

        # wait_for_syn()

    def train(self):
        shuffle = True
        model = self.model.to(self.device)
        if self.args.standalone is False:
            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[self.dev_id],
                                                        output_device=self.device)

        args = self.args
        model = model.to(self.device)
        self.optimizer = Adam(model.parameters(), lr=args.lr_model, weight_decay=args.weight_decay)
        fan = [int(fanout) for fanout in args.fan_out.split(",")]

        if args.sampler == "fast_gcn":
            from models.fast_gcn import FastGCNSampler
            sampler = FastGCNSampler(fan)
        else:
            sampler = dgl.dataloading.MultiLayerNeighborSampler(fan)

        dataloader = dgl.dataloading.DistNodeDataLoader(
            self.g,
            self.train_nid,
            sampler,
            batch_size=args.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=args.num_workers,
        )
        iter_tput = []
        step = 0
        # init_g_features(self.g)
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
                load_time = 0
                start = time.time()
                tic_step = time.time()
                for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                    # for c in range(self.n_class):
                    # start = time.time()
                    tic_step = time.time()
                    sample_time += tic_step - start
                    # input_nodes, output_nodes, blocks = self.data.retrieve_class_sampler(c, args=self.args)
                    before_load = time.time()
                    if args.condense:
                        batch_inputs, batch_labels = load_subtensor(self.g, output_nodes,
                                                                    input_nodes, self.device,
                                                                    args.reduction_rate, True)
                    else:
                        batch_inputs, batch_labels = raw_load_subtensor(self.g, output_nodes,
                                                                        input_nodes, self.device,
                                                                        args.reduction_rate, True)
                    load_time += time.time() - before_load
                    # t2 = time.time()
                    # batch_inputs, batch_labels = raw_load_subtensor(self.g, output_nodes,
                    #                                                input_nodes, self.device,
                    #                                                args.reduction_rate, True)
                    # t3 = time.time()
                    # load_time = t2-t1
                    # raw_load_time = t3-t2
                    # print(f"load_time:{load_time},raw_load_time{raw_load_time}")
                    blocks = [blk.int().to(self.device) for blk in blocks]
                    batch_labels = batch_labels.long()

                    num_seeds += len(blocks[-1].dstdata[dgl.NID])
                    num_inputs += len(blocks[0].srcdata[dgl.NID])

                    batch_inputs = batch_inputs.to(self.device)
                    start = time.time()
                    dist_full = model(batch_inputs, blocks)

                    forward_end = time.time()
                    # print(dist_full,batch_labels)
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
                            "TrainPart {} | Epoch {:05d} | Step {:06d} | Loss {:.4f} | "
                            "Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU "
                            "{:.1f} MB | time {:.3f} s".format(
                                self.g.rank(),
                                epoch,
                                step,
                                loss.item(),
                                acc.item(),
                                np.mean(iter_tput[3:]),
                                gpu_mem_alloc,
                                np.sum(step_time[-args.log_every:]),
                            )
                        )
                    # step = step + 1
                    start = time.time()

                # write_syn_feat(self.feat_syn.data)
                if self.args.condense:
                    if args.standalone:
                        write_model(model)
                    else:
                        write_model(model.module)

                elapsed_time, rx_speed, tx_speed = compute_nic(
                    start_counters, start_time, self.net_interface)
                toc = time.time()
                print(
                    "TrainPart {}, Epoch Time(s): {:.4f}, sample+data_copy: {:.4f}, "
                    "forward: {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, "
                    "#inputs: {}, throughput:{} nodes/s, nic时间:{:.2f}秒,接收: {:.2f} bytes/秒,发送: {:.2f} bytes/秒,"
                    "总消耗：{:.2f} bytes".format(
                        self.g.rank(),
                        toc - tic,
                        sample_time,
                        forward_time,
                        backward_time,
                        update_time,
                        num_seeds,
                        num_inputs,
                        float(num_inputs) / float(load_time),
                        elapsed_time,
                        rx_speed, tx_speed, (rx_speed + tx_speed) * elapsed_time
                    )
                )
                if epoch % args.eval_every == 0 and epoch != 0:
                    start = time.time()

                    val_acc, test_acc = self.evaluate(
                        model if args.standalone else model.module,
                        self.g,
                        self.g.ndata["features"],
                        self.g.ndata["labels"],
                        self.val_nid,
                        self.test_nid,
                        args.batch_size_eval,
                        self.device,
                    )
                    print(
                        "TrainPart {}, Val Acc {:.4f}, Test Acc {:.4f}, time: {:.4f}".format
                        (self.g.rank(), val_acc, test_acc, time.time() - start)
                    )

        """A = pge.inference()
        A[A < 0.5] = 0"""

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

    def evaluate(self, model, g, inputs, labels, val_nid, test_nid, batch_size, device):
        model.eval()
        # device = "cpu"
        # model = model.to("cpu")
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


class Condenser(object):
    def __init__(self, syn_data, device, args, dev_id=None, full_graph=True):
        self.n_class = syn_data.n_class
        self.args = args
        self.device = device
        self.dev_id = dev_id
        self.data = syn_data

        self.feat_syn = nn.Parameter(self.data.features_syn, requires_grad=True)

        #write_syn_feat(self.feat_syn.data)
        #write_syn_label_indices(syn_data.syn_class_indices)
        print("rank{}:{}".format(args.rank, syn_data.syn_class_indices))
        self.pge = PGE(self.feat_syn.shape[1], self.feat_syn.shape[0], args=args, nlayers=2)

        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        atexit.register(del_syn)
        from ipc import wait_for_model
        wait_for_model()

    def condense(self):
        iter_tput = []
        model = read_model()
        model = model.to(self.device)
        pge = self.pge.to(self.device)
        # pge = self.pge
        args = self.args
        loss_fn = self.loss_fn
        sorted_counter = self.data.sorted_counter
        # feat_syn = self.feat_syn
        epoch = 0
        while True:
            model.train()
            tic = time.time()
            # 获取训练进程的模型
            model = read_model()
            model = model.to(self.device)
            write_syn_feat(feat=self.feat_syn.data)
            with model.join():
                step_time = []
                sample_time = 0
                forward_time1 = 0
                forward_time2 = 0
                backward_time = 0
                update_time = 0
                num_seeds = 0
                num_inputs = 0
                step = 0
                # loss = torch.tensor(0.0).to(self.device)
                # 按类训练
                for c, num in sorted_counter:
                    # input_nodes:计算的数据（包含邻居），output_nodes：不含邻居，需要更新的数据，都是全局节点
                    # blocks：包含邻居的子图
                    # 对合成图和原图进行取样。合成图只要class = c的节点。原图要进行邻居采样

                    start = time.time()
                    feat_syn = self.feat_syn.data.to(self.device)
                    adj_syn = pge(feat_syn)
                    adj_syn = adj_syn.to('cpu')
                    syn_g = self.get_syn_g(adj_syn)
                    syn_g.ndata["features"] = feat_syn.to('cpu')
                    indices = self.data.syn_class_indices[c]
                    c_syn_g = dgl.node_subgraph(syn_g, np.arange(indices[0], indices[1]))
                    c_syn_feat = c_syn_g.ndata["features"]

                    input_nodes, output_nodes, blocks = self.data.retrieve_local_class_sampler(
                        c, args=self.args)
                    tic_step = time.time()
                    sample_time += tic_step - start
                    #
                    batch_inputs, batch_labels = self.data.load_local_sub_tensor(output_nodes,
                                                                                 input_nodes, self.device, True)
                    # print(batch_inputs, blocks)

                    blocks = [block.to(self.device) for block in blocks]
                    batch_inputs = batch_inputs.to(self.device)

                    num_seeds += len(blocks[-1].dstdata[dgl.NID])
                    num_inputs += len(blocks[0].srcdata[dgl.NID])

                    start = time.time()
                    dist_full = model(batch_inputs, blocks)
                    # c_syn_ins = self.data.syn_class_indices[c]
                    forward_end1 = time.time()

                    c_syn_g = c_syn_g.to(self.device)
                    c_syn_feat = c_syn_feat.to(self.device)
                    dist_syn = model.forward_by_adj(c_syn_feat, c_syn_g)
                    forward_end2 = time.time()
                    rc = dist_full.shape[0] / self.data.syn_num
                    loss = rc * self.loss_fn(dist_full, dist_syn)

                    forward_time1 += forward_end1 - start
                    forward_time2 += forward_end2 - forward_end1

                    self.optimizer_pge.zero_grad()
                    self.optimizer_feat.zero_grad()
                    loss.backward()

                    compute_end = time.time()
                    backward_time += compute_end - forward_end2

                    if epoch % 10 < 5:
                        self.optimizer_feat.step()
                    else:
                        self.optimizer_pge.step()

                    update_time += time.time() - compute_end
                    step_t = time.time() - tic_step
                    step_time.append(step_t)
                    iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)

                    # acc = self.compute_acc(dist_full, batch_labels)
                    if step % args.log_every == 0:
                        gpu_mem_alloc = (
                            th.cuda.max_memory_allocated() / 1000000
                            if th.cuda.is_available()
                            else 0
                        )
                        print(
                            "CondensePart {} | Epoch {:05d} | class {:06d} | Loss {:.4f} | "
                            "Speed (samples/sec) {:.4f} | GPU "
                            "{:.1f} MB | time {:.3f} s".format(
                                self.data.rank,
                                epoch,
                                c,
                                loss.item(),
                                np.mean(iter_tput[-args.log_every:]),
                                gpu_mem_alloc,
                                np.sum(step_time[-args.log_every:]),
                            )
                        )
                    step = step + 1
            write_syn_feat(self.feat_syn.data)
            epoch += 1
            toc = time.time()
            print(
                "CondensePart {}, Epoch Time(s): {:.4f}, sample+data_copy: {:.4f}, "
                "forward: {:.4f},forward syn: {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, "
                "#inputs: {}".format(
                    self.data.rank,
                    toc - tic,
                    sample_time,
                    forward_time1,
                    forward_time2,
                    backward_time,
                    update_time,
                    num_seeds,
                    num_inputs,
                )
            )

    def get_syn_g(self, adj):
        adj[adj < 0.5] = 0
        adj[adj >= 0.5] = 1
        edge_index = torch.nonzero(adj, as_tuple=False).t()
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=self.data.syn_num)
        syn_g = dgl.add_self_loop(g)
        return syn_g

    def loss_fn(self, dist_full, dist_syn):
        from mmd import compute_mmd3
        return compute_mmd3(dist_full, dist_syn, device=self.device)
