import argparse
import random
import time

import dgl
import numpy as np
import torch
from dgl.distributed import DistGraph
from ogb.nodeproppred import Evaluator

from data_multi import DGL2Data
from graph_agent_multi import DistGCDM, Condenser


def init(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # dgl.seed(args.seed)
    from sample import init_remote
    init_remote()
    dgl.distributed.initialize(args.ip_config)
    if args.standalone is False:
        torch.distributed.init_process_group(backend="gloo")


def evaluate(input_dict):
    y_true = input_dict["y_true"]
    y_pred = input_dict["y_pred"]
    return (y_true == y_pred).float().sum() / len(y_pred)


def train(x, args):
    init(args)

    g = DistGraph(args.dataset, part_config="data/{}.json".format(args.dataset))
    from sample import pull_handler
    if args.num_gpus == -1:
        device = torch.device("cpu")
        dcdm = DistGCDM(g, args, evaluate, device=device)
    else:
        dev_id = g.rank() % args.num_gpus
        device = torch.device("cuda:" + str(dev_id))
        dcdm = DistGCDM(g, args, evaluate, device=device, dev_id=dev_id)  # Evaluator(name=args.dataset))
    import torch.multiprocessing as mp
    rank = g.rank()
    # process = mp.Process(target=condense, args=(args, rank), daemon=True)
    # mp.spawn(condense, args=(args,), daemon=True)
    # process.start()
    # data = DGL2Data('data/{}.json'.format(args.dataset), rank, args)
    """if args.num_gpus == -1:
        device = torch.device("cpu")
        condenser = Condenser(data, device, args)
    else:
        dev_id = data.rank % args.num_gpus
        device = torch.device("cuda:" + str(dev_id))
        condenser = Condenser(data, device, args, dev_id)"""
    dcdm.train()
    # process.join()
    # condense(args, rank)


def condense(args, rank):
    # init(args)
    import os
    import sys

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #with open(f"condense_{rank}.txt", "w") as f:
        # print(2)

    data = DGL2Data('data/{}.json'.format(args.dataset), rank, args)
    # print(-2)
    if args.num_gpus == -1:
        device = torch.device("cpu")
        condenser = Condenser(data, device, args)
    else:
        dev_id = data.rank % args.num_gpus
        device = torch.device("cuda:" + str(dev_id))
        condenser = Condenser(data, device, args, dev_id)
    # print(3)
    condenser.condense()

def main(args):
    print(args)
    # torch.cuda.set_device(args.gpu_id)

    # random seed setting
    # init(args)
    # g = DistGraph(args.dataset, part_config="data/{}.json".format(args.dataset))

    import torch.multiprocessing as mp
    # mp.spawn()
    # ma = mp.Manager()
    # ma.start()
    if args.act == "train":
        train(1, args)
    else:
        condense(args, rank=args.rank)
    # mp.spawn(train, args=(args,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reduction_rate', type=float, default=0.005)
    parser.add_argument('--keep_ratio', type=float, default=1.0)  # buzhid
    parser.add_argument('--inner', type=int, default=0)
    parser.add_argument('--outer', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--num_gpus', type=int, default=-1, help='num_gpus')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
    #parser.add_argument('--dataset', type=str, default='ogb-product')
    parser.add_argument('--ip_config', type=str, default='ip_config.txt')
    parser.add_argument('--nlayers', type=int, default=3)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--lr_adj', type=float, default=0.01)
    parser.add_argument('--lr_feat', type=float, default=0.01)
    parser.add_argument('--lr_model', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--n_class', type=int, default=40)
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
    parser.add_argument("--fan_out", type=str, default="10,15,10")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--act", type=str, default="train")
    parser.add_argument("--rank", type=int, default=0)  # 压缩的时候一定要传入
    parser.add_argument("--condense", type=int, default=1)  # 不使用压缩的时候是0
    main(parser.parse_args())
