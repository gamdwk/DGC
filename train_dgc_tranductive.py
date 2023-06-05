import argparse
import random

import dgl
import numpy as np
import torch
from dgl.distributed import DistGraph
from ogb.nodeproppred import Evaluator

from data import DGL2Data
from graph_agent import DistGCDM


def evaluate(input_dict):
    y_true = input_dict["y_true"]
    y_pred = input_dict["y_pred"]
    return (y_true == y_pred).float().sum() / len(y_pred)


def main(args):
    # torch.cuda.set_device(args.gpu_id)

    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # dgl.seed(args.seed)
    from sample import init_remote
    init_remote()
    # args.num_gpus = 1
    dgl.distributed.initialize(args.ip_config)
    if args.standalone is False:
        torch.distributed.init_process_group(backend="gloo")

    g = DistGraph(args.dataset, part_config="data/{}.json".format(args.dataset))
    data = DGL2Data(g, args)
    dcdm = DistGCDM(g, data, args, evaluate)  # Evaluator(name=args.dataset))
    dcdm.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reduction_rate', type=float, default=0.005)
    parser.add_argument('--keep_ratio', type=float, default=1.0)#buzhid
    parser.add_argument('--inner', type=int, default=0)
    parser.add_argument('--outer', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--num_gpus', type=int, default=-1, help='num_gpus')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
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
        "--batch_size_eval", type=int, default=1000
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
    parser.add_argument(
        "--eval_every",
        type=int,
        default=10,
    )
    main(parser.parse_args())
