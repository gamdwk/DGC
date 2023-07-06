from dgl.distributed.kvstore import KVServer, KVClient, PullResponse, PullRequest, default_pull_handler, get_kvstore
from dgl.distributed.rpc import Request, Response
import numpy as np
import dgl.backend.pytorch as F
from dgl import NID, EID, to_block, DGLGraph
from dgl.distributed import DistGraph, DistTensor
from dgl.distributed import rpc
import dgl.distributed.dist_context
from collections import Counter
import torch
from ipc import read_syn_feat, read_syn_label_indices
import dgl.utils
import dgl
from dgl.utils import toindex
from dgl.dataloading.labor_sampler import LaborSampler
from dgl.dataloading import NeighborSampler
from dgl.sampling.labor import sample_labors as local_sample_labors


def sample_labors(g, nodes, fanout, edge_dir="in", prob=None,
                  importance_sampling=0, random_seed=None,
                  exclude_edges=None, output_device=None):
    gpb = g.get_partition_book()
    if not gpb.is_homogeneous:
        assert isinstance(nodes, dict)
        homo_nids = []
        for ntype in nodes:
            assert (
                    ntype in g.ntypes
            ), "The sampled node type does not exist in the input graph"
            if F.is_tensor(nodes[ntype]):
                typed_nodes = nodes[ntype]
            else:
                typed_nodes = toindex(nodes[ntype]).tousertensor()
            homo_nids.append(gpb.map_to_homo_nid(typed_nodes, ntype))
        nodes = F.cat(homo_nids, 0)
    elif isinstance(nodes, dict):
        assert len(nodes) == 1
        nodes = list(nodes.values())[0]

    def issue_remote_req(node_ids):
        if prob is not None:
            # See NOTE 1
            _prob = g.edata[prob].kvstore_key
        else:
            _prob = None
        return SamplingRequest(
            node_ids, fanout, edge_dir=edge_dir, prob=_prob, replace=replace
        )

    def local_access(local_g, partition_book, local_nids):
        # See NOTE 1
        _prob = (
            [g.edata[prob].local_partition] if prob is not None else None
        )
        return _sample_neighbors(
            local_g,
            partition_book,
            local_nids,
            fanout,
            edge_dir,
            _prob,
            replace,
        )

    frontier = _distributed_access(g, nodes, issue_remote_req, local_access)
    return frontier


class LayerSampler(LaborSampler):

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        for i, fanout in enumerate(reversed(self.fanouts)):
            random_seed_i = F.zerocopy_to_dgl_ndarray(
                self.random_seed + (i if not self.layer_dependency else 0)
            )
            frontier = sample_labors(
                g,
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                importance_sampling=self.importance_sampling,
                random_seed=random_seed_i,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
            eid = frontier.edata[EID]
            block = to_block(
                frontier, seed_nodes, include_dst_in_src=True, src_nodes=None
            )
            block.edata[EID] = eid
            if len(g.canonical_etypes) > 1:
                for etype, importance in zip(
                        g.canonical_etypes, importances
                ):
                    if importance.shape[0] == block.num_edges(etype):
                        block.edata["edge_weights"][etype] = importance
            elif importances[0].shape[0] == block.num_edges():
                block.edata["edge_weights"] = importances[0]
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)

        self.set_seed()
        return seed_nodes, output_nodes, blocks

class Sampler_FastGCN:
    def __init__(self, pre_probs, features, adj, **kwargs):
        super().__init__(features, adj, **kwargs)
        # NOTE: uniform sampling can also has the same performance!!!!
        # try, with the change: col_norm = np.ones(features.shape[0])
        col_norm = sparse_norm(adj, axis=0)
        self.probs = col_norm / np.sum(col_norm)

    def sampling(self, v):
        """
        Inputs:
            v: batch nodes list
        """
        all_support = [[]] * self.num_layers

        cur_out_nodes = v
        for layer_index in range(self.num_layers-1, -1, -1):
            cur_sampled, cur_support = self._one_layer_sampling(
                cur_out_nodes, self.layer_sizes[layer_index])
            all_support[layer_index] = cur_support
            cur_out_nodes = cur_sampled

        all_support = self._change_sparse_to_tensor(all_support)
        sampled_X0 = self.features[cur_out_nodes]
        return sampled_X0, all_support, 0

    def _one_layer_sampling(self, v_indices, output_size):
        # NOTE: FastGCN described in paper samples neighboors without reference
        # to the v_indices. But in its tensorflow implementation, it has used
        # the v_indice to filter out the disconnected nodes. So the same thing
        # has been done here.
        support = self.adj[v_indices, :]
        neis = np.nonzero(np.sum(support, axis=0))[1]
        p1 = self.probs[neis]
        p1 = p1 / np.sum(p1)
        sampled = np.random.choice(np.array(np.arange(np.size(neis))),
                                   output_size, True, p1)

        u_sampled = neis[sampled]
        support = support[:, u_sampled]
        sampled_p1 = p1[sampled]

        support = support.dot(sp.diags(1.0 / (sampled_p1 * output_size)))
        return u_sampled, support