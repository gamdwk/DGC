from dgl.distributed.kvstore import KVServer, KVClient, PullResponse, PullRequest, default_pull_handler, get_kvstore
from dgl.distributed.rpc import Request, Response
import numpy as np
import dgl.backend.pytorch as F
from dgl.distributed import DistGraph, DistTensor
from dgl.distributed import rpc
import dgl.distributed.dist_context
from collections import Counter
import torch
from ipc import read_syn_feat, read_syn_label_indices
import dgl.utils
import pdb


class SynFeatRequest(Request):
    def __getstate__(self):
        return self.idx, self.labels_name, self.rate

    def __setstate__(self, state):
        self.idx, self.labels_name, self.rate = state

    def process_request(self, server_state):
        kv_store = server_state.kv_store
        if self.labels_name not in kv_store.part_policy:
            raise RuntimeError("KVServer cannot find partition policy with name: %s" % self.labels_name)
        if self.labels_name not in kv_store.data_store:
            raise RuntimeError("KVServer Cannot find data tensor with name: %s" % self.labels_name)
        local_id = kv_store.part_policy[self.labels_name].to_local(self.idx)
        labels = kv_store.pull_handlers[self.labels_name](kv_store.data_store, self.labels_name, local_id)
        labels_dict = {}

        syn_label_indices = read_syn_label_indices()
        syn_feat = read_syn_feat()

        num_all = 0

        d = syn_feat.shape[1]

        c = Counter(labels.tolist())

        syn_ids = {}
        for i, (label, num) in enumerate(c.items()):
            num = num * self.rate

            num = int(num) if num > 1 else 1
            labels_dict[label] = (num_all, num_all + num)
            num_all += num
            syn_idx = np.random.randint(syn_label_indices[label][0], syn_label_indices[label][1], num)
            syn_ids[label] = syn_idx
        data = torch.zeros((num_all, d))
        for i, (label, num) in enumerate(c.items()):
            syn_idx = syn_ids[label]
            data[labels_dict[label][0]:labels_dict[label][1]] = syn_feat[syn_idx]
        local_to_full = torch.zeros((len(self.idx),), dtype=torch.int64)

        for i, label in enumerate(labels.tolist()):
            local_to_full[i] = torch.tensor(np.random.randint(labels_dict[label][0], labels_dict[label][1], 1),
                                            dtype=torch.int64)

        res = SynFeatResponse(kv_store.server_id, data, local_to_full)
        return res

    def __init__(self, idx, labels_name, rate):
        self.idx = idx
        self.labels_name = labels_name
        self.rate = rate


class SynFeatResponse(Response):
    def __init__(self, server_id, data_tensor, local_to_full):
        self.server_id = server_id
        self.data_tensor = data_tensor
        self.local_to_full = local_to_full

    def __getstate__(self):
        return self.server_id, self.data_tensor, self.local_to_full

    def __setstate__(self, state):
        self.server_id, self.data_tensor, self.local_to_full = state


def take_id(elem):
    return elem.server_id


def get_features_remote_syn(g: DistGraph, id_tensor, rate):
    print("id_tensor:", id_tensor)

    id_tensor = dgl.utils.toindex(id_tensor)
    id_tensor = id_tensor.tousertensor()
    print("id_tensor2:", id_tensor)
    features = g.ndata['features']
    labels = g.ndata['labels']
    part_policy = labels._part_policy
    labels_name = labels._name
    client = get_kvstore()

    pb = g.get_partition_book()

    # g.get_node_partition_policy(ntype=g.ntypes[0])
    part_id = pb.partid
    machine_id = part_policy.to_partid(id_tensor)
    sorted_id = F.tensor(np.argsort(F.asnumpy(machine_id)))
    back_sorted_id = F.tensor(np.argsort(F.asnumpy(sorted_id)))
    machine, count = np.unique(F.asnumpy(machine_id), return_counts=True)
    start = 0
    pull_count = 0
    local_id = None
    for idx, machine_idx in enumerate(machine):
        end = start + count[idx]
        if start == end:  # No data for target machine
            continue
        partial_id = id_tensor[start:end]
        if machine_idx == part_id:  # local pull
            # Note that DO NOT pull local data right now because we can overlap
            # communication-local_pull here
            # local_id = part_policy.to_local(partial_id)
            local_id = partial_id

            """print("local_id:", local_id)
            print("partial_id", partial_id)
            a = g.local_partition.ndata[dgl.NID]
            b = partial_id
            i = torch.unique(a[(a.unsqueeze(1) == b).any(dim=1)])
            # i = torch.intersect1d(g.local_partition.ndata[dgl.NID], partial_id)
            print("gobal_id", i)
            indices = torch.nonzero(torch.isin(g.local_partition.ndata[dgl.NID], i)).flatten()

            print("indices", indices)
            print(indices == local_id)"""
            """print(partial_id)
            print(features[partial_id])
            print(features.local_partition[local_id])"""
        else:  # pull data from remote server
            request = SynFeatRequest(partial_id, labels_name, rate)
            rpc.send_request_to_machine(machine_idx, request)
            pull_count += 1
        start += count[idx]
    # recv response
    response_list = []
    if local_id is not None:  # local pull

        # local_data = features.local_partition[local_id]
        """
        local_data = features.local_partition[local_id]"""
        local_data = features[local_id]
        server_id = part_id
        local_response = PullResponse(server_id, local_data)
        response_list.append(local_response)
    # wait response from remote server nodes
    for _ in range(pull_count):
        remote_response = rpc.recv_response()
        response_list.append(remote_response)
    # sort response by server_id and concat tensor
    response_list.sort(key=take_id)

    def handle(response):
        data = response.data_tensor
        s_id = response.server_id
        if s_id == part_id:
            return data
        else:
            local_to_full = response.local_to_full
            # print(local_to_full)
            data_full = data[local_to_full]
            return data_full

    data_tensor = F.cat(seq=[handle(response) for response in response_list], dim=0)
    return data_tensor[back_sorted_id]  # return data with original index order


def init_remote():
    MY_SYN_PULL = 638408
    rpc.register_service(MY_SYN_PULL, SynFeatRequest,
                         SynFeatResponse)


from dgl.dataloading import DistNodeDataLoader, DataLoader

if __name__ == '__main__':
    init_remote()
