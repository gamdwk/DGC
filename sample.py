from dgl.distributed.kvstore import KVServer, KVClient, PullResponse, PullRequest, default_pull_handler
from dgl.distributed.rpc import Request, Response
import numpy as np
import dgl.backend.pytorch as F
from dgl.distributed import DistGraph, DistTensor
from dgl.distributed import rpc
import dgl.distributed.dist_context
from collections import Counter
import torch
from ipc import read_syn_feat, read_syn_label_indices


class SynFeatRequest(Request):
    def __getstate__(self):
        return self.idx, self.labels_name, self.rate

    def __setstate__(self, state):
        self.idx, self.labels_name, self.rate = state

    def process_request(self, server_state):
        kv_store = server_state.kv_store
        local_id = kv_store.part_policy[self.labels_name].to_local(self.idx)
        labels = kv_store.data_store[self.labels_name][local_id]

        labels_dict = {}
        num_all = self.rate * len(self.idx)

        syn_label_indices = read_syn_label_indices()
        syn_feat = read_syn_feat()

        local_to_full = torch.zeros(num_all)
        d = syn_feat[1]
        data = torch.zeros((num_all, d))

        c = Counter(labels)
        for i, (label, num) in enumerate(c):
            if i != len(c) - 1:
                num = int(num * self.rate)
                labels_dict[label] = (len(labels_dict), len(labels_dict) + num)
            else:
                num = num_all - len(labels_dict)
                labels_dict[label] = (len(labels_dict), num_all)
            label_syn_feat = syn_feat[syn_label_indices[label][0]:syn_label_indices[label][1]]
            data[labels_dict[label][0]:labels_dict[label][1]] = np.random.choice(label_syn_feat, num)
        for i, label in enumerate(labels):
            local_to_full[i] = np.random.randint(labels_dict[label][0], labels_dict[label][1], 1)

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
    print('get_features_remote_syn')
    id_tensor = id_tensor.tousertensor()
    features = g.ndata['features']
    labels = g.ndata['labels']
    pb = g.get_partition_book()
    part_id = pb.part_id
    machine_id = pb.to_partid(id_tensor)
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
            local_id = pb.to_local(partial_id)
        else:  # pull data from remote server
            request = SynFeatRequest(partial_id, labels.name, rate)
            rpc.send_request_to_machine(machine_idx, request)
            pull_count += 1
        start += count[idx]
    # recv response
    response_list = []
    if local_id is not None:  # local pull
        local_data = features.local_partition[local_id]
        server_id = part_id
        local_response = PullResponse(server_id, local_data)
        response_list.append(local_response)
    # wait response from remote server nodes
    for _ in range(pull_count):
        remote_response = rpc.recv_response()
        response_list.append(remote_response)
    # sort response by server_id and concat tensor
    response_list.sort(key=take_id)
    seq = []
    for response in response_list:
        data = response.data_tensor
        if isinstance(response, PullResponse):
            seq = seq.append(data)
        elif isinstance(response, SynFeatResponse):
            local_to_full = response.local_to_full
            data_full = [data[i] for i in local_to_full]
            seq = seq.append(data_full)
    data_tensor = F.cat(seq=seq, dim=0)
    return data_tensor[back_sorted_id]  # return data with original index order


def init_remote():
    print("init_remote1")
    MY_SYN_PULL = 638408
    rpc.register_service(MY_SYN_PULL, SynFeatRequest,
                         SynFeatResponse)
    print("init_remote2")
