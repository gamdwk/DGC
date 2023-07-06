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


def pull_handler(target, name, id_tensor):
    """Default handler for PULL operation.

    On default, _pull_handler perform gather_row() operation for the tensor.

    Parameters
    ----------
    target : tensor
        target tensor
    name : str
        data name
    id_tensor : tensor
        a vector storing the ID list.

    Return
    ------
    tensor
        a tensor with the same row size of ID.
    """
    # TODO(chao): support Tensorflow backend
    return target[name][id_tensor]


class SynFeatRequest(Request):
    def __getstate__(self):
        return self.labels, self.rate

    def __setstate__(self, state):
        self.labels, self.rate = state

    def process_request(self, server_state):
        # print(100)
        # print(self.__getstate__())
        # import time
        # t1 = time.time()
        kv_store = server_state.kv_store
        labels = self.labels
        labels_dict = {}
        # t6 = time.time()
        syn_label_indices = read_syn_label_indices()
        # print("rank:{},{}".format(kv_store.server_id, syn_label_indices))
        # t9 = time.time()
        syn_feat = read_syn_feat()
        # t7 = time.time()
        num_all = 0

        d = syn_feat.shape[1]

        c = Counter(labels.tolist())
        # t8 = time.time()
        # print(f"读ipc{t9-t6},{t7-t6},{t8-t7}")
        syn_ids = {}
        # print(c)
        # t2 = time.time()
        for i, (label, num) in enumerate(c.items()):
            num = num * self.rate

            num = int(num) if num > 1 else 1
            labels_dict[label] = (num_all, num_all + num)
            num_all += num
            # print("rank:{}".format(kv_store.server_id))
            syn_idx = np.random.randint(syn_label_indices[label][0], syn_label_indices[label][1], num)
            syn_ids[label] = syn_idx
        # t3 = time.time()
        # print(labels_dict)
        data = torch.zeros((num_all, d))
        for i, (label, num) in enumerate(c.items()):
            syn_idx = syn_ids[label]
            data[labels_dict[label][0]:labels_dict[label][1]] = syn_feat[syn_idx]
        # t4 = time.time()
        local_to_full = torch.zeros((len(self.labels),), dtype=torch.int64)
        # print(local_to_full)
        for i, label in enumerate(labels.tolist()):
            local_to_full[i] = torch.tensor(np.random.randint(labels_dict[label][0], labels_dict[label][1], 1),
                                            dtype=torch.int64)
        # print(local_to_full)
        # t5 = time.time()
        # print(f"t:循环1:{t3-t2},2:{t4-t3},3:{t5-t4},长度{len(labels)}动态{len(data)}")
        res = SynFeatResponse(kv_store.server_id, data, local_to_full)
        # print("返回请求", time.time() - t1)
        # print(res)
        return res

    def __init__(self, labels, rate):
        self.labels = labels
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
    # print(1)
    # torch.set_printoptions(threshold=np.inf,profile="full")
    id_tensor = dgl.utils.toindex(id_tensor)
    id_tensor = id_tensor.tousertensor()
    features = g.ndata['features']

    labels = g.ndata['labels']
    part_policy = labels.part_policy
    # labels_name = labels._name
    # client = get_kvstore()

    pb = g.get_partition_book()

    # g.get_node_partition_policy(ntype=g.ntypes[0])
    part_id = pb.partid
    # print(part_policy.to_partid(1288835))
    # print(pb.nid2partid(1288835))
    # torch.save(id_tensor, f"id_tensor_{part_id}.pt")
    machine_id = part_policy.to_partid(id_tensor)
    # torch.save(machine_id, f"machine_id_{part_id}.pt")
    sorted_id = F.tensor(np.argsort(F.asnumpy(machine_id)))
    id_tensor = id_tensor[sorted_id]
    back_sorted_id = F.tensor(np.argsort(F.asnumpy(sorted_id)))
    machine, count = np.unique(F.asnumpy(machine_id), return_counts=True)
    # torch.save(machine, f"machine_{part_id}.pt")
    start = 0
    pull_count = 0
    local_id = None
    """print(part_id)
    print(g.rank())
    print(machine)"""
    for idx, machine_idx in enumerate(machine):
        end = start + count[idx]
        if start == end:  # No data for target machine
            continue
        partial_id = id_tensor[start:end]
        """print("machine_idx{},part_id{},partial_id{}".format(machine_idx, part_id, partial_id))
        print("in")"""
        # print(f"机器{machine_idx}获取{len(partial_id)}个节点，总共{len(id_tensor)}")
        # print(part_id, machine_idx, partial_id)
        # torch.save(partial_id, f"{part_id}get{machine_idx}.pt")
        if machine_idx == part_id:  # local pull
            # Note that DO NOT pull local data right now because we can overlap
            # communication-local_pull here
            # local_id = part_policy.to_local(partial_id)
            # local_id = features.part_policy.to_local(partial_id)
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
            # print(2)

            # print(partial_id)
            # torch.save(partial_id, f"idx{part_id}.pt")
            idx_labels = labels[partial_id]
            # print(idx_labels)
            # torch.save(idx_labels, f"labels{part_id}.pt")
            # print(partial_id[torch.nonzero(idx_labels == 45)])
            x = partial_id[torch.nonzero(idx_labels == 45)]

            # print(x)
            # if x.shape[0] > 0:
            #    print(part_policy.to_partid(x[0]))
            # print(idx_labels)
            request = SynFeatRequest(idx_labels, rate)
            # print(request)
            rpc.send_request_to_machine(machine_idx, request)
            pull_count += 1
        start += count[idx]
    # recv response
    response_list = []
    # wait response from remote server nodes
    for _ in range(pull_count):
        remote_response = rpc.recv_response()
        response_list.append(remote_response)
    if local_id is not None:  # local pull

        # local_data = features.local_partition[local_id]
        # local_data = features[local_id]
        server_id = part_id
        from dgl.distributed import DistTensor

        local_data = features[local_id]
        local_response = PullResponse(server_id, local_data)
        # print(8)
        response_list.append(local_response)
    # print(6)
    # sort response by server_id and concat tensor
    response_list.sort(key=take_id)

    def handle(response):
        data = response.data_tensor
        s_id = response.server_id
        if s_id == part_id or isinstance(response, PullResponse):
            assert s_id == part_id
            # print("s_id{},part_id{}".format(s_id, part_id))
            return data
        else:
            local_to_full = response.local_to_full
            # print(local_to_full)
            data_full = data[local_to_full]
            return data_full

    # print(7)
    data_tensor = F.cat(seq=[handle(response) for response in response_list], dim=0)
    return data_tensor[back_sorted_id]  # return data with original index order


def init_remote():
    MY_SYN_PULL = 61120
    rpc.register_service(MY_SYN_PULL, SynFeatRequest,
                         SynFeatResponse)


def init_g_features(g):
    name = g.ndata["features"]._name
    get_kvstore().register_pull_handler(name, pull_handler)


if __name__ == '__main__':
    init_remote()
