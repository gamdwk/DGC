import sysv_ipc
import dgl
import pickle
from sys import getsizeof

syn_feat = []
syn_label_indices = {}


"""def init_memory(key, value):
    size = getsizeof(value) * 1024 * 1024 * 1024 * 1024
    memory = sysv_ipc.SharedMemory(key, flag=sysv_ipc.IPC_CREAT | sysv_ipc.IPC_EXCL, size=size)
    # memory.attach()
    value = pickle.dumps(value)
    memory.write(value)


def set_memory(key, value):
    try:
        memory = sysv_ipc.SharedMemory(key)
    except sysv_ipc.ExistentialError:
        init_memory(key, value)
        return
    # memory.attach()
    value = pickle.dumps(value)
    memory.write(value)
    # memory.detach()


def get_memory(key):
    try:
        memory = sysv_ipc.SharedMemory(key)
    except sysv_ipc.ExistentialError:
        return None
    # memory.attach()
    data = memory.read()
    data = pickle.loads(data)
    return data


def del_memory(memory):
    memory.detach()
    memory.remove()"""


def write_syn_feat(feat):
    global syn_feat
    syn_feat = feat
    # set_memory(27697, feat)


def read_syn_feat():
    global syn_feat
    return syn_feat
    # return get_memory(27687)


def write_syn_label_indices(indicates):
    # set_memory(25578, indicates)
    global syn_label_indices
    syn_label_indices = indicates


def read_syn_label_indices():
    # return get_memory(25578)
    global syn_label_indices
    return syn_label_indices
