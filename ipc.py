import time

import sysv_ipc
import dgl
import pickle
from sys import getsizeof
import threading
syn_feat = []
syn_label_indices = {}

from threading import Semaphore

thread_local = threading.local()
def init_memory(key, value):
    size = 2 * 1024 * 1024 * 1024
    memory = sysv_ipc.SharedMemory(key, flags=sysv_ipc.IPC_CREAT | sysv_ipc.IPC_EXCL, size=size)
    # memory.attach()
    value = pickle.dumps(value)
    memory.write(value)


def set_memory(key, value):
    try:
        memory = sysv_ipc.SharedMemory(key)
    except:
        init_memory(key, value)
        return
    # memory.attach()
    value = pickle.dumps(value)
    memory.write(value)
    # memory.detach()


def get_memory(key):
    try:
        memory = sysv_ipc.SharedMemory(key)
    except:
        return None
    # memory.attach()
    try:
        data = memory.read()
        data = pickle.loads(data)
    except:
        data = None
    return data


def del_memory(memory):
    memory.detach()
    memory.remove()


def write_syn_feat(feat):
    """global syn_feat
    syn_feat = feat
    print("write_syn_feat{}".format(syn_feat))"""
    set_memory(27877, feat.cpu())


def read_syn_feat():
    """global syn_feat
    return syn_feat"""
    return get_memory(27877)


def write_syn_label_indices(indicates):
    set_memory(25578, indicates)
    """global syn_label_indices
    syn_label_indices = indicates"""


def read_syn_label_indices():
    return get_memory(25578)
    """global syn_label_indices
    return syn_label_indices"""


def write_model(model):
    set_memory(28567, model)
    """global syn_label_indices
    syn_label_indices = indicates"""


def read_model():
    return get_memory(28567)


def del_model():
    try:
        memory = sysv_ipc.SharedMemory(27877)
        del_memory(memory)
    except:
        pass


def del_syn():
    try:
        memory = sysv_ipc.SharedMemory(28567)
        del_memory(memory)
    except:
        pass
    finally:
        try:
            memory = sysv_ipc.SharedMemory(25578)
            del_memory(memory)
        except:
            pass


def wait_for_model():
    #x = None
    while read_model() is None:
        time.sleep(2)
        #x = read_model()
        #print(x)


def wait_for_syn():
    #x = None
    while read_syn_label_indices() is None:
        time.sleep(2)
        #print(x)
        pass
    #y = None
    while read_syn_feat() is None:
        time.sleep(2)
        #y = read_syn_feat()
        #print(y)
        pass


if __name__ == '__main__':
    """set_memory(21111, 7218)
    import time

    t = time.time()
    get_memory(21111)
    t2 = time.time()
    print(t2 - t)"""
    del_syn()
    del_model()
    """write_model(12)
    write_syn_label_indices(2)
    write_syn_feat(22)"""
