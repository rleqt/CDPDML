import sys
from time import time
import numpy as np
from my_naive_allreduce import *
from my_ring_allreduce import *
from mpi4py import MPI


la_comm = MPI.COMM_WORLD
ma_rank = la_comm.Get_rank()
la_size = la_comm.Get_size()


def _op(x, y):
    # TODO: add your code
    raise NotImplementedError()


for size in [2**12, 2**13, 2**14]:
    print("array size:", size)
    data = np.random.rand(size)
    res1 = np.zeros_like(data)
    res2 = np.zeros_like(data)
    start1 = time()
    allreduce(data, res1, la_comm, _op)
    end1 = time()
    print("naive impl time:", end1-start1)
    start1 = time()
    ringallreduce(data, res2, la_comm, _op)
    end1 = time()
    print("ring impl time:", end1-start1)
    assert np.allclose(res1, res2)
