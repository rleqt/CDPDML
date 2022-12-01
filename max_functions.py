import numpy as np
from numba import cuda, njit, prange, float32
import timeit
import os
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

def max_cpu(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """
    newMat = np.zeros((1000, 1000))
    for i in range(1000):
        for j in range(1000):
            newMat[i][j] = max(A[i][j], B[i][j])
    return newMat


@njit(parallel=True)
def max_numba(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """
    newMat = np.zeros((1000, 1000))
    for i in prange(1000):
        for j in prange(1000):
            newMat[i][j] = max(A[i][j], B[i][j])
    return newMat


def max_gpu(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """
    C = np.zeros((1000, 1000))
    a_gpu = cuda.to_device(A)
    b_gpu = cuda.to_device(A)
    c_gpu = cuda.to_device(C)
    max_kernel[1000, 1000](a_gpu, b_gpu, c_gpu)
    return c_gpu.copy_to_host()


@cuda.jit
def max_kernel(A, B, C):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    C[bid][tid] = np.max(A[bid][tid], B[bid][tid])
    cuda.syncthreads()


# this is the comparison function - keep it as it is.
def max_comparison():
    A = np.random.randint(0, 256, (1000, 1000))
    B = np.random.randint(0, 256, (1000, 1000))

    def timer(f):
        return min(timeit.Timer(lambda: f(A, B)).repeat(3, 20))

    # print('     [*] CPU:', timer(max_cpu))
    print('     [*] Numba:', timer(max_numba))
    print('     [*] CUDA:', timer(max_gpu))


if __name__ == '__main__':
    max_comparison()
