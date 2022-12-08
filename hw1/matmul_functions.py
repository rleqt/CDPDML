import numpy as np
from numba import njit, cuda, prange
import timeit


def matmul_transpose_trivial(X):
    row = X.shape[0]
    col = X.shape[1]
    outcome = np.zeros((row, row))
    for i in range(row):
        for j in range(row):
            for index in range(col):
                outcome[i, j] += X[i, index] * X[j, index]
    return outcome


@njit(parallel=True)
def matmul_transpose_numba(X):
    row = X.shape[0]
    col = X.shape[1]
    outcome = np.zeros((row, row))
    for i in prange(row):
        for j in prange(row):
            # X_(i,*) * X.t_(*,j) = X_(i,*) * X_(j,*)
            for index in prange(col):
                outcome[i, j] += X[i, index] * X[j, index]
    return outcome


def matmul_transpose_gpu(X):
    row = X.shape[0]
    x_gpu = cuda.to_device(X)
    C = np.zeros((row ** 2))
    c_gpu = cuda.to_device(C)
    matmul_kernel[1, 1024](x_gpu, c_gpu)
    return c_gpu.copy_to_host().reshape((row, row))


@cuda.jit
def matmul_kernel(A, C):
    threadID = cuda.threadIdx.x
    finalSize = A.shape[0]
    CSize = finalSize**2
    threadCurrentx = threadID // finalSize
    threadCurrenty = threadID % finalSize
    access = threadCurrentx * finalSize + threadCurrenty
    while access < CSize:
        sum1 = 0
        if not (threadCurrenty > threadCurrentx):
            for i in range(A.shape[1]):
                sum1 += (A[threadCurrentx, i]*A[threadCurrenty, i])
            C[access] = sum1
            C[threadCurrenty * finalSize + threadCurrentx] = sum1
        threadID += 1024
        threadCurrentx = threadID // finalSize
        threadCurrenty = threadID % finalSize
        access = threadCurrentx * finalSize + threadCurrenty


# this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()

    def timer(f, functionParameters):
        return min(timeit.Timer(lambda: f(X) if functionParameters == 1 else f(X, Xt)).repeat(3, 100))

    print(np.array_equal(matmul_transpose_gpu(X), np.matmul(X, X.T)))
    # print('Python:', timer(matmul_transpose_trivial, 1)) we will not consider this since it takes infinite time :)
    print('Numpy:', timer(np.matmul, 2))
    print('Numba:', timer(matmul_transpose_numba, 1))
    print('CUDA:', timer(matmul_transpose_gpu, 1))


if __name__ == '__main__':
    matmul_comparison()
