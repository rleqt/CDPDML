import numpy as np
from numba import njit, cuda, prange
import timeit

def matmul_transpose_trivial(X):
    row = X.shape[0]
    col = X.shape[1]
    outcome = np.zeros((row,row))
    for i in range(row):
        for j in range(row):
            for index in range(col):
                outcome[i,j] += X[i, index]* X[j, index]
    return outcome

@njit(parallel=True)
def matmul_transpose_numba(X):
    row = X.shape[0]
    col = X.shape[1]
    outcome = np.zeros((row,row))
    for i in prange(row):
        for j in prange(row):
            # X_(i,*) * X.t_(*,j) = X_(i,*) * X_(j,*)
            for index in prange(col):
                outcome[i,j] += X[i, index]* X[j, index]
    return outcome


def matmul_transpose_gpu(X):
    row = x.shape[0]
    x_gpu = cuda.to_device(X)
    c_gpu = cuda.to_device(np.zeros((row,row)))
    matmul_kernel[1,1024](x_gpu)
    return c_gpu.copy_to_host()


@cuda.jit
def matmul_kernel(A, C):
    raise NotImplementedError("To be implemented")


# this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()

    def timer(f, functionParameters):
        return min(timeit.Timer(lambda: f(X) if functionParameters == 1 else f(X, Xt)).repeat(3, 100))

    # print('Python:', timer(matmul_transpose_trivial, 1)) we will not consider this since it takes infinite time :)
    print('Numpy:', timer(np.matmul, 2))
    print('Numba:', timer(matmul_transpose_numba, 1))
    # print('CUDA:', timer(matmul_transpose_gpu, 1))


if __name__ == '__main__':
    matmul_comparison()
