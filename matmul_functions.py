import numpy as np
from numba import njit, cuda
import timeit


def rowMult(X, i, j):
    sum1 = 0
    for index in range(X.shape[1]):
        sum1 = sum1 + X[i][index]*X[j][index]
    return sum1

def matmul_transpose_trivial(X):
    outcome = np.zeros_like(X)
    for i in range(outcome.shape[0]):
        for j in range(outcome.shape[1]):
            outcome[i][j] = rowMult(X, i, j)

    return outcome


@njit
def matmul_transpose_numba(X):
    outcome = np.zeros_like(X)
    for i in range(outcome.shape[0]):
        for j in range(outcome.shape[1]):
            for index in range(X.shape[1]):
                outcome[i][j] = outcome[i][j] + X[i][index]*X[j][index]

    return outcome


def matmul_transpose_gpu(X):
    raise NotImplementedError("To be implemented")


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
