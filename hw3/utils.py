import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def xavier_initialization(m, n):
    xavier = 1 / (m ** 0.5)
    return np.random.uniform(low=-xavier, high=xavier, size=(m, n))


def random_weights(_list):
    return [xavier_initialization(m, n) for m, n in zip(_list[:-1], _list[1:])]


def zeros_weights(_list):
    return [np.zeros((m, n)) for m, n in zip(_list[:-1], _list[1:])]


def zeros_biases(_list):
    return [np.zeros(n) for n in _list]


def create_batches(data, labels, batch_size):
    return [(data[k:k + batch_size], labels[k:k + batch_size]) for k in range(0, len(data), batch_size)]


def add_elementwise(list1, list2):
    return [x + y for x, y in zip(list1, list2)]
