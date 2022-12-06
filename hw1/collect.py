import gzip
import pickle

import numpy as np


def load_mnist():
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        datasets = pickle.load(f, encoding='latin1')

    return [vectorized_results(data) for data in datasets]

def vectorized_results(data):
    def one_hot(x):
        e = np.zeros(10)
        e[x] = 1.0
        return e

    labels = [one_hot(x) for x in data[1]]
    return data[0], np.array(labels)

