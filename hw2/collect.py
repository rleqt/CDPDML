import gzip
import pickle
import os
import wget

import numpy as np


def load_mnist():
    if not os.path.exists(os.path.join(os.curdir, 'data')):
        os.mkdir(os.path.join(os.curdir, 'data'))
        wget.download('http://deeplearning.net/data/mnist/mnist.pkl.gz', out='data')
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        datasets = pickle.load(f, encoding='latin1')

    training_data, validation_data, test_data = [vectorized_results(data) for data in datasets]

    # changing the training set to be smaller (500) to fit the exercise.
    test_data = np.concatenate((test_data[0], training_data[0][500:])), np.concatenate((test_data[1], training_data[1][500:]))
    training_data = training_data[0][:500], training_data[1][:500]
    return training_data, validation_data, test_data


def vectorized_results(data):
    def one_hot(x):
        e = np.zeros(10)
        e[x] = 1.0
        return e
    labels = [one_hot(x) for x in data[1]]
    return data[0], np.array(labels)
