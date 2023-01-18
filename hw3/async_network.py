from network import *
import itertools
import sys
import numpy as np
import math
import mpi4py
from time import time

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI


class AsynchronicNeuralNetwork(NeuralNetwork):

    def __init__(self, sizes=list(), learning_rate=1.0, mini_batch_size=16, number_of_batches=16,
                 epochs=10, number_of_masters=1, matmul=np.matmul):
        # calling super constructor
        super().__init__(sizes, learning_rate, mini_batch_size, number_of_batches, epochs, matmul)
        # setting number of workers and masters
        self.num_masters = number_of_masters

    def fit(self, training_data, validation_data=None):
        # MPI setup
        MPI.Init()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
		self.num_workers = self.size - self.num_masters

        self.layers_per_master = self.num_layers // self.num_masters

        # split up work
        if self.rank < self.num_masters:
            self.do_master(validation_data)
        else:
            self.do_worker(training_data)

        # when all is done
        self.comm.Barrier()
        MPI.Finalize()

    def do_worker(self, training_data):
        """
        worker functionality
        :param training_data: a tuple of data and labels to train the NN with
        """
        # setting up the number of batches the worker should do every epoch
        # TODO: add your code

        for epoch in range(self.epochs):
            # creating batches for epoch
            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size)
            for x, y in mini_batches:
                # do work - don't change this
                self.forward_prop(x)
                nabla_b, nabla_w = self.back_prop(y)

                # send nabla_b, nabla_w to masters 
                # TODO: add your code

                # recieve new self.weight and self.biases values from masters
                # TODO: add your code

    def do_master(self, validation_data):
        """
        master functionality
        :param validation_data: a tuple of data and labels to train the NN with
        """
        # setting up the layers this master does
        nabla_w = []
        nabla_b = []
        for i in range(self.rank, self.num_layers, self.num_masters):
            nabla_w.append(np.zeros_like(self.weights[i]))
            nabla_b.append(np.zeros_like(self.biases[i]))

        for epoch in range(self.epochs):
            for batch in range(self.number_of_batches):

                # wait for any worker to finish batch and
                # get the nabla_w, nabla_b for the master's layers
                # TODO: add your code

                # calculate new weights and biases (of layers in charge)
                for i, dw, db in zip(range(self.rank, self.num_layers, self.num_masters), nabla_w, nabla_b):
                    self.weights[i] = self.weights[i] - self.eta * dw
                    self.biases[i] = self.biases[i] - self.eta * db

                # send new values (of layers in charge)
                # TODO: add your code

            self.print_progress(validation_data, epoch)

        # gather relevant weight and biases to process 0
        # TODO: add your code
