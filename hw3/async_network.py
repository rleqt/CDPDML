from network import *
import itertools
import sys
import numpy as np
import math
import mpi4py
from time import time_ns

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
        batches_to_serve = self.number_of_batches // self.num_workers
        batches_left = self.number_of_batches % self.num_workers
        worker_rank = self.rank - self.num_masters # relative index among the workers
        if worker_rank < batches_left: # the first workers should do the remainder
            batches_to_serve += 1
        self.number_of_batches = batches_to_serve

        for epoch in range(self.epochs):
            # creating batches for epoch
            data = training_data[0]
            labels = training_data[1]
            # creating batches (considering the batches number we've figured earlier)
            mini_batches = self.create_batches(data, labels, self.mini_batch_size)
            for x, y in mini_batches:
                # do work - don't change this
                self.forward_prop(x)
                nabla_b, nabla_w = self.back_prop(y)

                requests = []
                
                for i in range(self.num_masters):
                    for layer in range(i, self.num_layers, self.num_masters):
                        # send nabla_b, nabla_w to masters 
                        requests += [self.comm.Isend(nabla_b[layer], dest=i, tag=layer)]
                        requests += [self.comm.Isend(nabla_w[layer], dest=i, tag=layer + self.num_layers)]

                        # recieve new self.weight and self.biases values from masters
                        requests += [self.comm.Irecv(self.biases[layer], source=i, tag=layer)]
                        requests += [self.comm.Irecv(self.weights[layer], source=i, tag=layer + self.num_layers)]
                MPI.Request.Waitall(requests)  # waiting for all requests to complete


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
                
                stat = MPI.Status()
                # "busy waiting" until a worker waits to be served
                while not self.comm.Iprobe(status = stat):
                    pass
                
                requests = []
                curr_worker = stat.Get_source()
                # get the nabla_w, nabla_b for the master's layers (one layer at a time)
                for curr, layer in enumerate(range(self.rank, self.num_layers, self.num_masters)):
                    requests += [self.comm.Irecv(nabla_b[curr], source=curr_worker, tag=layer)]
                    requests += [self.comm.Irecv(nabla_w[curr], source=curr_worker, tag=layer + self.num_layers)]
                MPI.Request.Waitall(requests) # waiting for all requests

                # calculate new weights and biases (of layers in charge)
                for i, dw, db in zip(range(self.rank, self.num_layers, self.num_masters), nabla_w, nabla_b):
                    self.weights[i] = self.weights[i] - self.eta * dw
                    self.biases[i] = self.biases[i] - self.eta * db

                # send new values (of layers in charge)
                responses = []
                for layer in range(self.rank, self.num_layers, self.num_masters):
                    responses += [self.comm.Isend(self.biases[layer], dest=curr_worker, tag=layer)]
                    responses += [self.comm.Isend(self.weights[layer], dest=curr_worker, tag=layer + self.num_layers)]
                MPI.Request.Waitall(responses) # waiting for all requests before serving another worker

            self.print_progress(validation_data, epoch)

        # gather relevant weight and biases to process 0
        if self.rank == 0:
            # we want to gather only layers from other masters
            layers_to_gather = [layer for layer in range(0, self.num_layers) if layer not in range(self.rank, self.num_layers, self.num_masters)]
            
            # sending requests to get all the layers...
            responses = []
            for layer in layers_to_gather:
                responses += [self.comm.Irecv(self.biases[layer], source=MPI.ANY_SOURCE, tag=layer + 2 * self.num_layers)]
                responses += [self.comm.Irecv(self.weights[layer], source=MPI.ANY_SOURCE, tag=layer + 3 * self.num_layers)]
            MPI.Request.Waitall(responses)
            
        else:
            # sending the current master's layers to process 0
            responses = []
            for layer in range(self.rank, self.num_layers, self.num_masters):
                responses += [self.comm.Isend(self.biases[layer], dest=0, tag=layer + 2 * self.num_layers)]
                responses += [self.comm.Isend(self.weights[layer], dest=0, tag=layer + 3 * self.num_layers)]
            MPI.Request.Waitall(responses)