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
        # TODO: add your code
        batches_to_serve = self.num_workers // self.number_of_batches
        batches_left = self.num_workers % self.number_of_batches
        worker_rank = self.rank - self.num_masters # relative index among the workers
        if worker_rank < batches_left: # the first workers should do the remainder
            batches_to_serve += 1
        self.number_of_batches = batches_to_serve

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

                with open("shakom.txt", "a") as file:
                    file.write(str(self.rank - self.num_masters) + ": nabla_b:")
                    for i in nabla_b:
                        file.write(str(i) + ", ")
                    file.write(str(self.rank - self.num_masters) + ": nabla_w:")
                    for i in nabla_w:
                        file.write(str(i) + ", ")
                    file.write("\n\n\n\n")

                requests = []
                # indices = np.arange(self.num_layers)
                for i in range(self.num_masters):
                    for layer in range(i, self.num_layers, self.num_masters):
                        requests += [self.comm.Isend(nabla_b[layer], dest=i, tag=layer)]
                        requests += [self.comm.Isend(nabla_w[layer], dest=i, tag=layer)]
                    # requests += [self.comm.Isend(nabla_b[indices % self.num_masters == i], dest=i, tag=1)] #tag is 1 for updating b
                    # requests += [self.comm.Isend(nabla_w[indices % self.num_masters == i], dest=i, tag=2)] #tag is 2 for updating w
                with open("out.out", "a") as file:
                    file.write("sent")
                # recieve new self.weight and self.biases values from masters
                # TODO: add your code
                responses = []
                # indices = np.arange(self.biases.shape)
                # for i in range(self.num_masters):
                #     # updating biases and weights according to master with rank i
                #     responses += [self.comm.Irecv(self.biases[indices % self.num_masters == i], i, 3)]
                #     responses += [self.comm.Irecv(self.weights[indices % self.num_masters == i], i, 4)]
                MPI.Request.Waitall(requests)  # waiting for all requests to send
                MPI.Request.Waitall(responses) # waiting for all responses to send


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

                stat = MPI.Status()
                # "busy waiting" until a worker waits to be served
                while not self.comm.Iprobe(status = stat):
                    pass
                
                responses = []
                curr_worker = stat.Get_source()
                curr = 0
                for layer in range(self.rank, self.num_layers, self.num_masters):
                    responses += [self.comm.Irecv(nabla_b[curr], source=curr_worker, tag=layer)]
                    responses += [self.comm.Irecv(nabla_w[curr], source=curr_worker, tag=layer)]
                    curr += 1
                MPI.Request.Waitall(responses) # waiting for all requests

                # calculate new weights and biases (of layers in charge)
                for i, dw, db in zip(range(self.rank, self.num_layers, self.num_masters), nabla_w, nabla_b):
                    self.weights[i] = self.weights[i] - self.eta * dw
                    self.biases[i] = self.biases[i] - self.eta * db

                # send new values (of layers in charge)
                # TODO: add your code
                for layer in range(self.rank, self.num_layers, self.num_masters):
                    responses += [self.comm.Irecv(nabla_b[curr], source=curr_worker, tag=layer)]
                    responses += [self.comm.Irecv(nabla_w[curr], source=curr_worker, tag=layer)]
                    curr += 1
                MPI.Request.Waitall(responses) # waiting for all requests

            self.print_progress(validation_data, epoch)

        # gather relevant weight and biases to process 0
        # TODO: add your code
