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
        batches_to_serve = self.number_of_batches // self.num_workers
        batches_left = self.number_of_batches % self.num_workers
        worker_rank = self.rank - self.num_masters # relative index among the workers
        if worker_rank < batches_left: # the first workers should do the remainder
            batches_to_serve += 1
        self.number_of_batches = batches_to_serve

        with open(str(self.rank) + ".log", "a") as file:
            file.write("about to train on " + str(self.number_of_batches) + " batches \n")
        for epoch in range(self.epochs):
            # creating batches for epoch
            data = training_data[0]
            labels = training_data[1]
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
                        with open(str(self.rank) + ".log", "a") as file:
                            file.write("sending to master\n")
                        # recieve new self.weight and self.biases values from masters
                        requests += [self.comm.Irecv(self.biases[layer], source=i, tag=layer)]
                        requests += [self.comm.Irecv(self.weights[layer], source=i, tag=layer + self.num_layers)]
                MPI.Request.Waitall(requests)  # waiting for all requests to complete
                with open(str(self.rank) + ".log", "a") as file:
                    file.write("got information\n")
        with open(str(self.rank) + ".log", "a") as file:
            file.write("im done here...\n")


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
                with open(str(self.rank) + ".log", "a") as file:
                    file.write("waiting for worker\n")
                stat = MPI.Status()
                # "busy waiting" until a worker waits to be served
                while not self.comm.Iprobe(status = stat):
                    pass
                
                requests = []
                curr_worker = stat.Get_source()
                for curr, layer in enumerate(range(self.rank, self.num_layers, self.num_masters)):
                    requests += [self.comm.Irecv(nabla_b[curr], source=curr_worker, tag=layer)]
                    requests += [self.comm.Irecv(nabla_w[curr], source=curr_worker, tag=layer + self.num_layers)]
                MPI.Request.Waitall(requests) # waiting for all requests
                with open(str(self.rank) + ".log", "a") as file:
                    file.write("got info from worker " + str(curr_worker) + "\n")
                    file.write("calculating...\n")

                # calculate new weights and biases (of layers in charge)
                for i, dw, db in zip(range(self.rank, self.num_layers, self.num_masters), nabla_w, nabla_b):
                    self.weights[i] = self.weights[i] - self.eta * dw
                    self.biases[i] = self.biases[i] - self.eta * db

                # send new values (of layers in charge)
                with open(str(self.rank) + ".log", "a") as file:
                    file.write("sending updated info for worker " + str(curr_worker) + "\n")
                responses = []
                for layer in range(self.rank, self.num_layers, self.num_masters):
                    responses += [self.comm.Isend(self.biases[layer], dest=curr_worker, tag=layer)]
                    responses += [self.comm.Isend(self.weights[layer], dest=curr_worker, tag=layer + self.num_layers)]
                MPI.Request.Waitall(responses) # waiting for all requests

            self.print_progress(validation_data, epoch)

        # gather relevant weight and biases to process 0
        # TODO: add your code
        with open(str(self.rank) + ".log", "a") as file:
            file.write("sending final info to process 0...\n")
        for layer in range(self.rank, self.num_layers, self.num_masters):
            responses += [self.comm.Isend(self.biases[layer], dest=0, tag=layer)]
            responses += [self.comm.Isend(self.weights[layer], dest=0, tag=layer + self.num_layers)] 
