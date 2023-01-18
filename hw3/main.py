import os
import sys

from collect import *
from network import *
from time import time
from sync_network import *
from async_network import *


def print_use():
    print("to run sync network use the command srun -K -c 4 -n X --pty python3 main.py sync")
    print("to run async network use the command srun -K  -c 4 -n X --pty python3 main.py async M")


num_args = len(sys.argv)
if num_args < 2 or num_args > 3:
    print_use()
    sys.exit()

# initialize layer sizes as list
layers = [784, 128, 64, 10]

# initialize learning rate
learning_rate = 0.1

# initialize mini batch size
mini_batch_size = 16
batch_size = 632

# initialize epoch

epochs = 5

# initialize training, validation and testing data
training_data, validation_data, test_data = load_mnist()

start1 = time()
# initialize neuralnet
nn = NeuralNetwork(layers, learning_rate, mini_batch_size, batch_size, epochs)

# training neural network
nn.fit(training_data, validation_data)

stop1 = time()

print('Time reg:', stop1 - start1)

# testing neural network

accuracy = nn.validate(test_data) / 100.0
print("Test Accuracy: " + str(accuracy) + "%")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if sys.argv[1] == "sync":
    start1 = time()
    # initialize neuralnet
    nn = SynchronicNeuralNetwork(layers, learning_rate, mini_batch_size, batch_size, epochs)

    # training neural network
    nn.fit(training_data, validation_data)

    stop1 = time()
    print('Time sync:', stop1 - start1)

    # testing neural network

    accuracy = nn.validate(test_data) / 100.0
    print("Test Accuracy: " + str(accuracy) + "%")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if sys.argv[1] == "async":
    if len(sys.argv) != 3:
        print_use()
        sys.exit()

    masters = int(sys.argv[2])

    start1 = time()
    # initialize neuralnet
    nn = AsynchronicNeuralNetwork(layers, learning_rate, mini_batch_size, batch_size, epochs, masters)

    # training neural network
    nn.fit(training_data, validation_data)

    stop1 = time()
    print('Time async:', stop1 - start1)

    # testing neural network

    accuracy = nn.validate(test_data) / 100.0
    print("Test Accuracy: " + str(accuracy) + "%")
