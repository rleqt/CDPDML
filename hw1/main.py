# import os

from collect import *
from network import *
from time import time
from matmul_functions import *

# os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda-9.0/nvvm/lib64/libnvvm.so'
# os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda-9.0/nvvm/libdevice/'

# initialize layer sizes as list
layers = [784, 128, 64, 10]

# initialize learning rate
learning_rate = 0.1

# initialize mini batch size
mini_batch_size = 16

# initialize epoch

epochs = 5

# initialize training, validation and testing data
training_data, validation_data, test_data = load_mnist()

start1 = time()

# initialize neuralnet
nn = NeuralNetwork(layers, learning_rate, mini_batch_size, epochs)

# training neural network
nn.fit(training_data, validation_data)

stop1 = time()

print('Time matmul_np:', stop1 - start1)

''' Part 3
    add training of nn with matmul_numba and matmul_gpu after implementing them
'''
# testing neural network
accuracy = nn.validate(test_data) / 100.0
print("Test Accuracy: " + str(accuracy) + "%")
