from collect import *
from ip_network import *
from time import time


if __name__ == '__main__':

    # initialize training, validation and testing data
    training_data, validation_data, test_data = load_mnist()

    # initialize layer sizes as list
    layers = [784, 128, 64, 10]

    # initialize learning rate
    learning_rate = 0.1
    
    # initialize mini batch size
    mini_batch_size = 16
    
    number_of_batches = 300

    # initialize epoch
    epochs = 15
    start1 = time()

    # initialize neural-net
    nn = NeuralNetwork(layers, learning_rate, mini_batch_size, number_of_batches, epochs)

    # training neural network
    nn.fit(training_data, validation_data)

    stop1 = time()

    print('Time regular:', stop1 - start1)

    # testing neural network
    accuracy = (nn.validate(test_data) / len(test_data[0])) * 100
    print("Test Accuracy: " + str(accuracy) + "%")
    
    start1 = time()
    # initialize neural-net
    nn = IPNeuralNetwork(layers, learning_rate, mini_batch_size, number_of_batches, epochs)
    
    # training neural network
    nn.fit(training_data, validation_data)
    stop1 = time()
    print('Time with image processing:', stop1 - start1)

    # testing neural network
    accuracy = (nn.validate(test_data) / len(test_data[0])) * 100
    print("Test Accuracy: " + str(accuracy) + "%")
   

