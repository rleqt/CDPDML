from utils import *
import random


class NeuralNetwork(object):

    def __init__(self, sizes=None, learning_rate=1.0, mini_batch_size=16, number_of_batches=16,
                 epochs=10, matmul=np.matmul):
        """Initialize a Neural Network model.

        Parameters
        ----------
        sizes : list
            A list of integers specifying number of neurons in each layer.

        learning_rate : float
            Learning rate for gradient descent optimization. Defaults to 1.0

        mini_batch_size : int
            Size of each mini batch of training examples as used by Stochastic
            Gradient Descent. Denotes after how many examples the weights
            and biases would be updated. Default size is 16.
            
        number_of_batches: int
            Number of batches in an epoch.

        """

        sizes = [] if not sizes else sizes

        # Input layer is layer 0, followed by hidden layers layer 1, 2, 3...
        self.matmul = matmul
        self.sizes = sizes
        self.num_layers = len(sizes)

        # First term corresponds to layer 0 (input layer). No weights enter the
        # input layer and hence self.weights[0] is redundant.
        self.weights = [np.array([0])] + random_weights(sizes)

        # Input layer does not have any biases. self.biases[0] is redundant.
        self.biases = zeros_biases(sizes)

        # Input layer has no weights, biases associated. Hence z = wx + b is not
        # defined for input layer. self.zs[0] is redundant.
        self.zs = zeros_biases(sizes)

        # Training examples can be treated as activations coming out of input
        # layer. Hence self.activations[0] = (training_example).
        self.activations = zeros_biases(sizes)

        self.mini_batch_size = mini_batch_size
        self.number_of_batches = number_of_batches
        self.epochs = epochs
        self.eta = learning_rate

    def fit(self, training_data, validation_data=None):
        """Fit (train) the Neural Network on provided training data. Fitting is
        carried out using Stochastic Gradient Descent Algorithm.

        Parameters
        ----------
        training_data : tuple of lists
            A tuple of (training images array, image lables array)

        validation_data : tuple of lists, optional
            Same as `training_data`, if provided, the network will display
            validation accuracy after each epoch.

        """
        for epoch in range(self.epochs):
            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size)

            for x, y in mini_batches:
                self.forward_prop(x)
                nabla_b, nabla_w = self.back_prop(y)

                self.weights = [w - self.eta * dw for w, dw in zip(self.weights, nabla_w)]
                self.biases = [b - self.eta * db for b, db in zip(self.biases, nabla_b)]

            if validation_data:
                accuracy = self.validate(validation_data) / 100.0
                print("Epoch {0}, accuracy {1} %.".format(epoch + 1, accuracy))
            else:
                print("Processed epoch {0}.".format(epoch))

    def validate(self, validation_data):
        """Validate the Neural Network on provided validation data. It uses the
        number of correctly predicted examples as validation accuracy metric.

        Parameters
        ----------
        validation_data : tuple of lists

        Returns
        -------
        int
            Number of correctly predicted images.

        """
        data = validation_data[0]
        labels = validation_data[1]
        validation_results = self.predict(data) == np.argmax(labels, axis=1)
        return np.sum(validation_results)

    def predict(self, x):
        """Predict the label of a single test example (image).

        Parameters
        ----------
        x : numpy.array

        Returns
        -------
        int
            Predicted label of example (image).

        """

        self.forward_prop(x)
        return np.argmax(self.activations[-1], axis=1)

    def forward_prop(self, x):
        self.activations[0] = x
        for i in range(1, self.num_layers):
            self.zs[i] = self.matmul(self.activations[i - 1], self.weights[i]) + self.biases[i]
            self.activations[i] = sigmoid(self.zs[i])

    def back_prop(self, y):
        nabla_b = zeros_biases(self.sizes)
        nabla_w = [np.array([0])] + zeros_weights(self.sizes)

        error = (self.activations[-1] - y) * sigmoid_prime(self.zs[-1])
        nabla_b[-1] = np.sum(error, axis=0)
        nabla_w[-1] = self.matmul(self.activations[-2].T, error)

        for l in range(self.num_layers - 2, 0, -1):
            error = self.matmul(error, self.weights[l + 1].T) * sigmoid_prime(self.zs[l])
            nabla_b[l] = np.sum(error, axis=0)
            nabla_w[l] = self.matmul(self.activations[l - 1].T, error)

        return nabla_b, nabla_w

    def create_batches(self, data, labels, batch_size):
        """
         Parameters
         ----------
         data : np.array of input data
         labels : np.array of input labels
         batch_size : int size of batch

         Returns
         -------
         list
             list of tuples of (data batch of batch_size, labels batch of batch_size)

        """
        batches = []
        for k in range(self.number_of_batches):
            indexes = random.sample(range(0, data.shape[0]), batch_size)
            batches.append((data[indexes], labels[indexes]))
        return batches
        

