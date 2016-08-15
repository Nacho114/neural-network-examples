"""
network_m.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.
Matrix based implementation.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

    ### Note: before matrix_backprop is called the set_matrix_biases has to be
    ###       called.
class Network_m(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.matrix_biases = None # set with set_matrix_biases

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    # Matrix based SGD
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        self.set_matrix_biases(mini_batch_size)
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    # SGD that returns the final performance after the
    def SGD_GET_DATA(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """SGD to test performance"""
        self.set_matrix_biases(mini_batch_size)
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if j == epochs - 1: return self.evaluate(test_data)*1. / n_test


    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using matrix backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        # matrix_backprop returns the cumulative partials for the samples
        # unzip the tuples of X & Y
        x, y = zip(*mini_batch)
        nabla_b, nabla_w = self.backprop(x, y)

        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    # For nabla_b & nabla_w we will sum up the samples for the aggregate values
    # which are averaged out in the caller function (mini_batch something)
    # Note: we need the set_matrix_biases to be called before calling
    # matrix_backprop.
    def backprop(self, x, y):
        """Matrix based implementation of backprop"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        n_samples = len(x)
        # Turn list of vectors into corresponding matrices
        X = np.column_stack(x)
        Y = np.column_stack(y)

        # feedforward (z is now a matrix where each column
        # is the z as in the simple implementation)
        activation = X
        activations = [X] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.matrix_biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], Y) * \
            sigmoid_prime(zs[-1])
        # suming up the nabla_b for all samples
        nabla_b[-1] = delta.sum(axis=1).reshape(delta.shape[0], 1)

        # Need to take outer product between each column of delta
        # and activations to get the cummulative nable_w
        for i in xrange(0, n_samples):
            nabla_w[-1] += outer_product(delta[:, [i]], (activations[-2])[:, [i]])

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta.sum(axis=1).reshape(delta.shape[0], 1)
            for i in xrange(0, n_samples):
                nabla_w[-l] += outer_product(delta[:, [i]], (activations[-l-1])[:, [i]])

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def set_matrix_biases(self, mini_batch_size):
        """Sets the matrix biases for the matri implemention"""
        self.matrix_biases = [np.tile(b, (1, mini_batch_size))
                             for b in self.biases]

    def reset(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def outer_product(x, y):
    """The outer product of vectors x and y"""
    return np.dot(x, y.transpose())
