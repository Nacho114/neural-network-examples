
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

class Network_m(object):

    def __init__(self, sizes, B0):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.B0 = B0

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for w in self.weights:
            a = sigmoid(np.dot(w, a))
        return a

    def GD(self, training_data, epochs, eta=1., test_data=None):
        if test_data: n_test = len(test_data)

        for j in xrange(epochs):
            self.update_mini_batch(training_data, eta)
            if test_data and False:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)

        return self.weights

    # Matrix based SGD
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
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
        nabla_w = self.backprop(x, y)
        self.weights = [w-(eta*1./len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]

        #If w gets out of the B0 ball, project it back.
        if l2_norm(self.weights) >= self.B0:
            self.weights = 1. / l2_norm(self.weights) * self.weights


    # For nabla_b & nabla_w we will sum up the samples for the aggregate values
    # which are averaged out in the caller function (mini_batch something)
    # Note: we need the set_matrix_biases to be called before calling
    # matrix_backprop.
    def backprop(self, x, y):
        """Matrix based implementation of backprop"""
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
        for w in self.weights:
            z = np.dot(w, activation)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], Y) * \
            sigmoid_prime(zs[-1])

        # Need to take outer product between each column of delta
        # and activations to get the cummulative nable_w
        for i in xrange(0, n_samples):
            nabla_w[-1] += outer_product(delta[:, [i]], (activations[-2])[:, [i]])

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            for i in xrange(0, n_samples):
                nabla_w[-l] += outer_product(delta[:, [i]], (activations[-l-1])[:, [i]])
        return nabla_w

        # Need to take into account new strucutre of conditional probability
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        test_results = [(np.random.binomial(1, self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def reset(self):
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

def l2_norm(w0):
    return np.linalg.norm(w0)
