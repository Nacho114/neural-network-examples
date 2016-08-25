import numpy as np
import network_landscape as network

dim = 10
n_samples = 1200
w0_len = 3
eta = 1.
epochs = 1000# number of iterations

dim_a = [20, 40, 80, 160, 320]
dim_b = [5, 10, 20, 40, 80]

"""Generate a random training set (X, Y) according to the Montanari paper
    dim = number of dimensions
    n_samples = number of samples
"""
def generate_random_training_set(n_samples, dim, w0):
    X = [np.random.randn(dim, 1) for x in range(n_samples)]
    Pr_Y = [sigmoid(w0.dot(xi)) for xi in X]
    Y = [np.asarray([np.random.binomial(1, pr_y)]).reshape(1, 1) for pr_y in Pr_Y]
    return zip(X, Y)

"""Generate true weights of dimension dim and length len
"""
def generate_w0(dim, len, error=0.001):
    while (True):
        test = np.random.randn(dim, 1)
        if (l2_norm(test) < len + error and
                l2_norm(test) > len - error):
            return test.reshape(1, dim)

def l2_norm(w0):
    return np.linalg.norm(w0)

def get_B0(w0):
    return 2 * l2_norm(w0)

def cost(w1, w2):
    return l2_norm(w1 - w2)


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


"""Generate a list of weights for a given data"""
def generate_weight_list(n_tests, net, training_data, epochs, eta, test_data=None):
    weight_list = []
    for _ in xrange(int(n_tests)):
        net.GD(training_data, epochs, eta)
        weight_list.append(net.weights[0])
        net.reset()
    return weight_list





w0 = generate_w0(dim, w0_len)
B0 = get_B0(w0)
training_data = generate_random_training_set(n_samples, dim, w0)

net = network.Network_m([dim, 1], B0)

n_tests = 2.
weight_list = generate_weight_list(n_tests, net, training_data, epochs, eta)
mean = sum(weight_list) / n_tests
var = sum([(w - mean)**2 for w in weight_list]) / n_tests
print mean
print var
