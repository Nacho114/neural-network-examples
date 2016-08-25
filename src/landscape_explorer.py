
import numpy as np
import theano
import theano.tensor as T


class landscape_explorer(object):


    def __init__(self, dim, n_samples, error=0.001):
        self.dim = dim
        self.n_samples = n_samples
        self.w0 = generate_w0(dim, length, error)
        self.weights = np.random.randn(1, dim)

        T.sigmoid(X)


    def generate_random_training_set(n_samples, dim, w0):
        X = [np.random.randn(dim, 1) for x in range(n_samples)]
        Pr_Y = [sigmoid(w0.dot(xi)) for xi in X]
        Y = [np.asarray([np.random.binomial(1, pr_y)]).reshape(1, 1) for pr_y in Pr_Y]
        return zip(X, Y)

    """Generate true weights of dimension dim and length len
    """
    def generate_w0(dim, length, error=0.001):
        while (True):
            test = np.random.randn(dim, 1)
            if (l2_norm(test) < length + error and
                    l2_norm(test) > length - error):
                return test.reshape(1, dim)

    def l2_norm(w0):
        return np.linalg.norm(w0)

    def get_B0(w0):
        return 2 * l2_norm(w0)
