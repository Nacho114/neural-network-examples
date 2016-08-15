import mnist_loader
import network_m
import matplotlib.pyplot as plt
import math

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

### Hyper-parameters
net = network_m.Network_m([784, 30, 10])
mini_batch_size = 10
epochs = 5

### testing parameters
eta_0 = 0.1
eta_f = 2
eta_jump = 0.5
n_tests = 5 # how many tests are we averaging per eta

### Auxiliar functions
def train_data(epochs, eta):
    return net.SGD_GET_DATA(training_data, epochs,
                                mini_batch_size, eta, test_data=test_data)

def average_train_data(n_tests, epochs, eta):
    acc = 0
    for i in xrange(0, n_tests, 1):
        acc += train_data(epochs, eta)
    return acc * 1. / n_tests


# xrange for float values
def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

print 'running . . . '

learning_rates = frange(eta_0, eta_f, eta_jump)
l_rates_list = list(learning_rates)
# Accumulator for the performance
performance = []

for num, eta in enumerate(l_rates_list, start=1):
    next_perf = average_train_data(n_tests, epochs, eta)
    net.reset() ## need to check if the reset actually works..
    performance.append(next_perf)
    print '{0} out of {1} completed.'.format(num, len(l_rates_list))


for l, p in zip(learning_rates, performance):
    print 'eta = {0}, result = {1}'.format(l, p)

### ploting
print len(performance)
print len(l_rates_list)

plt.plot(l_rates_list, performance)
plt.plot(l_rates_list, performance, 'ro')
plt.xlabel('eta (learning rate)')
plt.ylabel('accuracy')
plt.show()
