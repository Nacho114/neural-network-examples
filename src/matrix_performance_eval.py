import mnist_loader
import network
import network_m
import time
import math

# Compare speed performance between normal and matrix based implementation

current_milli_time = lambda: int(round(time.time() * 1000))
units = '[ms]'

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

### Hyper-parameters
mini_batch_size = 10
eta = 3.0
n_epoch = 1

# Number of nodes per layer
l1 = 784
l2 = 30
l3 = 10

net = network.Network([l1, l2, l3])
net_m = network_m.Network_m([l1, l2, l3])

t0 = current_milli_time()
net.SGD(training_data, n_epoch, mini_batch_size, eta, test_data=test_data)
t1 = current_milli_time()

SGD_t = t1 - t0

t0 = current_milli_time()
net_m.SGD(training_data, n_epoch, mini_batch_size, eta, test_data=test_data)
t1 = current_milli_time()

SGD_M_t = t1 - t0

diff = SGD_t - SGD_M_t
speed_up = math.ceil(diff * 1. / SGD_t * 100)

print 'SGD   run time = {0}'.format(SGD_t), units
print 'SGD_M run time = {0}'.format(SGD_M_t), units
print 'difference = {0}'.format(diff), units
print '{0} %  speed up'.format(speed_up)
