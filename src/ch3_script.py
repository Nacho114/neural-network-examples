import mnist_loader
#import network2_m as network2
import network2_m as network2
import matplotlib.pyplot as plt
import numpy as np


def map_accuracy(total_samples, sample_accuracy):
    return map(lambda x:
                x * 1. / total_samples * 100, sample_accuracy)

training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()

net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost, reg=network2.L2)
epochs = 60
total_training_samples = 50000
total_evaluation_samples = 10000
file_name = 'net2_test'

evaluation_cost, evaluation_accuracy, \
training_cost, training_accuracy = \
        net.SGD(training_data, epochs, 10, 0.1,
                lmbda = 5.0,
                evaluation_data=validation_data,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=False,
                monitor_training_accuracy=True,
                monitor_training_cost=False)

#net.save(file_name)

tr_accuracy = map_accuracy(total_training_samples, training_accuracy)
test_accuracy = map_accuracy(total_evaluation_samples, evaluation_accuracy)
epochs_list = list(xrange(0, len(training_accuracy), 1))

plt.plot(epochs_list, tr_accuracy, color='r', label='acurracy on training data')
plt.plot(epochs_list, test_accuracy, color='b', label='acurracy on test data')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.ylim([90,100])
plt.legend(loc='lower right')
plt.show()
