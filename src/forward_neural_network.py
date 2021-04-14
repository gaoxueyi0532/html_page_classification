# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:23:40 2018
@author: gaoxueyi@chianso.com
"""

import os
import sys
import math
import copy
import random
import json
import numpy as np

def equal(a, b):
    return abs(a-b) < 0.000001

def less(a, b):
    return (a - b) < -0.01

class ReluCell(object):
    @staticmethod
    def activate(z):
        ''' z is a 1-dim ndarray, such as [-1.0, 1.0, 0.0], for element which
        is smaller than 0, reset to 0, others keep unchanged '''
        z[z < 0] = 0
        return z

    @staticmethod
    def prime(z):
        ''' Derivative of relu function, 0 when z <= 0, otherwise 1 '''
        zt = np.where(z <= 0, 0, z)
        zt[zt > 0] = 1
        return zt

class SigmodCell(object):
    @staticmethod
    def activate(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmod(z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def prime(z):
        """ Derivative of the sigmoid function. """
        return SigmodCell.sigmod(z) * (1.0 - SigmodCell.sigmod(z))


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        """ Return the cost associated with an output ``a`` and desired
        output ``y``.  Note that np.nan_to_num is used to ensure
        numerical stability. In particular, if both ``a`` and
        ``y`` have a 1.0 in the same slot, then the expression
        (1-y)*np.log(1-a) returns NaN. The np.nan_to_num
        ensures that that is converted to the correct value (0.0). """
        #return np.sum(np.nan_to_sum(-y*np.log(a) - (1-y)*np.log(1-a)))
        return np.sum(-y*np.log(a+0.000001) - (1.0-y)*np.log(1.0-a+0.000001))

    @staticmethod
    def delta(z, a, y):
        """ Return the error delta from the output layer. Note that the
        parameter ``z`` is not used by the method.  It is included in the
        method's parameters in order to make the interface consistent with
        the delta method for other cost classes. """
        return (a-y)

    @staticmethod
    def derivation(a_s, y_s):
        """ Derivative of cross entroy cost, where a_s, y_s represent
        avtivation value of output layer and labels of training data
        respectivly, particularly, once a*(1.0-a) is zero, (a-y) /
        a*(1.0-a) should be 1.0 according law of lobida. """
        o = []
        for a,y in np.nditer([a_s, y_s]):
            if equal(float(a), 1.0) or equal(float(a), 0.0):
                o.append(1.0)
            else:
                o.append((a-y) / a*(1.0-a))
        return np.array(o, dtype='float')


class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        ''' compute square of 2-norm of a-y '''
        return 0.5 * np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """ Return the error delta from the output layer. """
        return (a-y) * sigmod_prime(z)

    @staticmethod
    def derivation(a_s, y_s):
        """ Derivative of square error cost, where a_s, y_s represent
        avtivation value of output layer and labels of training data respectivly """
        return (a_s - y_s)


class Forward_neural_netWork(object):
    def __init__(self, sizes, hide_cell=ReluCell, output_cell=SigmodCell, cost=CrossEntropyCost):
        # layer num of Network
        self.num_layers = len(sizes)

        # cell type of hide and output layer
        self.hide_cell = hide_cell
        self.output_cell = output_cell

        # unit num in every layer
        self.sizes = sizes

        # learning rate and decay setting
        # formulation: eta = eta * eta_decay ^ (current_epoch / eta_decay_step)
        # such as eta = 0.1 * 0.9 ^ (40 / 10) 
        self.eta = 0.1
        self.eta_decay = 0.9
        self.eta_decay_step = 10

        # L2 regularization
        self.l2_open = True
        self.l2_lamda = 0.1

        # mini batch size
        self.batch_size = 100

        # cost function
        self.cost = cost

        # init the weights and bias layer by layer
        self.init()

    def init(self):
        self.biases = []
        self.weights = []
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        for x,y in zip(self.sizes[:-1], self.sizes[1:]):
            ''' avoid variance too big to decay learning rate prematurely '''
            self.weights.append(np.random.randn(y,x) / np.sqrt(x))

    def feedforward(self, a):
        """ Return the out of the network if "a" is input. """
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = (self.hide_cell).activate(np.dot(w,a) + b)
        else:
            z = np.dot(self.weights[-1], a) + self.biases[-1]
            a = (self.output_cell).activate(z)
            return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None,
            monitor_test_cost=False,
            monitor_test_accuracy=False,
            monitor_train_cost=False,
            monitor_train_accuracy=False):
        """ Train the neural network using mini-batch stochastic
        gradient descent. The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs. The other non-optional parameters are
        self-explanatory. If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out. This is useful for
        tracking progress, but slows things down substantially. """
        self.batch_size = mini_batch_size
        self.eta = eta

        n_test = 0
        if test_data: n_test = len(test_data)

        # cost and accuracy in test data
        test_cost, test_accuracy = [], []
        # cost and accuracy in train data
        train_cost, train_accuracy = [], []

        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)

            mini_batches = []
            # generate mini batches
            for k in range(0, n, self.batch_size):
                batch = training_data[k : k+self.batch_size]
                mini_batches.append(batch)

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, n)

            # adjust learning rate
            self.eta = self.eta * self.eta_decay ** (j / self.eta_decay_step)

            # visualize
            if monitor_test_cost and test_data:
                cost = self.total_cost(test_data)
                print("Epoch {0} test cost: {1}".format(j, cost))
            if monitor_test_accuracy and test_data:
                acc = self.evaluate(test_data)
                print("Epoch {0} test accuracy: {1} / {2}".format(j, acc, n_test))
            if monitor_train_cost:
                cost = self.total_cost(training_data, test=False)
                print("Epoch {0} train cost: {1}".format(j, cost))
            if monitor_train_accuracy:
                acc = self.evaluate(training_data, test=False)
                print("Epoch {0} train accuracy: {1} / {2}".format(j, acc, n))

    def update_mini_batch(self, mini_batch, n):
        """ Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of sample, and "n" is
        size of training data. """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        ''' single-sample update mode '''
        #for x, y in mini_batch:
        #  delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        #  nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        #  nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        ''' matrix-based update mode '''
        nabla_b, nabla_w = self.backprop_v2(mini_batch)

        if self.l2_open:
            ws = []
            bs = []
            for w, nw in zip(self.weights, nabla_w):
                e = w - (self.eta/self.batch_size)*nw - (self.l2_lamda*self.eta/n)*w
                ws.append(e)
            for b, nb in zip(self.biases, nabla_b):
                e = b - (self.eta/self.batch_size)*nb
                bs.append(e)
            self.weights = ws
            self.biases = bs
        else:
            self.weights = [w-(self.eta/self.batch_size)*nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b-(self.eta/self.batch_size)*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop_v2(self, mini_batch):
        """ matrix-based impletation of backprop().
        Return a tuple "(nabla_b, nabla_w)" representing the
        gradient for the cost function C. "nabla_b" and
        "nabla_w" are layer-by-layer lists of numpy arrays, similar
        to "self.biases" and "self.weights". """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # x = [x1,x2,...,xn].transpose()
        a = np.array(mini_batch, dtype='float').transpose()
        x = a[:-1]
        y = a[-1].reshape(1, len(mini_batch))

        """ feedforward """
        # list to store all the activations, layer by layer
        at = x
        ats = [x]

        # list to store all the z vectors, layer by layer
        zs = []
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            # hide layer
            # Auto-broadcast in numpy
            z = np.dot(w,at) + b
            zs.append(z)
            at = (self.hide_cell).activate(z)
            ats.append(at)
        else:
            # output layer
            z = np.dot(self.weights[-1], ats[-1]) + self.biases[-1]
            zs.append(z)
            at = (self.output_cell).activate(z)
            ats.append(at)

        """ backward """
        dt = (self.cost).derivation(ats[-1], y) * (self.output_cell).prime(zs[-1])
        nabla_b[-1] = np.sum(dt, axis=1, keepdims=True)
        nabla_w[-1] = np.dot(dt, ats[-2].transpose())
        for l in range(2, self.num_layers):
            dt = np.dot(self.weights[-l+1].transpose(), dt) * (self.hide_cell).prime(zs[-l])
            nabla_b[-l] = np.sum(dt, axis=1, keepdims=True)
            nabla_w[-l] = np.dot(dt, ats[-l-1].transpose())
        return (nabla_b, nabla_w)

    def backprop(self, x, y):
        """ Return a tuple "(nabla_b, nabla_w)" representing the
        gradient for the cost function C_x. "nabla_b" and
        "nabla_w" are layer-by-layer lists of numpy arrays, similar
        to "self.biases" and "self.weights". """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        """ feedforward """
        activation = x
        # list to store all the activations, layer by layer
        activations = [x]
        # list to store all the z vectors, layer by layer
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmod(z)
            activations.append(activation)

        """ backward """
        delta = self.cost_derivation(activations[-1], y) * sigmod_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmod_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activation[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, data, test=True):
        """ Return the number of test inputs for which the neural
        network outputs the correct result. """
        res = []
        count = 0
        if test:
            for x,y in data:
                a = self.feedforward(np.array(x,dtype='float').reshape(len(x),1))
                res.append((a,y))
        else:
            for x in data:
                a = self.feedforward(np.array(x[:-1],dtype='float').reshape(len(x[:-1]),1))
                res.append((a, x[-1]))
        for e in res:
            distance = abs(float(e[0]) - e[1])
            if less(distance, 0.5):
                count += 1
        return count

    def test(self, x):
        """ return true if vector x represents a hub in a big probility. """
        a = self.feedforward(np.array(x,dtype='float').reshape(len(x),1))
        return (float(a) > 0.5, a)

    def total_cost(self, data, test=True):
        """ Return the total cost for the data set ``data``. The flag
        ``test`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data. See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above. """
        cost = 0.0
        res = []
        if test:
            for x,y in data:
                a = self.feedforward(np.array(x,dtype='float').reshape(len(x),1))
                res.append((a, y))
        else:
            for x in data:
                a = self.feedforward(np.array(x[:-1],dtype='float').reshape(len(x[:-1]),1))
                res.append((a, x[-1]))
        for e in res:
            cost += self.cost.fn(e[0], e[1])
        return cost

    def save_model(self, fn):
        """ save parameters wrapped by json format, to file """
        model = {}
        model["sizes"] = self.sizes
        model["biases"] = [b.tolist() for b in self.biases]
        model["weights"] = [w.tolist() for w in self.weights]
        with open(fn, "w") as f:
            json.dump(model, f)

def loads_model(fn):
    """ load parameters from file """
    with open(fn, "r") as f:
        data = json.load(f)
        fnn = Forward_neural_netWork(data["sizes"])
        fnn.biases = [np.array(b) for b in data["biases"]]
        fnn.weights = [np.array(w) for w in data["weights"]]
        return fnn

def load_data(test=True, sampling_density=0.1):
    """ load train data and label, test data and label respectively,
    all are ndarray format, where sampling ratio defaut is 0.1 """
    train_data = []
    with open("../data/pos_v2.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if not line.strip():
                continue
            lst = line.strip().split(',')
            if "" in lst:
                continue
            sample = map(int, lst)
            train_data.append(list(map(float, sample)))
    with open("../data/neg.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if not line.strip():
                continue
            lst = line.strip().split(',')
            if "" in lst:
                continue
            sample = map(int, lst)
            train_data.append(list(map(float, sample)))
    # shuffle data
    random.shuffle(train_data)
    if not test:
        return (train_data, None)
    # sampling test data
    test_data = []
    s = set()
    while float(len(test_data)) < sampling_density * float(len(train_data)):
        i = np.random.randint(0, len(train_data))
        if i not in s:
            test_data.append((train_data[i][:-1], train_data[i][-1]))
            s.add(i)
    # remove test data from train data
    for e in test_data:
        item = copy.deepcopy(e[0])
        item.append(e[1])
        train_data.remove(item)
    return (train_data, test_data)

def main():
    #train_data, test_data = load_data()
    #fnn = Forward_neural_netWork([512, 100, 1])
    #fnn.SGD(train_data, 10, 500, 0.1, test_data, True, True, True, True)
    #fnn.save_model("../model/hub_rg_model_v2.txt")
    model = loads_model("../model/hub_rg_model_v2.txt")
    lst = [-1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    res = model.test(lst)
    print(res)

if __name__ == '__main__':
    main()
