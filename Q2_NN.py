import csv
import pandas as pd

# df = pd.read_csv("DS2_train.csv")
import numpy as np
from scipy import stats
# from numpy import genfromtxt

# Initialization
NumHiddenNodes = 10
NumClasses = 4  # Num of classes

def sigmoid(x):
    return 1./(1+np.exp(-x))


def sigmoid_derivative(x):
    return np.multiply(sigmoid(x), 1-sigmoid(x))


def softmax(t):
    return np.exp(t)/np.sum(np.exp(t))


def softmax_derivative(t):
    return np.multiply(softmax(t), 1-softmax(t))


def predict_output(x ,hidden_layer_weights, output_layer_weights):
    (num_samples, num_dim) = np.shape(x)
    aj = np.dot(np.hstack((np.ones((num_samples, 1)), x)), hidden_layer_weights)
    # print np.shape(aj)
    print aj
    z = sigmoid(aj)
    # print z
    ak = np.dot(np.hstack((np.ones((num_samples, 1)), z)), output_layer_weights)
    y = softmax(ak)
    # print y
    # predicted_class = y.argmax(axis=1)
    return y, z


def backprop_step(x_train, y_train, num_hidden_nodes, num_classes, step_size_alpha, step_size_beta, max_steps, tolerance, regularisation):
    [n, p] = np.shape(x_train)

    hidden_layer_weights = np.random.rand(p + 1, num_hidden_nodes)
    output_layer_weights = np.random.rand(num_hidden_nodes + 1, num_classes)

    iter_num = 0
    rss = tolerance+1

    y_vector = (np.tile(np.arange(num_classes)+1, (n, 1)) == np.transpose(np.tile(y_train, (num_classes, 1)))) + 0 # Turns bool to int

    while iter_num < max_steps :  # & rss > tolerance )
        (y, z) = predict_output(x_train, hidden_layer_weights, output_layer_weights)
        error_mat = (y_vector-y)

        # print np.shape(temp)
        for samp_int in xrange(n):
            print output_layer_weights
            if samp_int== 10:
                break

            (y, z) = predict_output(x_train, hidden_layer_weights, output_layer_weights)
            error_mat = (y_vector - y)

            print error_mat[1:10]

            temp = softmax_derivative(np.dot(z[samp_int], output_layer_weights[1:, :]))
            temp3 = np.transpose(np.tile(np.hstack((1, z[samp_int])), (num_classes, 1)))
            temp2 = np.multiply(np.tile(temp, (num_hidden_nodes+1, 1)), temp3)
            # print np.shape(temp2)
            temp1 = np.tile(-2*error_mat[samp_int], (num_hidden_nodes+1, 1))
            # print np.shape(temp1)
            dRBeta = np.multiply(temp1 , temp2 )

            # print output_layer_weights

            output_layer_weights -= step_size_beta*dRBeta
            # print output_layer_weights
            # break

            # print dRBeta
            # print np.shape(temp2)

            acti_dash = sigmoid_derivative(np.dot(x_train[samp_int], hidden_layer_weights[1:, :]))
            temp4 = np.multiply(np.tile(temp, (num_hidden_nodes, 1)), temp1[:-1])
            temp5 = np.multiply(temp4, output_layer_weights[1:])
            temp6 = np.sum(temp5, axis=1)

            temp8 = np.multiply(acti_dash,temp6)

            temp7 = np.transpose(np.tile(np.hstack((1, x_train[samp_int])), (num_hidden_nodes, 1)))

            dRAlpha = np.multiply( np.tile(temp8,(p+1,1)),temp7 )

            hidden_layer_weights -= step_size_alpha*dRAlpha

            print np.sum(error_mat)
            # print np.shape(acti_dash)
            # print np.shape(np.sum(temp5, axis=1))

            if samp_int ==10:
                break



        #
        # print np.shape(temp)
        # print np.shape(z)
        # print np.shape(error_mat)
        break

        iter_num += 1


def backprop():
    return 0

TrainData = np.genfromtxt('DS2_train.csv', delimiter=',').astype(int)
TestData = np.genfromtxt('DS2_test.csv', delimiter=',').astype(int)

X_train = stats.zscore(TrainData[:, :-1], axis= 1)
Y_train = TrainData[:, -1]

[N, P] = np.shape(X_train)
N_train = np.shape(X_train)[0]

step_size_alpha = 1
step_size_beta = 100e-1
backprop_step(X_train, Y_train, NumHiddenNodes, NumClasses, step_size_alpha, step_size_beta, 10, 1e-4, 0)

#
# FirstLayerWeights = np.random.rand(P+1, NumHiddenNodes)
# OutputLayerWeights = np.random.rand(NumHiddenNodes+1, K)

#
# start = 700
# windowsize = 100
# TestWindow = xrange(start, start+windowsize)
# print np.shape(y)
#
# print y[TestWindow]
# print Y_train[TestWindow]

# print FirstLayerWeights
# print OutputLayerWeights

#
# with open('DS2_train.csv', 'rb') as csvfile:
#     TrainFile=csv.reader(csvfile)
#     # for row in TrainFile:
#     #     TrainData.append(int(row))
#
# with open('DS2_test.csv', 'rb') as csvfile:
#     TestData=csv.reader(csvfile)

