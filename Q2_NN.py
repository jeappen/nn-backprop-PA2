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
    return 1. / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return np.multiply(sigmoid(x), 1 - sigmoid(x))


def softmax(t):
    return np.divide(np.exp(t), np.transpose(np.tile(np.sum(np.exp(t), axis=1), (np.shape(t)[1], 1))))  # np.exp(t)/


def softmax_derivative(t):
    return np.multiply(softmax(t), 1 - softmax(t))


# def predict_output2(x, weight_list):
#     num_samples = np.shape(x)[0]
#     num_of_layers = len(weight_list)
#     layer_op = list();
#     layer_op.append(np.hstack((np.ones((num_samples, 1)), x)))
#     for layer_ind in xrange(num_of_layers):
#         (prev_layer_size, layer_size) = np.shape(weight_list[layer_ind])
#         if layer_ind != num_of_layers-1:
#             layer_op.append(np.hstack(np.ones((num_samples, 1)), sigmoid(np.dot(layer_op[layer_ind], weight_list))))
#         else:
#             layer_op.append(softmax(np.dot(layer_op[layer_ind], weight_list)))
#         layer_op.append(np.zeros((num_samples, layer_size)))
#
#     print layer_op
#     # print np.shape(aj)
#     print aj
#     z = sigmoid(aj)
#     # print z
#     ak = np.dot(np.hstack((np.ones((num_samples, 1)), z)), output_layer_weights)
#     y = softmax(ak)
#     # print y
#     # predicted_class = y.argmax(axis=1)
#     return y, z


def predict_output(x, hidden_layer_weights, output_layer_weights):
    (num_samples, num_dim) = np.shape(x)
    aj = np.dot(np.hstack((np.ones((num_samples, 1)), x)), hidden_layer_weights)
    # print np.shape(aj)
    # print np.shape(aj)
    z = sigmoid(aj)
    # print z
    ak = np.dot(np.hstack((np.ones((num_samples, 1)), z)), output_layer_weights)
    # print np.shape(ak)
    y = softmax(ak)
    # print y
    # predicted_class = y.argmax(axis=1)
    return y, z


def backprop_step(x_train, y_train, num_hidden_nodes, num_classes, step_size_alpha, step_size_beta, max_steps,
                  tolerance, lambda_reg):
    [n, p] = np.shape(x_train)

    hidden_layer_weights = np.random.randn(p + 1, num_hidden_nodes)
    output_layer_weights = np.random.randn(num_hidden_nodes + 1, num_classes)

    iter_num = 0
    rss = tolerance + 1

    error_vector = np.zeros((max_steps, 1))

    y_vector = (np.tile(np.arange(num_classes) + 1, (n, 1)) == np.transpose(
        np.tile(y_train, (num_classes, 1)))) + 0  # Turns bool to int

    while iter_num < max_steps:  # & rss > tolerance )
        (y, z) = predict_output(x_train, hidden_layer_weights, output_layer_weights)
        error_mat = (y - y_vector)

        delta = [[], [], []]

        delta[2] = np.multiply(np.multiply((2 * error_mat), y), 1 - y)

        layer_op = np.multiply(1 - z, z)
        temp = np.dot(delta[2], np.transpose(output_layer_weights[1:, :]))

        delta[1] = np.multiply(layer_op, temp)

        layer_op = np.multiply(1 - x_train, x_train)
        temp2 = np.dot(delta[1], np.transpose(hidden_layer_weights[1:, :]))

        delta[0] = np.multiply(layer_op, temp2)

        output_layer_weights -= (step_size_beta * (
        np.dot(np.transpose(np.hstack((np.ones((n, 1)), z))), delta[2])) + lambda_reg * output_layer_weights)
        hidden_layer_weights -= (step_size_alpha * (
        np.dot(np.transpose(np.hstack((np.ones((n, 1)), x_train))), delta[1])) + lambda_reg * hidden_layer_weights)

        # print 'output'
        # print output_layer_weights
        # print 'hidden'
        # print hidden_layer_weights

        #
        # # print error_mat
        # # break
        # # print np.shape(temp)
        # for samp_int in xrange(n):
        #     # print output_layer_weights
        #
        #
        #     # (y, z) = predict_output(x_train, hidden_layer_weights, output_layer_weights)
        #     # error_mat = (y_vector - y)
        #
        #     temp = softmax_derivative(np.dot([z[samp_int,:]], output_layer_weights[1:, :]))
        #     temp3 = np.transpose(np.tile(np.hstack((1, z[samp_int,:])), (num_classes, 1)))
        #     temp2 = np.multiply(np.tile(temp, (num_hidden_nodes+1, 1)), temp3)
        #     # print np.shape(temp2)
        #     temp1 = np.tile(-error_mat[samp_int,:], (num_hidden_nodes+1, 1))
        #     print 'temp1',np.shape(temp1)
        #     # print np.shape(temp1)
        #     dRBeta = np.multiply(temp1, temp2 )
        #
        #     # print output_layer_weights
        #
        #     output_layer_weights -= step_size_beta*dRBeta
        #     # print output_layer_weights
        #     # break
        #
        #     # print dRBeta
        #     # print np.shape(temp2)
        #
        #     acti_dash = sigmoid_derivative(np.dot(x_train[samp_int], hidden_layer_weights[1:, :]))
        #     temp4 = np.multiply(np.tile(temp, (num_hidden_nodes, 1)), temp1[:-1])
        #     temp5 = np.multiply(temp4, output_layer_weights[1:])
        #     temp6 = np.sum(temp5, axis=1)
        #
        #     temp8 = np.multiply(acti_dash,temp6)
        #
        #     temp7 = np.transpose(np.tile(np.hstack((1, x_train[samp_int])), (num_hidden_nodes, 1)))
        #
        #     dRAlpha = np.multiply( np.tile(temp8,(p+1,1)),temp7 )
        #
        #     hidden_layer_weights -= step_size_alpha*dRAlpha
        #
        #     # print np.sum(error_mat)
        #     # break
        #     # print np.shape(acti_dash)
        #     # print np.shape(np.sum(temp5, axis=1))
        #
        #
        #
        #
        # # print np.shape(temp)
        # # print np.shape(z)
        # # print np.shape(error_mat)
        # # break
        error_vector[iter_num] = np.sum(np.square(error_mat))
        iter_num += 1
    # print error_vector
    return hidden_layer_weights, output_layer_weights, error_vector


def backprop():
    return 0


def regularise(x, min_val=None, ranges=None):
    if min_val is None:
        min_val = np.min(x, axis=0)
    if ranges is None:
        ranges = np.max(x - np.tile(min_val, (np.shape(x)[0], 1)), axis=0)
    # print x[1:3, :]
    # print 'ranges', ranges
    scaled_data = np.divide((x - np.tile(min_val, (np.shape(x)[0], 1))) - 0.0, np.tile(ranges, (np.shape(x)[0], 1)))
    # print scaled_data[1:3, :]
    # print np.shape(scaled_data )
    return scaled_data, min_val, ranges


sample_windows = range(1, 1006)

TrainData = np.genfromtxt('DS2_train.csv', delimiter=',').astype(int)
TestData = np.genfromtxt('DS2_test.csv', delimiter=',').astype(int)

Num_classes = 4
(X_train, min_train, range_train) = regularise(TrainData[sample_windows, :-1])
Y_train = TrainData[:, -1]
# print min_train,range_train
(X_test, dum1, dum2) = regularise(TestData[:, :-1], min_train, range_train)
# print np.shape(TestData[:, :-1])
# print np.shape(X_test)
Y_test = TestData[:, -1]

# raw_input()

[N, P] = np.shape(X_train)
N_train = np.shape(X_train)[0]
rand_ind = np.random.permutation(N_train)
val_fraction = 0.2

val_Indices = rand_ind[0:np.ceil(val_fraction * N_train)]
train_Indices = rand_ind[np.ceil(val_fraction * N_train):]

X_val = X_train[val_Indices, :]
Y_val = Y_train[val_Indices]
X_train = X_train[train_Indices, :]
Y_train = Y_train[train_Indices]

print np.shape(Y_train)
print np.shape(Y_val)

step_size_alpha = 1e-7
step_size_beta = 1e-4
max_num_iter = 100
Lambda = 1e-3
tol = 1e-4  # Not Implemented

alpha_step_range = np.logspace(-8, 1, 10)
beta_step_range = np.logspace(-8, 1, 10)
lambda_range = np.logspace(-3, 2, 6)

best_accuracy = 0;
best_beta_step = 0;
best_alpha_step = 0;
best_Lambda = 0

count = 0;

for Lambda in lambda_range:
    for step_size_alpha in alpha_step_range:
        for step_size_beta in beta_step_range:
            (hlayer, oplayer, error_vector) = backprop_step(X_val, Y_val, NumHiddenNodes, NumClasses, step_size_alpha,
                                                            step_size_beta, max_num_iter, tol, Lambda)
            (y, z) = predict_output(X_test, hlayer, oplayer)
            accuracy = np.sum(Y_test == (np.argmax(y, axis=1) + 1)) / (np.shape(Y_test)[0] + 0.0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_beta_step = step_size_beta
                best_alpha_step = step_size_alpha
                best_Lambda = Lambda;
                print 'found better accuracy', best_accuracy
            count += 1
            print count

print (best_accuracy, best_beta_step, best_alpha_step, best_Lambda)
(hlayer, oplayer, error_vector) = backprop_step(X_train, Y_train, NumHiddenNodes, NumClasses, best_alpha_step,
                                                best_beta_step, max_num_iter, tol, best_Lambda)
(y, z) = predict_output(X_test, hlayer, oplayer)
accuracy = np.sum(Y_test == (np.argmax(y, axis=1) + 1)) / (np.shape(Y_test)[0] + 0.0)
Y_predicted = np.argmax(y, axis=1) + 1;
print 'accuracy on test data', accuracy

Precision = np.zeros((NumClasses, 1))
Recall = np.zeros((NumClasses, 1))

for class_ind in xrange(NumClasses):
    TP = np.sum(np.logical_and(Y_predicted == class_ind + 1, Y_test == class_ind + 1))
    FP = np.sum(np.logical_and(Y_predicted == class_ind + 1, Y_test != class_ind + 1))
    TN = np.sum(np.logical_and(Y_predicted != class_ind + 1, Y_test != class_ind + 1))
    FN = np.sum(np.logical_and(Y_predicted != class_ind + 1, Y_test == class_ind + 1))
    Precision[class_ind] = TP / (TP + FP)
    Recall[class_ind] = TP / (TP + FN)

F1Score = np.divide(np.multipy(2 * Precision, Recall), Precision + Recall)

with open('precision.txt', 'w') as file1:
    file1.writelines('\t'.join(i) + '\n' for i in Precision)
with open('recall.txt', 'w') as file2:
    file2.writelines('\t'.join(i) + '\n' for i in Recall)
with open('f1score.txt', 'w') as file3:
    file3.writelines('\t'.join(i) + '\n' for i in F1Score)


# print (np.argmax(y, axis=1)+1)
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
