import tensorflow as tf
import keras
from keras import layers
import os
import time
import matplotlib.pyplot as plt
import sys
import numpy as np
import json
import random
import copy
import math
import csv
import re

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.98, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def initialize(self, params):
        self.m = {key: tf.zeros_like(value) for key, value in params.items()}
        self.v = {key: tf.zeros_like(value) for key, value in params.items()}

    def update(self, params, grads):
        self.t += 1

        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            params[key] -= self.learning_rate * m_hat / (tf.sqrt(v_hat) + self.epsilon)
        
        return params

class FeedForwardLayer:
    def __init__(self, input_size, hidden_size, output_size, index, learning_rate=0.001, dropout=0):
        def xavier_initialization(shape):
            return tf.random.uniform(shape, minval=-np.sqrt(6 / (shape[0] + shape[1])), maxval=np.sqrt(6 / (shape[0] + shape[1])), dtype='float32')
        self.W1 = xavier_initialization((input_size, hidden_size))
        self.b1 = tf.zeros((1, hidden_size), dtype='float32')
        self.W2 = xavier_initialization((hidden_size, output_size))
        self.b2 = tf.zeros((1, output_size), dtype='float32')
        self.index = index
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.optimizer = AdamOptimizer(learning_rate=learning_rate, epsilon=1e-7)
        #self.optimizer = SGD(learning_rate=learning_rate)
        #self.optimizer = AdaBelief(learning_rate=learning_rate, epsilon=1e-16)
        self.optimizer.initialize({
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
        })

    def forward(self, X, output=0):
        self.Z1 = tf.linalg.matmul(X, self.W1) + self.b1
        self.A1 = tf.nn.relu(self.Z1)
        self.Z2 = tf.linalg.matmul(self.A1, self.W2) + self.b2
        epsilon = 0
        self.noise_matrix = tf.random.uniform(self.Z2.shape, 1 - epsilon, 1 + epsilon)
        #self.learning_rate += random.uniform(-self.learning_rate / 9, self.learning_rate / 10)
        #self.learning_rate = max(self.learning_rate, 0.00001)
        dropout_mask = tf.cast(tf.random.uniform(self.Z2.shape) >= self.dropout, dtype='float32')
        if output:
            return dropout_mask * tf.nn.sigmoid(self.Z2) * self.noise_matrix
        else:
            return dropout_mask * tf.nn.relu(self.Z2) * self.noise_matrix

    def compute_loss(self, Y_pred, Y_true):
        return tf.reduce_mean(abs(Y_pred - Y_true))

    def compute_accuracy(self, Y_pred, Y_true):
        #y_pred_classes = tf.argmax(Y_pred, axis=-1)
        #y_true_classes = tf.argmax(Y_true, axis=-1)
        y_pred_classes = tf.cast(tf.round(Y_pred), dtype='int32')
        y_true_classes = tf.cast(tf.round(Y_true), dtype='int32')
        #print(y_pred_classes, y_true_classes)
        threshold = 2
        correct_predictions = abs(y_true_classes - y_pred_classes) < threshold
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def backward(self, X, A2, dA2, output=0):
        m = dA2.shape[0] 
        self.noise_matrix = 1.0 / self.noise_matrix
        if output:
            dZ2 = dA2 * (tf.nn.sigmoid(A2) * (1 - tf.nn.sigmoid(A2))) * self.noise_matrix
        else:
            dZ2 = dA2 * tf.cast(A2 > 0, dtype='float32') * self.noise_matrix
        dW2 = tf.linalg.matmul(tf.transpose(self.A1), dZ2) / m
        db2 = tf.math.reduce_sum(dZ2, axis=0, keepdims=True) / m 

        dA1 = tf.linalg.matmul(dZ2, tf.transpose(self.W2))
        dZ1 = dA1 * tf.cast(self.A1 > 0, dtype='float32')
        dW1 = tf.linalg.matmul(tf.transpose(X), dZ1) / m 
        db1 = tf.math.reduce_sum(dZ1, axis=0, keepdims=True) / m 

        mnval = -10
        mxval = 10
        dW1 = tf.clip_by_value(dW1, mnval, mxval)
        db1 = tf.clip_by_value(db1, mnval, mxval)
        dW2 = tf.clip_by_value(dW2, mnval, mxval)
        db2 = tf.clip_by_value(db2, mnval, mxval)
        #print(dW1)

        def optimize():
            return self.optimizer.update({
                'W1': self.W1,
                'b1': self.b1,
                'W2': self.W2,
                'b2': self.b2,
            }, {
                'W1': dW1,
                'b1': db1,
                'W2': dW2,
                'b2': db2,
            })
        
        new_params = optimize()
        self.W1 = new_params['W1']
        self.b1 = new_params['b1']
        self.W2 = new_params['W2']
        self.b2 = new_params['b2']

        return dA1
    
    def save_model(self, filepath):
        np.save(filepath + 'dense' + str(self.index) + 'W1.npy', self.W1)
        np.save(filepath + 'dense' + str(self.index) + 'b1.npy', self.b1)
        np.save(filepath + 'dense' + str(self.index) + 'W2.npy', self.W2)
        np.save(filepath + 'dense' + str(self.index) + 'b2.npy', self.b2)
        np.save(filepath + 'dense' + str(self.index) + 'adam.npy', np.array([self.optimizer.t], dtype='int32'))
    
    def load_model(self, filepath):
        self.W1 = np.load(filepath + 'dense' + str(self.index) + 'W1.npy')
        self.b1 = np.load(filepath + 'dense' + str(self.index) + 'b1.npy')
        self.W2 = np.load(filepath + 'dense' + str(self.index) + 'W2.npy')  
        self.b2 = np.load(filepath + 'dense' + str(self.index) + 'b2.npy')
        self.optimizer.initialize({
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
        })
        self.optimizer.t = np.load(filepath + 'dense' + str(self.index) + 'adam.npy')[0]

def test():
    for i in range(100):
        shape = (1000, 1000)
        A = tf.random.uniform(shape, -0.3, 0.3)
        A_s = A * tf.cast(tf.random.uniform(shape) > 0.5, dtype='float32')
        S = tf.linalg.matmul(A_s, tf.linalg.matmul(tf.transpose(A), tf.linalg.inv(tf.linalg.matmul(A, tf.transpose(A)))))
        print(tf.linalg.det(S), tf.linalg.det(A), tf.linalg.det(A_s))
        U = tf.linalg.svd(S)[0]
        print(tf.math.reduce_max(U), tf.reduce_sum(tf.cast(U / tf.math.reduce_max(U) > 0.001, dtype='float32')))
        #print('A_s', A_s)
        #print('SA', tf.linalg.matmul(S, A))

def train():
    input_size = 1000
    ff = FeedForwardLayer(input_size, input_size, 1, index=1)
    X = []
    Y = []
    for i in range(100):
        a = []
        for j in range(input_size):
            a.append(random.uniform(-100, 100))
        res = 0
        for j in range(input_size):
            res ^= (a[j] < 0)
        X.append(a)
        Y.append([res])
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')
    for i in range(100):
        A2 = ff.forward(X, output=1)
        print(ff.compute_loss(A2, Y))
        dA2 = ff.backward(X, A2, A2 - Y, output=1)
    A = tf.transpose(ff.W1)
    A_s_9 = A * tf.cast(tf.random.uniform(A.shape) > 0.9, dtype='float32')
    A_s_99 = A * tf.cast(tf.random.uniform(A.shape) > 0.99, dtype='float32')
    S_9 = tf.linalg.matmul(A_s_9, tf.linalg.matmul(tf.transpose(A), tf.linalg.inv(tf.linalg.matmul(A, tf.transpose(A)))))
    S_99 = tf.linalg.matmul(A_s_99, tf.linalg.matmul(tf.transpose(A), tf.linalg.inv(tf.linalg.matmul(A, tf.transpose(A)))))
    print(tf.linalg.det(S_9), tf.linalg.det(S_99))# tf.linalg.det(A), tf.linalg.det(A_s))
    U_9 = tf.linalg.svd(S_9)[0]
    U_99 = tf.linalg.svd(S_99)[0]
    print('for 0.9:')
    #print(U_9)
    print(tf.math.reduce_max(U_9), tf.reduce_sum(tf.cast(U_9 / tf.math.reduce_max(U_9) > 0.001, dtype='float32')))
    print('for 0.99:')
    #print(U_99)
    print(tf.math.reduce_max(U_99), tf.reduce_sum(tf.cast(U_99 / tf.math.reduce_max(U_99) > 0.001, dtype='float32')))
train()
