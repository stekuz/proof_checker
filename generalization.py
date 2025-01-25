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
from sentence_pairs import sentence_pairs
from sentence_pairs import validation_sentences

sys.setrecursionlimit(100000)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

all_tokens = {}

data_train = []
mxlen = 0

data_raw = sentence_pairs

special_chars = ['„', '”', '"', '\'', '.', ',', '/', '\\', ' ', '   ', '*', ';', ':', '`', '[', ']', '{', '}', '!', '?', '\n', '$']

for i in range(len(data_raw)):
    en = []
    de = []
    now = ''
    for j in data_raw[i][0]:
        if j in special_chars:
            if now != '':
                en.append(now + j)
            #en.append(j)
            now = ''
        else:
            now += j.lower()
    if now != '':
        en.append(now)
        now = ''
    for j in data_raw[i][1]:
        if j in special_chars:
            if now != '':
                de.append(now + j)
            #de.append(j)
            now = ''
        else:
            now += j.lower()
    if now != '':
        de.append(now)
        now = ''
    #data_train.append([de, en])
    mxlen = max([mxlen, len(de) + 2, len(en) + 2])

def data_preparation_common_seq2seq():
    global mxlen
    data_raw = sentence_pairs

    for i in range(len(data_raw)):
        en = []
        de = []
        now = ''
        for j in data_raw[i][0]:
            if j in special_chars:
                if now != '':
                    en.append(now + j)
                #en.append(j)
                now = ''
            else:
                now += j.lower()
        if now != '':
            en.append(now)
            now = ''
        for j in data_raw[i][1]:
            if j in special_chars:
                if now != '':
                    de.append(now + j)
                #de.append(j)
                now = ''
            else:
                now += j.lower()
        if now != '':
            de.append(now)
            now = ''
        for j in de:
            all_tokens[j] = 1
        for j in en:
            all_tokens[j] = 1
        data_train.append([de, en])
        mxlen = max([mxlen, len(de) + 2, len(en) + 2])
    with open('./all_tokens.txt', 'w') as f:
        for i in all_tokens:
            f.write(i + '\n')

    all_tokens_list = open('./all_tokens.txt', 'r').readlines()
    all_tokens_list = [token[:-1] for token in all_tokens_list]
    all_tokens_list = sorted(all_tokens_list)
    for i in range(len(all_tokens_list)):
        all_tokens[all_tokens_list[i]] = i
    
    for i in range(len(data_train)):
        for j in range(len(data_train[i][1])):
            data_train[i][1][j] = [1, all_tokens[data_train[i][1][j]] / (len(all_tokens) + 1)]
        for j in range(len(data_train[i][0])):
            data_train[i][0][j] = all_tokens[data_train[i][0][j]]

    
    X = []
    Y = []
    for i in range(len(data_train)):
        X.append(data_train[i][1])
        Y.append(data_train[i][0])
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            Y[i][j] /= len(all_tokens) + 1
    for i in range(len(X)):
        nowlen = len(X[i])
        for j in range(mxlen - nowlen):
            X[i].append([1, len(all_tokens) / (len(all_tokens) + 1)])
        for j in range(mxlen):
            X[i][j][0] = j / mxlen
        nowlen = len(Y[i])
        for j in range(mxlen - nowlen):
            Y[i].append(len(all_tokens) / (len(all_tokens) + 1))
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')

    np.save('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy', X)
    np.save('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_y.npy', Y)

#data_preparation_common_seq2seq()

mxlen = int(open('./mxlen.txt', 'r').readlines()[0])
all_tokens_list = open('./all_tokens.txt', 'r').readlines()
all_tokens_list = [token[:-1] for token in all_tokens_list]
all_tokens_list = sorted(all_tokens_list)
print(all_tokens_list)

for i in range(len(all_tokens_list)):
    all_tokens[all_tokens_list[i]] = i

validation_x = []
validation_y = []
data_validation = []
data_raw = validation_sentences
for i in range(len(data_raw)):
    en = []
    de = []
    now = ''
    for j in data_raw[i][0]:
        if j in special_chars:
            if now != '':
                en.append(now + j)
            #en.append(j)
            now = ''
        else:
            now += j.lower()
    if now != '':
        en.append(now)
        now = ''
    for j in data_raw[i][1]:
        if j in special_chars:
            if now != '':
                de.append(now + j)
            #de.append(j)
            now = ''
        else:
            now += j.lower()
    if now != '':
        de.append(now)
        now = ''
    data_validation.append([de, en])
    #mxlen = max([mxlen, len(de) + 5, len(en) + 5])

for i in range(len(data_validation)):
    for j in range(len(data_validation[i][1])):
        data_validation[i][1][j] = [1, all_tokens[data_validation[i][1][j]] / (len(all_tokens) + 1)]
    for j in range(len(data_validation[i][0])):
        data_validation[i][0][j] = all_tokens[data_validation[i][0][j]]

validation_x = []
validation_y = []
for i in range(len(data_validation)):
    validation_x.append(data_validation[i][1])
    validation_y.append(data_validation[i][0])

for i in range(len(validation_x)):
    nowlen = len(validation_x[i])
    for j in range(mxlen - nowlen):
        validation_x[i].append([1, len(all_tokens) / (len(all_tokens) + 1)])
    for j in range(mxlen):
        validation_x[i][j][0] = j / mxlen
    nowlen = len(validation_y[i])
    for j in range(mxlen - nowlen):
        validation_y[i].append(len(all_tokens))
    for j in range(mxlen):
        validation_y[i][j] /= len(all_tokens) + 1
validation_x = np.array(validation_x, dtype='float32')
validation_y = np.array(validation_y, dtype='float32')

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

class SGD:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.t = 0

    def initialize(self, params):
        return

    def update(self, params, grads):
        result = copy.deepcopy(params)
        for key in params:
            result[key] += self.learning_rate * grads[key]
        return result

def periodic_activation(x):
    return tf.cast(x > 0, dtype='float32') * x#(tf.math.sin(2 * np.pi * x) + 1) / 2

def periodic_activation_derivative(x):
    return tf.cast(x > 0, dtype='float32') #* np.pi * tf.math.cos(2 * np.pi * x)

def xavier_initialization(shape):
    return tf.random.uniform(shape, minval=-np.sqrt(6 / (shape[0] + shape[1])), maxval=np.sqrt(6 / (shape[0] + shape[1])), dtype='float32')

class ToroidalFFNN:
    def __init__(self, input_size, output_size, index, learning_rate=0.0001):
        self.W1 = xavier_initialization((input_size, output_size))
        self.b1 = tf.zeros((1, output_size), dtype='float32')
        self.index = index
        self.learning_rate = learning_rate
        self.optimizer = AdamOptimizer(learning_rate=learning_rate)
        self.optimizer.initialize({
            'W1': self.W1,
            'b1': self.b1
        })

    def forward(self, X):
        self.A1 = tf.linalg.matmul(X, self.W1) + self.b1
        self.Z1 = periodic_activation(self.A1)
        return self.Z1
    
    def compute_loss(self, Y_pred, Y_true):
        loss = tf.reduce_sum(tf.math.sqrt(tf.math.minimum(abs(Y_true - Y_pred), 1 - abs(Y_true - Y_pred)))) / (Y_pred.shape[0] * Y_pred.shape[1])
        return loss

    def compute_accuracy(self, Y_pred, Y_true):
        y_pred_classes = tf.cast(tf.round(Y_pred * (len(all_tokens) + 1)), dtype='int32')
        y_true_classes = tf.cast(tf.round(Y_true * (len(all_tokens) + 1)), dtype='int32')
        threshold = 1
        correct_predictions = abs(y_true_classes - y_pred_classes) < threshold
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def backward(self, X, A1, dA1):
        m = dA1.shape[1]
        dZ1 = dA1 * periodic_activation_derivative(A1)
        dW1 = tf.linalg.matmul(tf.transpose(X), dZ1) / m
        db1 = tf.math.reduce_sum(dZ1, axis=0, keepdims=True) / m 
        dA0 = tf.linalg.matmul(dZ1, tf.transpose(self.W1))

        def optimize():
            return self.optimizer.update({
                'W1': self.W1,
                'b1': self.b1
            }, {
                'W1': dW1,
                'b1': db1
            })
        
        new_params = optimize()
        self.W1 = new_params['W1']
        self.b1 = new_params['b1']

        return dA0
    
    def save_model(self, filepath):
        np.save(filepath + 'dense' + str(self.index) + 'W1.npy', self.W1)
        np.save(filepath + 'dense' + str(self.index) + 'b1.npy', self.b1)
        np.save(filepath + 'dense' + str(self.index) + 'adam.npy', np.array([self.optimizer.t], dtype='int32'))
    
    def load_model(self, filepath):
        self.W1 = np.load(filepath + 'dense' + str(self.index) + 'W1.npy')
        self.b1 = np.load(filepath + 'dense' + str(self.index) + 'b1.npy')
        self.optimizer.initialize({
            'W1': self.W1,
            'b1': self.b1
        })
        self.optimizer.t = np.load(filepath + 'dense' + str(self.index) + 'adam.npy')[0]

def train():
    X = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy')[:2]
    Y = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_y.npy')[:2]
    X = X.tolist()
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = X[i][j][1]
    X = np.array(X, dtype='float32')
    X = []
    Y = []
    for i in range(100):
        a, b = random.uniform(0, 1), random.uniform(0, 1)
        X.append([a, b])
        Y.append([a * b])
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')
    '''
    input_layer = keras.Input((X.shape[1], ))
    middle = layers.Dense(100, activation='relu')(input_layer)
    output_layer = layers.Dense(1, activation='linear')(middle)
    model = keras.Model(input_layer, output_layer)
    model.compile(optimizer='adam', loss='mae')
    model.summary()
    model.fit(X, Y, batch_size=1, epochs=100)'''
    hidden_size = 1000
    #learning_rate = 
    l1 = ToroidalFFNN(X.shape[1], hidden_size, 0)
    l2 = ToroidalFFNN(hidden_size, Y.shape[1], 1)
    def toroidal_difference(Y_pred, Y_true):
        dif = Y_pred - Y_true
        mask = tf.cast(abs(dif) > 1 - abs(dif), dtype='float32')
        return dif - mask * tf.math.sign(dif)
    for i in range(10000):
        A0 = l1.forward(X)
        A1 = l2.forward(A0)
        print(f'Epoch: {i + 1}, loss: {l2.compute_loss(A1, Y)}, accuracy: {l2.compute_accuracy(A1, Y)}')
        dif = toroidal_difference(A1, Y)
        dA0 = l2.backward(A0, A1, A1 - Y)
        dA0 = l1.backward(X, A0, dA0)
train()
