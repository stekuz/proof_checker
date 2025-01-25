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

sys.setrecursionlimit(100000)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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
            return dropout_mask * tf.nn.softmax(self.Z2) * self.noise_matrix
        else:
            return dropout_mask * tf.nn.relu(self.Z2) * self.noise_matrix

    def compute_loss(self, Y_pred, Y_true):
        #return tf.reduce_mean(abs(Y_pred - Y_true))
        Y_pred = tf.clip_by_value(Y_pred, 1e-12, 1.0)
        return -tf.reduce_mean(tf.reduce_sum(Y_true * tf.math.log(Y_pred), axis=1))
        #return tf.keras.losses.categorical_crossentropy(Y_true, Y_pred)

    def compute_accuracy(self, Y_pred, Y_true):
        y_pred_classes = tf.argmax(Y_pred, axis=-1)
        y_true_classes = tf.argmax(Y_true, axis=-1)
        #y_pred_classes = tf.cast(tf.round(Y_pred), dtype='int32')
        #y_true_classes = tf.cast(tf.round(Y_true), dtype='int32')
        #print(y_pred_classes, y_true_classes)
        #return tf.reduce_sum(tf.cast(tf.equal(y_pred_classes, y_true_classes), dtype='float32')) / (Y_pred.shape[0] * Y_pred.shape[1])
        return tf.reduce_sum(tf.cast(tf.equal(y_pred_classes, y_true_classes), dtype='float32')) / Y_pred.shape[0]

    def backward(self, X, A2, dA2, output=0):
        m = dA2.shape[0] 
        self.noise_matrix = 1.0 / self.noise_matrix
        if output:
            dZ2 = dA2 #* (tf.nn.sigmoid(A2) * (1 - tf.nn.sigmoid(A2))) * self.noise_matrix
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

class FeedForwardLayerWithReduction:
    def __init__(self, input_size, hidden_size, output_size, index, learning_rate=0.001, dropout=0, sparse_rate=0.9):
        def xavier_initialization(shape):
            return tf.random.uniform(shape, minval=-np.sqrt(6 / (shape[0] + shape[1])), maxval=np.sqrt(6 / (shape[0] + shape[1])), dtype='float32')
        self.W1 = xavier_initialization((input_size, hidden_size))
        self.b1 = tf.zeros((1, hidden_size), dtype='float32')
        self.sparse_rate = sparse_rate
        self.sparse_template = tf.cast(tf.random.uniform((input_size, hidden_size)) > sparse_rate, dtype='float32')
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
            return dropout_mask * tf.nn.softmax(self.Z2) * self.noise_matrix
        else:
            return dropout_mask * tf.nn.relu(self.Z2) * self.noise_matrix

    def compute_loss(self, Y_pred, Y_true):
        #return tf.reduce_mean(abs(Y_pred - Y_true))
        Y_pred = tf.clip_by_value(Y_pred, 1e-12, 1.0)
        return -tf.reduce_mean(tf.reduce_sum(Y_true * tf.math.log(Y_pred), axis=1))
        #return tf.keras.losses.categorical_crossentropy(Y_true, Y_pred)

    def compute_accuracy(self, Y_pred, Y_true):
        y_pred_classes = tf.argmax(Y_pred, axis=-1)
        y_true_classes = tf.argmax(Y_true, axis=-1)
        #y_pred_classes = tf.cast(tf.round(Y_pred), dtype='int32')
        #y_true_classes = tf.cast(tf.round(Y_true), dtype='int32')
        #print(y_pred_classes, y_true_classes)
        #return tf.reduce_sum(tf.cast(tf.equal(y_pred_classes, y_true_classes), dtype='float32')) / (Y_pred.shape[0] * Y_pred.shape[1])
        return tf.reduce_sum(tf.cast(tf.equal(y_pred_classes, y_true_classes), dtype='float32')) / Y_pred.shape[0]

    def backward(self, X, A2, dA2, output=0):
        m = dA2.shape[0] 
        self.noise_matrix = 1.0 / self.noise_matrix
        if output:
            dZ2 = dA2 #* (tf.nn.sigmoid(A2) * (1 - tf.nn.sigmoid(A2))) * self.noise_matrix
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


class FeedForwardLayerWithMiddle:
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

    def set_middle(self):
        def xavier_initialization(shape):
            return tf.random.uniform(shape, minval=-np.sqrt(6 / (shape[0] + shape[1])), maxval=np.sqrt(6 / (shape[0] + shape[1])), dtype='float32')
        self.W3 = xavier_initialization((self.W1.shape[1], self.W2.shape[0]))
        self.b3 = tf.zeros((1, self.W2.shape[0]), dtype='float32')
        self.b1 = tf.zeros((1, self.W1.shape[1]), dtype='float32')
        self.optimizer.initialize({
            'W3': self.W3, 
            'b3': self.b3,
            'b1': self.b1,
        })

    def forward(self, X, output=0):
        self.Z1 = tf.linalg.matmul(X, self.W1) + self.b1
        self.A1 = tf.nn.relu(self.Z1)
        self.Z3 = tf.linalg.matmul(self.A1, self.W3) + self.b3
        self.A3 = tf.nn.relu(self.Z3)
        self.Z2 = tf.linalg.matmul(self.A3, self.W2) + self.b2
        epsilon = 0
        self.noise_matrix = tf.random.uniform(self.Z2.shape, 1 - epsilon, 1 + epsilon)
        #self.learning_rate += random.uniform(-self.learning_rate / 9, self.learning_rate / 10)
        #self.learning_rate = max(self.learning_rate, 0.00001)
        dropout_mask = tf.cast(tf.random.uniform(self.Z2.shape) >= self.dropout, dtype='float32')
        if output:
            return dropout_mask * tf.nn.softmax(self.Z2) * self.noise_matrix
        else:
            return dropout_mask * tf.nn.relu(self.Z2) * self.noise_matrix

    def compute_loss(self, Y_pred, Y_true):
        #return tf.reduce_mean(abs(Y_pred - Y_true))
        Y_pred = tf.clip_by_value(Y_pred, 1e-12, 1.0)
        return -tf.reduce_mean(tf.reduce_sum(Y_true * tf.math.log(Y_pred), axis=1))
        #return tf.keras.losses.categorical_crossentropy(Y_true, Y_pred)

    def compute_accuracy(self, Y_pred, Y_true):
        y_pred_classes = tf.argmax(Y_pred, axis=-1)
        y_true_classes = tf.argmax(Y_true, axis=-1)
        #y_pred_classes = tf.cast(tf.round(Y_pred), dtype='int32')
        #y_true_classes = tf.cast(tf.round(Y_true), dtype='int32')
        #print(y_pred_classes, y_true_classes)
        #return tf.reduce_sum(tf.cast(tf.equal(y_pred_classes, y_true_classes), dtype='float32')) / (Y_pred.shape[0] * Y_pred.shape[1])
        return tf.reduce_sum(tf.cast(tf.equal(y_pred_classes, y_true_classes), dtype='float32')) / Y_pred.shape[0]

    def backward(self, X, A2, dA2, output=0):
        m = dA2.shape[0] 
        self.noise_matrix = 1.0 / self.noise_matrix
        if output:
            dZ2 = dA2 #* (tf.nn.sigmoid(A2) * (1 - tf.nn.sigmoid(A2))) * self.noise_matrix
        else:
            dZ2 = dA2 * tf.cast(A2 > 0, dtype='float32') * self.noise_matrix

        dA3 = tf.linalg.matmul(dZ2, tf.transpose(self.W2))
        dZ3 = dA3 * tf.cast(self.A3 > 0, dtype='float32')   
        dW3 = tf.linalg.matmul(tf.transpose(self.A1), dZ3) / m
        db3 = tf.math.reduce_sum(dZ3, axis=0, keepdims=True) / m
        dA1 = tf.linalg.matmul(dZ3, tf.transpose(self.W3))
        dZ1 = dA1 * tf.cast(self.A1 > 0, dtype='float32')
        db1 = tf.math.reduce_sum(dZ1, axis=0, keepdims=True) / m 

        mnval = -10
        mxval = 10
        db1 = tf.clip_by_value(db1, mnval, mxval)
        db1 = tf.zeros(db1.shape, dtype='float32')
        dW3 = tf.clip_by_value(dW3, mnval, mxval)
        db3 = tf.clip_by_value(db3, mnval, mxval)
        #print(dW1)

        def optimize():
            return self.optimizer.update({
                'W3': self.W3, 
                'b3': self.b3,
                'b1': self.b1,
            }, {
                'W3': dW3,
                'b3': db3,
                'b1': db1,
            })
        
        new_params = optimize()
        self.b1 = new_params['b1']
        self.W3 = new_params['W3']
        self.b3 = new_params['b3']

        return dA1
    
    def save_model(self, filepath):
        np.save(filepath + 'dense' + str(self.index) + 'W1.npy', self.W1)
        np.save(filepath + 'dense' + str(self.index) + 'b1.npy', self.b1)
        np.save(filepath + 'dense' + str(self.index) + 'W3.npy', self.W3)
        np.save(filepath + 'dense' + str(self.index) + 'b3.npy', self.b3)
        np.save(filepath + 'dense' + str(self.index) + 'W2.npy', self.W2)
        np.save(filepath + 'dense' + str(self.index) + 'b2.npy', self.b2)
        np.save(filepath + 'dense' + str(self.index) + 'adam.npy', np.array([self.optimizer.t], dtype='int32'))
    
    def load_model(self, filepath):
        self.W1 = np.load(filepath + 'dense' + str(self.index) + 'W1.npy')  
        self.b1 = np.load(filepath + 'dense' + str(self.index) + 'b1.npy')
        self.W2 = np.load(filepath + 'dense' + str(self.index) + 'W2.npy')  
        self.b2 = np.load(filepath + 'dense' + str(self.index) + 'b2.npy')
        self.W3 = np.load(filepath + 'dense' + str(self.index) + 'W3.npy')  
        self.b3 = np.load(filepath + 'dense' + str(self.index) + 'b3.npy')
        self.optimizer.initialize({
            'b1': self.b1,
            'W3': self.W3,
            'b3': self.b3,
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

content = '''

I, Billy Herrington, stand here today, humbled by the task before us, mindful of the sacrifices borne by our Nico Nico ancestors.

We are in the midst of a crisis. Nico Nico Douga is at war against a far-reaching storm of disturbance and deletion. Nico Nico's economy is badly weakened: a consequence of carelessness and irresponsibility on the part of acknowledgement, but also on the collective failure to make hard choices and to prepare for a new, MAD age.

Today, I say to you, that the challenges are real, and they are many. They will not be easily met, or in a short span of time, but know this, Nico Nico: they will be met. In reaffirming the greatness of our site, we understand that greatness is never given, our journey has never been one of shortcuts. It has not been for the faint-hearted, or who seek the fleshly pleasures. Rather, it has been the risk-takers, the wasted genii, the creators of MAD things. For us, they toiled in sweatshops, endured the lash of the spanking. Time and again, these men struggled, and sacrificed, so that we might ... LIVE BETTER.

We remain the most powerful site on the Internet, our minds no less inventive, and services no less needed than they were last week, or yesterday, or the day before the day after tomorrow. Starting today, we must pull up our pants, dust ourselves off, and begin again the work of remaking Nico Nico Douga.

Now, there are some who question the scale of our ambitions, who suggest our server system cannot tolerate too many movies. Their memories are short, for they have forgotten what Nico Nico already has done, what free men can achieve when imagination is joined to common purpose.

And so, to all the people who are watching this video, from the grandest cities, to the small villages where IKZO was born, know that Nico Nico is a friend to every man, who seeks the future of love and peace. Now we will begin to responsibly leave authorized common materials to Nico Nico people, and forge a hard-earned peace in this MAD world.

What is required of us now is a new era of responsibility. This is the price, and the promise, of Nico NiCommons citizenship. Nico Nico Douga, in the face of common dangers, in this winter of our hardship, let us remember these timeless words: ASS, WE CAN.

Let it be said by our children's children, that when we were tested by DOS attacks, when we were refused by YouTube, that we did not turn back, nor did we falter, and we carried forth that great gift of freedom be delivered, and safely to future generations.

Thank you. God bless, and God bless Nico Nico Douga.
'''

content += content
content += content

def train_custom():
    hidden_size = 500
    X = []
    Y = []
    #content = 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72'
    alphabet = sorted(list(set(content)))
    for i in range(hidden_size, hidden_size + hidden_size):
        now = ''
        for j in range(i - hidden_size, i):
            now += content[j]
        now = [alphabet.index(now[j]) / len(alphabet) for j in range(hidden_size)]
        next = alphabet.index(content[i])
        X.append(now)
        Y.append([j == next for j in range(len(alphabet))])
    X = X[:hidden_size]
    Y = Y[:hidden_size]
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')
    input_size = X.shape[1]
    print(X.shape)
    main_network = FeedForwardLayer(input_size, input_size, Y.shape[1], index=0)
    for i in range(10000):
        A2 = main_network.forward(X, output=1)
        print(main_network.compute_loss(A2, Y), main_network.compute_accuracy(A2, Y))
        if i % 100 == 0:
            main_network.save_model('./custom_models/test_model')
        if main_network.compute_loss(A2, Y) < 0.1:
            main_network.save_model('./custom_models/test_model')
            break
        dA2 = main_network.backward(X, A2, A2 - Y, output=1)
    A = copy.deepcopy(main_network.W1)
    Xs = []
    nbins = 20
    Ys = []
    for ii in range(1000):
        print(ii)
        threshold = 0.001
        sparse_rate = 0.9
        kt = 100
        #A = tf.linalg.matmul(tf.random.uniform((hidden_size, hidden_size - 100), -0.1, 0.1),
        #                     tf.random.uniform((hidden_size - 100, hidden_size), -0.1, 0.1))
        
        '''B = tf.linalg.matmul(tf.random.uniform((hidden_size, 20), -0.1, 0.1),
                             tf.random.uniform((20, hidden_size), -0.1, 0.1))
        S, u, v = tf.linalg.svd(B)
        k = 20
        B = tf.linalg.matmul(u[:, :k], tf.linalg.matmul(tf.linalg.diag(S[:k]), tf.transpose(v)[:k, :]))
        B = B.numpy()
        B = np.array(B, dtype='float64')
        B_inv = np.linalg.pinv(B)
        B_inv = tf.convert_to_tensor(B_inv, dtype='float64')
        S, u, v = tf.linalg.svd(B_inv)
        k = round(float(tf.reduce_sum(tf.cast(S / tf.math.reduce_max(S) > threshold, dtype='float32'))))
        np.save('./singular_values_b_inv.npy', S[:k])
        np.save('./left_singular_vectors_b_inv.npy', u[:, :k])
        np.save('./right_singular_vectors_b_inv.npy', tf.transpose(v)[:k, :])
        B_inv = tf.linalg.matmul(u[:, :k], tf.linalg.matmul(tf.linalg.diag(S[:k]), tf.transpose(v)[:k, :]))
        D = tf.linalg.matmul(B, tf.cast(A, dtype='float64'))
        S, u, v = tf.linalg.svd(D)
        np.save('./singular_values_d.npy', S[:k])
        np.save('./left_singular_vectors_d.npy', u[:, :k])
        np.save('./right_singular_vectors_d.npy', tf.transpose(v)[:k, :])
        D = tf.linalg.matmul(u[:, :k], tf.linalg.matmul(tf.linalg.diag(S[:k]), tf.transpose(v)[:k, :]))
        true_A1 = tf.linalg.matmul(X, A)
        approx_A1 = tf.linalg.matmul(tf.cast(X, dtype='float64'), tf.linalg.matmul(B_inv, D))
        avg = 0
        for i in range(1000):
            i1 = random.randint(0, len(true_A1) - 1)
            i2 = random.randint(0, len(true_A1[0]) - 1)
            avg += abs(float(true_A1[i1][i2]) - float(approx_A1[i1][i2]))
        if avg / 1000 < 0.01:
            Ys.append(round(float(tf.reduce_sum(tf.cast(S / tf.math.reduce_max(S) > threshold, dtype='float32')))))
        if len(Ys):
            if Ys[-1] <= 2:
                break
        continue'''
        S, u, v = tf.linalg.svd(A)
        '''print(tf.math.reduce_max(S), tf.reduce_sum(tf.cast(S / tf.math.reduce_max(S) > threshold, dtype='float32')))
        k = 100
        true_A1 = tf.linalg.matmul(X, A)
        main_network.W1 = tf.linalg.matmul(u[:, :k], tf.linalg.matmul(tf.linalg.diag(S[:k]), tf.transpose(v)[:k, :]))
        approx_A1 = tf.linalg.matmul(X, main_network.W1)
        Xs = []
        Ys = []
        avg = 0
        for i in range(10000):
            i1 = random.randint(0, len(true_A1) - 1)
            i2 = random.randint(0, len(true_A1[0]) - 1)
            avg += abs(float(true_A1[i1][i2]) - float(approx_A1[i1][i2]))
            Xs.append(float(true_A1[i1][i2]))
            Ys.append(float(approx_A1[i1][i2]))
        print('avg', avg / 10000)
        plt.plot(Xs, Ys, 'ro', markersize=0.2)
        plt.show()'''
        #A2 = main_network.forward(X, output=1)
        #print(main_network.compute_loss(A2, Y), main_network.compute_accuracy(A2, Y))
        A_s = A * tf.cast(tf.random.uniform(A.shape) > sparse_rate, dtype='float32')
        S_a = tf.linalg.matmul(A_s, tf.linalg.matmul(tf.transpose(A), tf.linalg.inv(tf.linalg.matmul(A, tf.transpose(A)))))
        #print(tf.linalg.det(S_a))# tf.linalg.det(A), tf.linalg.det(A_s))
        #print('W1 shape', A.shape)
        #print('S_a shape', S_a.shape)
        S, u, v = tf.linalg.svd(S_a)
        #print('for sparse_rate:')
        #print(S_9)
        #print(tf.math.reduce_max(S), tf.reduce_sum(tf.cast(S / tf.math.reduce_max(S) > threshold, dtype='float32')))
        k_a = round(float(tf.reduce_sum(tf.cast(S / tf.math.reduce_max(S) > threshold, dtype='float32'))))
        #k_a = max(k_a, 200)
        k_a = max(k_a, hidden_size // 5)
        k_a = 300
        M = tf.linalg.matmul(u[:, :k_a], tf.linalg.matmul(tf.linalg.diag(S[:k_a]), tf.transpose(v)[:k_a, :]))
        M = tf.linalg.matmul(u[: ,:k_a], tf.linalg.diag(S[:k_a]))
        #true_A1 = A_s
        #approx_A1 = tf.linalg.matmul(tf.linalg.matmul(M, tf.transpose(v_m)[:k_a, :]), A)
        '''Xs = []
        Ys = []
        for i in range(1000):
            i1 = random.randint(0, len(true_A1) - 1)
            i2 = random.randint(0, len(true_A1[0]) - 1)
            Xs.append(float(true_A1[i1][i2]))
            Ys.append(float(approx_A1[i1][i2]))
        plt.plot(Xs, Ys, 'ro', markersize=0.2)
        plt.show()'''
        #print('M shape', M.shape)
        M_s = M * tf.cast(tf.random.uniform(M.shape) > sparse_rate, dtype='float32')
        #print(tf.linalg.matmul(M, tf.transpose(M)))
        S_m = tf.linalg.matmul(M_s, tf.linalg.matmul(tf.transpose(M), tf.linalg.inv(tf.linalg.matmul(M, tf.transpose(M)))))
        B = tf.linalg.matmul(tf.linalg.matmul(S_m, S_a), A)
        S, u, v = tf.linalg.svd(B)
        print(S)
        #print(tf.math.reduce_max(S), tf.reduce_sum(tf.cast(S / tf.math.reduce_max(S) > threshold, dtype='float32')))
        #print('S_m shape', S_m.shape)
        S, u, v_b = tf.linalg.svd(tf.linalg.matmul(S_m, S_a))
        print(S)
        print('prod svd', tf.math.reduce_max(S), tf.reduce_sum(tf.cast(S / tf.math.reduce_max(S) > threshold, dtype='float32')))
        return 1
        k = round(float(tf.reduce_sum(tf.cast(S / tf.math.reduce_max(S) > threshold, dtype='float32'))))
        k = max(k_a, hidden_size // 50)
        k = kt
        B = tf.linalg.matmul(u[:, :k], tf.linalg.diag(S[:k]))
        #true_Am = tf.linalg.matmul(X, tf.linalg.matmul(tf.linalg.matmul(S_m, S_a), A))
        #approx_Am = tf.linalg.matmul(X, tf.linalg.matmul(tf.linalg.matmul(M, tf.transpose(v_a)[:k_a, :]), 
        #                                                 tf.linalg.matmul(A, tf.linalg.matmul(B, tf.transpose(v_m)[:k, :]))))
        #approx_Am = tf.linalg.matmul(X, tf.linalg.matmul(tf.linalg.matmul(B, tf.transpose(v_b)[:k, :]), A))
        U = tf.linalg.matmul(B, tf.transpose(v_b)[:k, :])
        U = tf.cast(U, dtype='float64')
        U_f = tf.linalg.matmul(U, tf.cast(A, dtype='float64'))
        U_f = tf.cast(U_f, dtype='float64')
        before = U_f
        S, u, v = tf.linalg.svd(U_f)
        print('U_f svd', tf.math.reduce_max(S), tf.reduce_sum(tf.cast(S / tf.math.reduce_max(S) > threshold, dtype='float32')))
        k = round(float(tf.reduce_sum(tf.cast(S / tf.math.reduce_max(S) > threshold, dtype='float32'))))
        k = kt
        U_f = tf.linalg.matmul(tf.linalg.matmul(u[:, :k], tf.linalg.diag(S[:k])), tf.transpose(v)[:k, :])
        print(tf.reduce_sum(abs(U_f - before)))
        S, u, v_u = tf.linalg.svd(U)
        U = U.numpy()
        U = np.array(U, dtype='float64')
        U_inv = np.linalg.pinv(U)
        U_inv = tf.convert_to_tensor(U_inv, dtype='float64')
        S, u, v = tf.linalg.svd(U_inv)
        print(U_inv)
        #print('U_inv svd', tf.math.reduce_max(S), tf.reduce_sum(tf.cast(S / tf.math.reduce_max(S) > threshold, dtype='float32')))
        k = round(float(tf.reduce_sum(tf.cast(S / tf.math.reduce_max(S) > threshold, dtype='float32'))))
        k = kt
        Ys.append(k)
        U_inv = tf.linalg.matmul(tf.linalg.matmul(u[:, :k], tf.linalg.diag(S[:k])), tf.transpose(v)[:k, :])
        print(Ys[-1])
        #continue
        U = tf.convert_to_tensor(U, dtype='float64')
        D = tf.linalg.matmul(U_inv, U_f)
        #print(tf.linalg.matmul(U_inv, U))
        S, u, v_d = tf.linalg.svd(D)
        S = tf.cast(S, dtype='float64')
        u = tf.cast(u, dtype='float64')
        print('D svd', tf.math.reduce_max(S), tf.reduce_sum(tf.cast(S / tf.math.reduce_max(S) > threshold, dtype='float32')))
        #k = round(float(tf.reduce_sum(tf.cast(S / tf.math.reduce_max(S) > threshold, dtype='float32'))))
        #k = max(k, hidden_size // 4)
        #D_a = tf.linalg.matmul(u[:, :k], tf.linalg.diag(S[:k]))
        #D_a = tf.cast(D_a, dtype='float64')
        #v_d = tf.cast(v_d, dtype='float64')
        #true_Am = tf.linalg.matmul(X, A)
        #approx_Am = tf.linalg.matmul(tf.cast(X, dtype='float64'), tf.linalg.matmul(D_a, tf.transpose(v_d)[:k, :]))
        true_Am = tf.linalg.matmul(X, A)
        approx_Am = tf.linalg.matmul(tf.cast(X, dtype='float64'), D)
        Xs = []
        Ys = []
        avg = 0
        for i in range(1000):
            i1 = random.randint(0, len(true_Am) - 1)
            i2 = random.randint(0, len(true_Am[0]) - 1)
            avg += abs(float(true_Am[i1][i2]) - float(approx_Am[i1][i2]))
            Xs.append(float(true_Am[i1][i2]))
            Ys.append(float(approx_Am[i1][i2]))
        print('avg', avg / 1000)
        plt.plot(Xs, Ys, 'ro', markersize=0.2)
        plt.show()
    print('min', min(Ys))
    plt.hist(Ys, bins=nbins)
    plt.show()
    return 1
    main_network.W1 = tf.cast(tf.linalg.matmul(tf.linalg.matmul(U, D_a), tf.transpose(v_d)[:k, :]), dtype='float32')
    S, _, _ = tf.linalg.svd(main_network.W1)
    print('result svd', tf.math.reduce_max(S), tf.reduce_sum(tf.cast(S / tf.math.reduce_max(S) > threshold, dtype='float32')))
    A2 = main_network.forward(X, output=1)
    print(main_network.compute_loss(A2, Y), main_network.compute_accuracy(A2, Y))
    return 1
    print('B shape', B.shape)
    middle_network = FeedForwardLayerWithMiddle(input_size, B.shape[1], Y.shape[1], index=1, learning_rate=0.00001)
    middle_network.W1 = B
    middle_network.W2 = main_network.W2
    middle_network.b2 = main_network.b2
    middle_network.set_middle()
    for i in range(30000):
        A2 = middle_network.forward(X, output=1)
        print(middle_network.compute_loss(A2, Y), middle_network.compute_accuracy(A2, Y))
        if i % 100 == 0:
            middle_network.save_model('./custom_models/test_model')
        if middle_network.compute_loss(A2, Y) < 0.1:
            middle_network.save_model('./custom_models/test_model')
            break
        dA2 = middle_network.backward(X, A2, A2 - Y, output=1)

train_custom()

def use_custom():
    hidden_size = 500
    ff = FeedForwardLayer(1, 1, 1, 0)
    ff.load_model('./custom_models/test_model')
    X = []
    Y = []
    alphabet = sorted(list(set(content)))
    for i in range(hidden_size, hidden_size + hidden_size):
        now = ''
        for j in range(i - hidden_size, i):
            now += content[j]
        now = [alphabet.index(now[j]) / len(alphabet) for j in range(hidden_size)]
        next = alphabet.index(content[i])
        X.append(now)
        Y.append([j == next for j in range(len(alphabet))])
    X = X[:hidden_size]
    Y = Y[:hidden_size]
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')
    print(len(np.load('singular_values_d.npy')))
    d = tf.linalg.matmul(np.load('./left_singular_vectors_d.npy'), tf.linalg.matmul(tf.linalg.diag(np.load('singular_values_d.npy')), np.load('right_singular_vectors_d.npy')))
    b_inv = tf.linalg.matmul(np.load('./left_singular_vectors_b_inv.npy'), tf.linalg.matmul(tf.linalg.diag(np.load('singular_values_b_inv.npy')), np.load('right_singular_vectors_b_inv.npy')))
    ff.W1 = tf.linalg.matmul(b_inv, d)
    ff.W1 = tf.cast(ff.W1, dtype='float32')
    A2 = ff.forward(X, output=1)
    print(ff.compute_loss(A2, Y), ff.compute_accuracy(A2, Y))

#use_custom()

def use():
    ff = FeedForwardLayerWithMiddle(1, 1, 1, 1)
    ff.load_model('./custom_models/test_model')
    alphabet = sorted(list(set(content)))
    now = ''
    new = ''
    for i in range(1000, 2000):
        now += content[i]
    print(len(now))
    now = list(now)
    #now[100:500] = [alphabet[random.randint(0, len(alphabet) - 1)] for i in range(400)]
    now = ''.join(now)
    print(now)
    for j in range(100):
        X = [[]]
        for i in range(len(now)):
            X[0].append(alphabet.index(now[i]) / len(alphabet))
        X = np.array(X, dtype='float32')
        now = now[1:] + alphabet[tf.argmax(ff.forward(X, output=1), axis=-1)[0]]
        new += now[-1]
    print(new)

#use()