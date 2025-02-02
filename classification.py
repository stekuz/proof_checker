import gc
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
import tracemalloc
import psutil
from tensorflow.python.profiler import trace
from guppy import hpy

sys.setrecursionlimit(100000)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
tf.random.set_seed(42)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

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
        self.m = {key: tf.Variable(tf.zeros_like(value), dtype='float32') for key, value in params.items()}
        self.v = {key: tf.Variable(tf.zeros_like(value), dtype='float32') for key, value in params.items()}

    def update(self, params, grads):
        self.t += 1     
        result = copy.deepcopy(params)
        for key in params.keys():
            self.m[key].assign(self.beta1 * self.m[key] + (1 - self.beta1) * grads[key])
            self.v[key].assign(self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2))

            self.m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            self.v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            result[key].assign(result[key] - self.learning_rate * self.m_hat / (tf.sqrt(self.v_hat) + self.epsilon))
            tf.reduce_sum(self.m[key])
            tf.reduce_sum(self.v[key])
            tf.reduce_sum(self.m_hat)
            tf.reduce_sum(self.v_hat)
            tf.reduce_sum(result[key])
        return result

    def save(self, filepath):
        np.save(filepath + 'm.npy', self.m)
        np.save(filepath + 'v.npy', self.v)
        np.save(filepath + 't.npy', self.t)

    def load(self, filepath):
        self.mn = np.load(filepath + 'm.npy', allow_pickle=True)
        self.m = {}
        self.m['W1'] = self.mn.item().get('W1')
        self.m['b1'] = self.mn.item().get('b1')
        self.vn = np.load(filepath + 'v.npy', allow_pickle=True)
        self.v = {}
        self.v['W1'] = self.vn.item().get('W1')
        self.v['b1'] = self.vn.item().get('b1')
        self.t = np.load(filepath + 't.npy', allow_pickle=True)
    
class SGD:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def initialize(self, params):
        return

    def update(self, params, grads):
        result = copy.deepcopy(params)
        for key in params:
            result[key] = result[key] + self.learning_rate * grads[key]
        return result
    
    def save(self, filepath):
        return
    
    def load(self, filepath):
        return

class BatchNormalization:
    def __init__(self, momentum=0.99, epsilon=1e-5):
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None

    def initialize(self, shape):
        self.gamma = tf.ones(shape)
        self.beta = tf.zeros(shape)
        self.running_mean = tf.zeros(shape)
        self.running_var = tf.ones(shape)

    def forward(self, x, training=True):
        if training:
            mean = tf.reduce_mean(x, axis=0)
            var = tf.reduce_mean(tf.square(x - mean), axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            x_normalized = (x - mean) / tf.sqrt(var + self.epsilon)
            self.mean = mean
            self.var = var
            return self.gamma * x_normalized + self.beta
        else:
            x_normalized = (x - self.running_mean) / tf.sqrt(self.running_var + self.epsilon)
            return self.gamma * x_normalized + self.beta

    def backward(self, x, dout):
        mean = self.mean
        var = self.var
        N, D = x.shape
        x_normalized = (x - mean) / tf.sqrt(var + self.epsilon)

        dgamma = tf.reduce_sum(dout * x_normalized, axis=0)
        dbeta = tf.reduce_sum(dout, axis=0)

        dx_normalized = dout * self.gamma
        dvar = tf.reduce_sum(dx_normalized * (x - mean) * -0.5 * tf.pow(var + self.epsilon, -1.5), axis=0)
        dmean = tf.reduce_sum(dx_normalized * -1 / tf.sqrt(var + self.epsilon), axis=0) + dvar * tf.reduce_mean(-2 * (x - mean), axis=0)

        dx = (dx_normalized / tf.sqrt(var + self.epsilon)) + (dvar * 2 / N * (x - mean)) + (dmean / N)

        self.gamma -= dgamma
        self.beta -= dbeta
        
        return dx
    
    def save(self, filepath):
        np.save(filepath + 'rm.npy', self.running_mean)
        np.save(filepath + 'rv.npy', self.running_var)
        np.save(filepath + 'gamma.npy', self.gamma)
        np.save(filepath + 'beta.npy', self.beta)

    def load(self, filepath):
        self.running_mean = np.load(filepath + 'rm.npy')
        self.running_var = np.load(filepath + 'rv.npy')
        self.gamma = np.load(filepath + 'gamma.npy')
        self.beta = np.load(filepath + 'beta.npy')

class FeedForwardLayerDouble:
    def __init__(self, input_size, hidden_size, output_size, index, learning_rate=0.001, dropout=0, l2_lambda=0):
        def xavier_initialization(shape):
            return tf.random.uniform(shape, minval=-np.sqrt(6 / (shape[0] + shape[1])), maxval=np.sqrt(6 / (shape[0] + shape[1])), dtype='float32')
        self.W1 = xavier_initialization((input_size, hidden_size))
        self.b1 = tf.zeros((1, hidden_size), dtype='float32')
        self.W2 = xavier_initialization((hidden_size, output_size))
        self.b2 = tf.zeros((1, output_size), dtype='float32')
        self.index = index
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.l2_lambda = l2_lambda
        self.batch_norm1 = BatchNormalization()
        self.batch_norm2 = BatchNormalization()
        self.batch_norm1.initialize((hidden_size, ))
        self.batch_norm2.initialize((output_size, ))
        self.optimizer = AdamOptimizer(learning_rate=learning_rate / 10, epsilon=1e-8)
        self.optimizer.initialize({
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
        })

    def forward(self, X, output=0, middle=0, training=1):
        self.Z1 = tf.linalg.matmul(X, self.W1) + self.b1
        self.B1 = self.batch_norm1.forward(self.Z1, training=training)
        self.A1 = tf.nn.relu(self.B1)
        dropout_mask = tf.cast(tf.random.uniform(self.Z1.shape) >= self.dropout, dtype='float32')
        self.A1 = self.A1 * dropout_mask
        self.Z2 = tf.linalg.matmul(self.A1, self.W2) + self.b2
        self.B2 = self.batch_norm2.forward(self.Z2, training=training)
        if output:
            return tf.nn.softmax(self.B2)
        elif middle:
            return tf.nn.sigmoid(self.B2)
        else:
            return tf.nn.relu(self.B2)

    def compute_loss(self, Y_pred, Y_true, latent=0):
        #loss = tf.reduce_sum(tf.math.sqrt(abs(Y_true - Y_pred))) / (Y_pred.shape[0] * Y_pred.shape[1])
        l2_reg_cost = 0*(self.l2_lambda / 2) * (tf.reduce_sum(tf.square(self.W1)) + tf.reduce_sum(tf.square(self.W2)))
        if latent == 0:
            loss = keras.losses.binary_crossentropy(Y_true, Y_pred)
        else:
            loss = keras.losses.mae(Y_true, Y_pred)
        return tf.reduce_mean(loss)

    def compute_accuracy(self, Y_pred, Y_true):
        #y_pred_classes = tf.cast(tf.round(Y_pred), dtype='int32')
        #y_true_classes = tf.cast(tf.round(Y_true), dtype='int32')
        
        #y_pred_classes = tf.math.argmax(Y_pred, axis=-1)
        #top_k = 1
        #_, y_pred_classes = tf.math.top_k(Y_pred, top_k)
        #y_true_classes = tf.cast(tf.math.argmax(Y_true, axis=-1), dtype='int32')
        #y_true_classes = tf.broadcast_to(tf.reshape(y_true_classes, (y_true_classes.shape[0], 1)),
        #                              (y_true_classes.shape[0], top_k))
        y_pred_classes = tf.cast(tf.round(Y_pred), dtype='int32')
        y_true_classes = tf.cast(tf.round(Y_true), dtype='int32')
        correct_predictions = tf.equal(y_true_classes, y_pred_classes)
        #threshold = 5
        #correct_predictions = abs(y_true_classes - y_pred_classes) < threshold
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def backward(self, X, A2, dA2, output=0, middle=0):
        m = dA2.shape[1] 
        if output:
            dZ2 = dA2 
        elif middle:
            dZ2 = dA2 * tf.nn.sigmoid(A2) * (1 - tf.nn.sigmoid(A2))
        else:
            dZ2 = dA2 * tf.cast(A2 > 0, dtype='float32')
        dB2 = self.batch_norm2.backward(self.Z2, dZ2)
        dW2 = tf.linalg.matmul(tf.transpose(self.A1), dB2) / m + (self.l2_lambda * self.W2) / m
        db2 = tf.math.reduce_sum(dZ2, axis=0, keepdims=True) / m 

        
        dA1 = tf.linalg.matmul(dZ2, tf.transpose(self.W2))
        m = dA1.shape[1]
        dZ1 = dA1 * tf.cast(self.A1 > 0, dtype='float32')
        dB1 = self.batch_norm1.backward(self.Z1, dZ1)
        dW1 = tf.linalg.matmul(tf.transpose(X), dB1) / m + (self.l2_lambda * self.W1) / m
        db1 = tf.math.reduce_sum(dZ1, axis=0, keepdims=True) / m 
        dA0 = tf.linalg.matmul(dZ1, tf.transpose(self.W1))

        mnval = -1
        mxval = 1
        dW1 = tf.clip_by_value(dW1, mnval, mxval)
        db1 = tf.clip_by_value(db1, mnval, mxval)
        dW2 = tf.clip_by_value(dW2, mnval, mxval)
        db2 = tf.clip_by_value(db2, mnval, mxval)
        if self.index == 'sscomb1':
            pass
            #print(dW1)
        #if tf.reduce_max(dW1) > 0.1 or tf.reduce_max(dW1) < 0.00001:
        #    print(self.index, tf.reduce_max(dW1), tf.reduce_min(dW1))

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

        return dA0
    
    def save_model(self, filepath):
        self.batch_norm1.save(filepath + 'bn1_' + str(self.index))
        self.batch_norm2.save(filepath + 'bn2_' + str(self.index))
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

class FeedForwardLayer:
    def __init__(self, input_size, output_size, index, learning_rate=0.001, dropout=0, l2_lambda=0, skipping=0):
        def xavier_initialization(shape):
            return tf.random.uniform(shape, minval=-np.sqrt(6 / (shape[0] + shape[1])), maxval=np.sqrt(6 / (shape[0] + shape[1])), dtype='float32')
        self.W1 = tf.Variable(xavier_initialization((input_size, output_size)), dtype='float32')
        self.b1 = tf.Variable(tf.zeros((1, output_size), dtype='float32'), dtype='float32')
        self.index = index
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.l2_lambda = l2_lambda
        self.skipping = skipping
        self.selu_l = 1.0507
        self.selu_a = 1.6733
        self.batch_norm = BatchNormalization()
        self.batch_norm.initialize((output_size, ))
        self.optimizer = AdamOptimizer(learning_rate=learning_rate / 10, epsilon=1e-8)
        #self.optimizer = SGD(learning_rate=learning_rate)
        self.optimizer.initialize({
            'W1': self.W1,
            'b1': self.b1,
        })

    def forward(self, X, output=0, middle=0, training=1):
        self.dropout_mask = tf.cast(tf.random.uniform(X.shape) > self.dropout * training, dtype='float32')
        if self.skipping:
            return X
        self.Z1 = tf.linalg.matmul(X, self.W1) + self.b1
        self.B1 = self.batch_norm.forward(self.Z1, training=training)
        #self.dropout_mask = tf.cast(tf.random.uniform(self.B1.shape) > self.dropout * training, dtype='float32')
        if output:
            return tf.nn.tanh(self.B1)
        elif middle:
            return tf.nn.sigmoid(self.B1)
        else:
            return tf.nn.selu(self.B1)

    def compute_loss(self, Y_pred, Y_true, latent=1, mean=1):
        #loss = tf.reduce_sum(tf.math.sqrt(abs(Y_true - Y_pred))) / (Y_pred.shape[0] * Y_pred.shape[1])
        if latent == 0:
            loss = keras.losses.binary_crossentropy(Y_true, Y_pred)
        else:
            loss = keras.losses.mae(Y_true, Y_pred)
        if mean:
            return tf.reduce_mean(loss)
        else:
            return loss

    def compute_accuracy(self, Y_pred, Y_true):
        #y_pred_classes = tf.cast(tf.round(Y_pred), dtype='int32')
        #y_true_classes = tf.cast(tf.round(Y_true), dtype='int32')
        
        #y_pred_classes = tf.math.argmax(Y_pred, axis=-1)
        #top_k = 1
        #_, y_pred_classes = tf.math.top_k(Y_pred, top_k)
        #y_true_classes = tf.cast(tf.math.argmax(Y_true, axis=-1), dtype='int32')
        #y_true_classes = tf.broadcast_to(tf.reshape(y_true_classes, (y_true_classes.shape[0], 1)),
        #                              (y_true_classes.shape[0], top_k))
        y_pred_classes = tf.cast(tf.reduce_sum(Y_pred, axis=-1) > 0, dtype='int32')
        y_true_classes = tf.cast(tf.round(Y_true), dtype='int32')
        correct_predictions = tf.equal(y_true_classes, y_pred_classes)
        #threshold = 5
        #correct_predictions = abs(y_true_classes - y_pred_classes) < threshold
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    def backward(self, X, A1, dA1, output=0, middle=0):
        if self.skipping:
            return dA1
        m = dA1.shape[1] 
        if output:
            dZ1 = dA1 * (1 - tf.nn.tanh(A1) ** 2)
        elif middle:
            dZ1 = dA1 * tf.nn.sigmoid(A1) * (1 - tf.nn.sigmoid(A1))
        else:
            dZ1 = dA1 * (tf.cast(A1 > 0, dtype='float32') * self.selu_l +
                         tf.cast(A1 < 0, dtype='float32') * self.selu_l * self.selu_a * tf.math.exp(A1 * tf.cast(A1 < 0, dtype='float32')))
        dB1 = self.batch_norm.backward(self.Z1, dZ1)
        dW1 = tf.linalg.matmul(tf.transpose(X), dB1) / m + (self.l2_lambda * self.W1) / m
        db1 = tf.math.reduce_sum(dZ1, axis=0, keepdims=True) / m 
        dA0 = tf.linalg.matmul(dZ1, tf.transpose(self.W1))

        mnval = -1
        mxval = 1
        dW1 = tf.clip_by_value(dW1, mnval, mxval)
        db1 = tf.clip_by_value(db1, mnval, mxval)

        def optimize():
            return self.optimizer.update({
                'W1': self.W1,
                'b1': self.b1,
            }, {
                'W1': dW1,
                'b1': db1,
            })
        
        new_params = optimize()
        self.W1.assign(new_params['W1'])
        self.b1.assign(new_params['b1'])
        return dA0
    
    def save_model(self, filepath):
        self.batch_norm.save(filepath + 'bn_' + str(self.index))
        np.save(filepath + 'dense' + str(self.index) + 'W1.npy', self.W1)
        np.save(filepath + 'dense' + str(self.index) + 'b1.npy', self.b1)
        self.optimizer.save(filepath + 'adam' + str(self.index))
    
    def load_model(self, filepath):
        self.batch_norm.load(filepath + 'bn_' + str(self.index))
        self.W1 = np.load(filepath + 'dense' + str(self.index) + 'W1.npy')
        self.b1 = np.load(filepath + 'dense' + str(self.index) + 'b1.npy')
        self.optimizer.load(filepath + 'adam' + str(self.index))
        
class ChaosTransformLayer:
    def __init__(self, input_shape, hidden_size, output_size, index, learning_rate=0.001, total_input=10, dropout=0, skipping=1):
        self.index = index
        self.total_input = total_input
        self.dense_input = []
        self.dense_middle = []
        for i in range(self.total_input):
            self.dense_input.append(FeedForwardLayer(input_size=input_shape[1] * 2, output_size=hidden_size, index='ssoriginp' + str(index) + '_' + str(i), learning_rate=learning_rate, dropout=dropout))
            self.dense_middle.append(FeedForwardLayer(input_size=hidden_size, output_size=hidden_size, index='ssorigmid' + str(index) + '_' + str(i), learning_rate=learning_rate, skipping=skipping, dropout=dropout))
        self.dense_combinator = FeedForwardLayer(input_size=self.total_input * hidden_size, output_size=output_size, index='sscomb' + str(index), learning_rate=learning_rate)
        #self.chaos_0 = np.random.rand(2, 2)
        #self.chaos_large = np.dot(np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype='int32'), 
        #                          np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]], dtype='int32'))
        self.chaos_large = np.array([[5, 2], [7, 3]], dtype='int64')
        #self.chaos_large = np.array([[1, 0], [0, 1]], dtype='float32')
        self.chaos_2_1 = np.array([[2, 1], [1, 1]], dtype='float32')
        #self.mat_1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='int32')
        self.mat_1 = np.array([[1, 0], [0, 1]], dtype='int64')
        self.chaos_1 = []
        self.chaos_1_inv = []
        self.mat_2 = np.array([[1, 0], [0, 1]], dtype='float32')
        self.chaos_2 = []
        for i in range(self.total_input):
            self.chaos_1.append(tf.convert_to_tensor(copy.deepcopy(self.mat_1), dtype='float32'))
            #print(tf.linalg.det(self.chaos_1[-1]))
            #self.chaos_1_inv.append(tf.convert_to_tensor(copy.deepcopy(np.linalg.inv(self.mat_1)), dtype='float32'))
            #self.chaos.append(tf.random.uniform((2, 2)))
            self.mat_1 = np.dot(self.mat_1, self.chaos_large)
            self.chaos_2.append(tf.convert_to_tensor(copy.deepcopy(self.mat_2), dtype='float32'))
            self.mat_2 = np.dot(self.mat_2, self.chaos_2_1)
        print(self.mat_1, self.mat_2)
        self.X = None
        self.input_norm_matrix = None

    def attention(self, Q, K, V):
        return tf.linalg.matmul(tf.nn.softmax(tf.linalg.matmul(Q, tf.transpose(K)) / (K.shape[1])** 0.5), V)
    
    def forward_chaos(self, X_original, double_matrix=0, flattened=0):
        if flattened:
            div_len = X_original.shape[1] // 2
            X_original = tf.stack([
                X_original[:, :div_len],
                X_original[:, div_len:]
            ], axis=-1)

        X_1 = []
        X_2 = []
        X = []
        for i in range(self.total_input):
            transformed_X1 = tf.linalg.matmul(X_original, self.chaos_1[i]) % 1
            X_1.append(transformed_X1)

            X_1[i] = tf.concat([tf.squeeze(X_1[i][:, :, :1]),
                                tf.squeeze(X_1[i][:, :, 1:])], axis=1)

            if double_matrix:
                transformed_X2 = tf.linalg.matmul(X_original, self.chaos_2[i]) % 1
                X_2.append(tf.squeeze(transformed_X2[:, :, 1:]))
                X.append((X_1[i] + X_2[i]) % 1)
            else:
                X.append(X_1[i])

        return tf.convert_to_tensor(X, dtype='float32')
    
    def backward_chaos(self, X_original, double_matrix=0):
        div_len = X_original.shape[1] // 2
        X_original = tf.stack([
            X_original[:, :div_len],
            X_original[:, div_len:],
        ], axis=-1)

        X_1 = []
        X_2 = []
        X = []

        for i in range(self.total_input):
            transformed_X1 = tf.linalg.matmul(X_original, self.chaos_1_inv[i]) % 1
            X_1.append(transformed_X1)

            X_1[i] = tf.concat([tf.squeeze(X_1[i][:, :, :1]),
                                tf.squeeze(X_1[i][:, :, 1:])], axis=1)

            if double_matrix:
                transformed_X2 = tf.linalg.matmul(X_original, self.chaos_2[i]) % 1
                X_2.append(tf.squeeze(transformed_X2[:, :, 1:]))
                X.append((X_1[i] + X_2[i]) % 1)
            else:
                X.append(X_1[i])

        return tf.convert_to_tensor(X, dtype='float32')
    
    def forward(self, X_original, double_matrix=0, new_X=0, output=0, middle=0, training=1, skip_X=None):
        if not (skip_X is None) and 0:
            X_original += skip_X
        if self.X is None:
            self.X = self.forward_chaos(X_original)
        X = self.X
        self.X = None
        if new_X:
            X = self.forward_chaos(X_original)
        Y_pred = []
        self.A2inp = []
        self.A2mid = []
        #X = [self.attention(X[i], X[i], X[i]) for i in range(self.total_input)]
        for i in range(self.total_input):
            self.A2inp.append(self.dense_input[i].forward(X[i], training=training))
            self.A2mid.append(self.dense_middle[i].forward(self.A2inp[i], training=training))
            Y_pred.append(self.A2mid[i])
        Y_pred_to_combinator = Y_pred[0]
        for i in range(1, self.total_input):
            Y_pred_to_combinator = tf.concat((Y_pred_to_combinator, Y_pred[i]), axis=-1)
        X_skip_to_combinator = tf.concat([X[i] for i in range(self.total_input)], axis=0)
        X_skip_to_combinator = tf.linalg.matmul(X_skip_to_combinator, tf.ones((X_skip_to_combinator.shape[1], Y_pred_to_combinator.shape[1]))) % 1
        Y_pred_final = self.dense_combinator.forward(Y_pred_to_combinator, output=output, middle=middle, training=training)
        return Y_pred_final, X_original
    
    def backward(self, X_original, A2, dA2, double_matrix=0, new_X=0, output=0, middle=0, skip_X=None):
        if self.X is None:
            self.X = self.forward_chaos(X_original)
        X = self.X
        self.X = None
        if new_X:
            X = self.forward_chaos(X_original)
        #X = [self.attention(X[i], X[i], X[i]) for i in range(self.total_input)]
        self.A2 = self.A2mid[0]
        for i in range(1, self.total_input):
            self.A2 = tf.concat((self.A2, self.A2mid[i]), axis=1)
        A2 = self.dense_combinator.forward(self.A2, output=output, middle=middle)
        dA2_final = self.dense_combinator.backward(self.A2, A2, dA2, output=output, middle=middle)
        dA2 = self.dense_middle[0].backward(self.A2inp[0], self.A2mid[0], dA2_final[:, 0 * self.dense_input[0].W1.shape[1]:(0 + 1) * self.dense_input[0].W1.shape[1]])
        dA2 = self.dense_input[0].backward(X[0], self.A2inp[0], dA2)
        dA0 = (dA2 * X[0])[:, :mxlen] + (dA2 * X[0])[:, mxlen:]
        dX = tf.convert_to_tensor([self.forward_chaos(skip_X, flattened=0)[i] for i in range(self.total_input)], dtype='float32')
        for i in range(1, self.total_input):
            dA2 = self.dense_middle[i].backward(self.A2inp[i], self.A2mid[i], dA2_final[:, i * self.dense_input[i].W1.shape[1]:(i + 1) * self.dense_input[i].W1.shape[1]])
            dA2 = self.dense_input[i].backward(X[i], self.A2inp[i], dA2)
            dA0 += (dA2 * dX[i])[:, :mxlen] + (dA2 * dX[i])[:, mxlen:]
        return dA0
    
    def save_model(self, filepath):
        np.save('./chaos_transform_matrix.npy', self.chaos_1)
        np.save(filepath + str(self.index) + 'norm_matrix.npy', self.input_norm_matrix)
        for i in range(self.total_input):
            self.dense_input[i].save_model(filepath)
            self.dense_middle[i].save_model(filepath)
        self.dense_combinator.save_model(filepath)
    
    def load_model(self, filepath):
        self.chaos_1 = np.load('./chaos_transform_matrix.npy')
        self.input_norm_matrix = np.load(filepath + str(self.index) + 'norm_matrix.npy')
        for i in range(self.total_input):
            self.dense_input[i].load_model(filepath)
            self.dense_middle[i].load_model(filepath)
        self.dense_combinator.load_model(filepath)

all_tokens = {}
mxlen = 0
mxlen = int(open('./mxlen_imdb.txt', 'r').readlines()[0])
all_tokens_list = open('./all_tokens_imdb.txt', 'r').readlines()
all_tokens_list = [token[:-1] for token in all_tokens_list]
all_tokens_list = sorted(all_tokens_list)
for i in range(len(all_tokens_list)):
    all_tokens[all_tokens_list[i]] = i

def train_representation_chaos_imdb():
    global mxlen
    batches_selected = 50
    validation_size = 500
    validation_x = np.load('/home/user/Desktop/datasets/imdb_50k_x_5.npy')[:validation_size]
    validation_y = np.load('/home/user/Desktop/datasets/imdb_50k_y_5.npy')[:validation_size]
    validation_y = validation_y.tolist()
    for i in range(len(validation_y)):
        validation_y[i] = [validation_y[i][1]]
    validation_y = np.array(validation_y, dtype='float32')
    validation_x = []
    learning_rate = 0.001
    hidden_size = 100
    total_input = 3
    batch_size = 300
    n = 4
    middle = 1
    dropout = 0
    skipping = 1
    layers_to_latent = []
    for i in range(n):
        layers_to_latent.append(ChaosTransformLayer((0, mxlen), hidden_size, mxlen, i, learning_rate=learning_rate / 1, total_input=total_input, skipping=skipping))
    layers_to_latent.append(ChaosTransformLayer((0, mxlen), hidden_size, mxlen, n, learning_rate=learning_rate / 1, total_input=total_input, dropout=dropout, skipping=skipping))
    #hidden_size *= total_input
    #dense_middle1 = FeedForwardLayerDouble(hidden_size, hidden_size, hidden_size, 1, learning_rate=learning_rate)
    #dense_middle2 = FeedForwardLayerDouble(hidden_size, hidden_size, hidden_size, 2, learning_rate=learning_rate)
    #dense_middle3 = FeedForwardLayerDouble(hidden_size, hidden_size, hidden_size, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayerDouble(hidden_size, hidden_size, hidden_size, 4, learning_rate=learning_rate)
    print(f'''
    learning_rate = {learning_rate}
    hidden_size = {hidden_size}
    total_input = {total_input}
    batch_size = {batch_size}
    n = {n}
    middle = {middle}
    dropout = {dropout}
    training_samples = {batches_selected * batch_size}
    validation_size = {validation_size}
    skipping = {skipping}
    ''')
    filepath = './custom_models/test_model'
    #for i in range(n + 1):
    #    layers_to_latent[i].load_model(filepath)
    loss_graph_x = []
    loss_graph_y = []
    accuracy_graph_x = []
    accuracy_graph_y = []
    avg_loss = [0 for i in range(100)]
    avg_accuracy = [0 for i in range(100)]
    threshold_loss = 0
    sequence_order_padding = []
    for i in range(mxlen):
        sequence_order_padding.append([i / mxlen])
    sequence_order_padding = tf.convert_to_tensor(sequence_order_padding, dtype='float32')
    sequence_mxlen_padding = []
    for i in range(mxlen):
        sequence_mxlen_padding.append([i / mxlen, len(all_tokens) / (len(all_tokens) + 1)])
    sequence_mxlen_padding = tf.convert_to_tensor(sequence_mxlen_padding, dtype='float32')

    def pad_arrays(X, P):
        padded_X = []
        for i in range(len(X)):
            padded_X.append(tf.concat((X[i], P[len(X[i]):]), axis=0))
        return tf.convert_to_tensor(padded_X, dtype='float32')
    
    def make_categorical(Y):
        categorical = []
        for i in range(len(Y)):
            categorical.append(tf.cast(tf.equal(tf.range(len(all_tokens) + 1, dtype='int32'),
                                                tf.broadcast_to(tf.convert_to_tensor(int(tf.round(Y[i])), dtype='int32'), (len(all_tokens) + 1, ))), dtype='float32'))
        return tf.convert_to_tensor(categorical, dtype='float32')

    def trainint_loop(epoch):
        start_time = time.time()
        loss = 0
        accuracy = 0
        for i in range(batches_selected):
            X = np.load(f'/home/user/Desktop/batches/training_batches_x_{i}.npy', mmap_mode='r')
            Y = np.load(f'/home/user/Desktop/batches/training_batches_y_{i}.npy', mmap_mode='r')
            Y = tf.convert_to_tensor(Y, dtype='float32')
            Y = Y[:, 1:]
            Y = tf.reshape(Y, (-1, 1))
            X_col_1 = tf.stack([x[:, 1] for x in X], axis=0)
            mask_0 = tf.broadcast_to(tf.equal(Y, 0), X_col_1.shape)
            mask_0 = tf.cast(mask_0, dtype='float32')
            mask_1 = tf.broadcast_to(tf.not_equal(Y, 0), X_col_1.shape)
            mask_1 = tf.cast(mask_1, dtype='float32')
            X_0 = X_col_1 * mask_0
            X_1 = X_col_1 * mask_1
            sum_X_0 = tf.reduce_sum(X_0, axis=0)
            len_X_0 = tf.cast(tf.shape(X_0)[0], dtype=tf.float32)
            Y_0 = (sum_X_0 + len_X_0 * X_0) / (2 * len_X_0)
            sum_X_1 = tf.reduce_sum(X_1, axis=0)
            len_X_1 = tf.cast(tf.shape(X_1)[0], dtype=tf.float32)
            Y_1 = (sum_X_1 + len_X_1 * X_1) / (2 * len_X_1)
            Y = Y_0 + Y_1
            Y_pred_chaos = [0]
            skip_X = [0]
            Y_pred_chaos[0], skip_X[0] = layers_to_latent[0].forward(X, new_X=1, middle=middle)
            skip_X_orig = tf.concat((tf.zeros((X.shape[0], X.shape[1], 1)), X[:, :, 1:]), axis=2)
            Y_pred_chaos_with_order = []
            Y_pred_chaos_with_order.append(tf.concat((tf.broadcast_to(sequence_order_padding, (Y_pred_chaos[-1].shape[0], Y_pred_chaos[-1].shape[1], 1)),
                                                      tf.reshape(Y_pred_chaos[-1], (Y_pred_chaos[-1].shape[0], Y_pred_chaos[-1].shape[1], 1))), axis=2))
            for j in range(1, len(layers_to_latent) - 1):
                Y_pred_chaos.append(0)
                skip_X.append(0)
                Y_pred_chaos[-1], skip_X[-1] = layers_to_latent[j].forward(Y_pred_chaos_with_order[-1], new_X=1, middle=middle, skip_X=skip_X_orig)
                Y_pred_chaos_with_order.append(tf.concat((tf.broadcast_to(sequence_order_padding, (Y_pred_chaos[-1].shape[0], Y_pred_chaos[-1].shape[1], 1)),
                                                      tf.reshape(Y_pred_chaos[-1], (Y_pred_chaos[-1].shape[0], Y_pred_chaos[-1].shape[1], 1))), axis=2))
            Y_pred_final, _ = layers_to_latent[-1].forward(Y_pred_chaos_with_order[-1], new_X=1, output=1, skip_X=skip_X_orig)
            loss = (loss * i + dense_output.compute_loss(Y_pred_final, Y, latent=1)) / (i + 1)
            accuracy = (accuracy * i + dense_output.compute_accuracy(Y_pred_final, Y)) / (i + 1)
            if math.isnan(loss):
                return 1 / 0
            dA2 = (Y_pred_final - Y) / batch_size
            dA2 = layers_to_latent[-1].backward(Y_pred_chaos_with_order[-1], Y_pred_final, dA2, new_X=1, output=1)#, skip_X=skip_X[-1])
            for j in range(len(Y_pred_chaos) - 1, 0, -1):
                dA2 = layers_to_latent[j].backward(Y_pred_chaos_with_order[j - 1], Y_pred_chaos[j], dA2, new_X=1, middle=middle)#, skip_X=skip_X[i - 1])
            dA2 = layers_to_latent[0].backward(X, Y_pred_chaos[0], dA2, new_X=1, middle=middle)
        avg_loss[epoch % 100] = loss
        avg_accuracy[epoch % 100] = accuracy
        val_accuracy = 0
        if not (loss < threshold_loss or accuracy > 0.98):
            print(f'epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Time: {time.time() - start_time}')
            print(f'average loss (per 100): {sum(avg_loss) / 100}, average accuracy: {sum(avg_accuracy) / 100}')
        if (epoch % 100 < 20 or epoch % 100 == 0 or loss < threshold_loss or accuracy > 0.98 or 1) and len(validation_x) > 0:
            X = validation_x
            Y = validation_y
            Y_pred_chaos = [0]
            skip_X = [0]
            Y_pred_chaos[0], skip_X[0] = layers_to_latent[0].forward(X, new_X=1, middle=middle, training=0)
            skip_X_orig = tf.concat((tf.zeros((X.shape[0], X.shape[1], 1)), X[:, :, 1:]), axis=2)
            Y_pred_chaos_with_order = []
            Y_pred_chaos_with_order.append(tf.concat((tf.broadcast_to(sequence_order_padding, (Y_pred_chaos[-1].shape[0], Y_pred_chaos[-1].shape[1], 1)),
                                                      tf.reshape(Y_pred_chaos[-1], (Y_pred_chaos[-1].shape[0], Y_pred_chaos[-1].shape[1], 1))), axis=2))
            for j in range(1, len(layers_to_latent) - 1):
                Y_pred_chaos.append(0)
                skip_X.append(0)
                Y_pred_chaos[-1], skip_X[-1] = layers_to_latent[j].forward(Y_pred_chaos_with_order[-1], new_X=1, middle=middle, skip_X=skip_X_orig, training=0)
                Y_pred_chaos_with_order.append(tf.concat((tf.broadcast_to(sequence_order_padding, (Y_pred_chaos[-1].shape[0], Y_pred_chaos[-1].shape[1], 1)),
                                                      tf.reshape(Y_pred_chaos[-1], (Y_pred_chaos[-1].shape[0], Y_pred_chaos[-1].shape[1], 1))), axis=2))
            Y_pred_final, _ = layers_to_latent[-1].forward(Y_pred_chaos_with_order[-1], new_X=1, output=1, skip_X=skip_X_orig, training=0)
            val_loss = dense_output.compute_loss(Y_pred_final, Y)
            val_accuracy = dense_output.compute_accuracy(Y_pred_final, Y)
            print(f'    epoch: {epoch}, validation loss: {val_loss}, validation accuracy: {val_accuracy}')
        if epoch % 100 == 0 or (loss < threshold_loss or accuracy > 0.98) and 0 or val_accuracy > 0.98 or 1:
            for i in range(len(layers_to_latent)):
                layers_to_latent[i].save_model(filepath)
            dense_output.save_model(filepath)
        loss_graph_x.append(epoch)
        loss_graph_y.append(loss)
        accuracy_graph_x.append(epoch)
        accuracy_graph_y.append(accuracy)
        return (loss, accuracy, val_accuracy)
    
    def print_memory_usage():
        process = psutil.Process()
        print(f"Memory Usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    
    
    for epoch in range(1,30001):
        logdir = f'./logs/memtrace_{epoch}_'
        #print(len(tf.compat.v1.get_default_graph().get_operations()))
        print_memory_usage()
        #tf.profiler.experimental.start(logdir)
        loss, accuracy, val_accuracy = trainint_loop(epoch)
        np.empty((0,))
        gc.collect()
        #tf.keras.backend.clear_session()
        #tf.profiler.experimental.stop()
        if (loss < threshold_loss or accuracy > 0.999) and 0 or val_accuracy > 0.98:
            break
    figure, axis = plt.subplots(1, 2)
    axis[0].plot(loss_graph_x, loss_graph_y)
    axis[0].set_title('loss')
    axis[1].plot(accuracy_graph_x, accuracy_graph_y)
    axis[1].set_title('accuracy')
    plt.show()

def train_chaos_words_imdb():
    global mxlen
    batches_selected = 10
    validation_size = 500
    validation_x = np.load('/home/user/Desktop/datasets/imdb_50k_x_5.npy')[:validation_size]
    validation_y = np.load('/home/user/Desktop/datasets/imdb_50k_y_5.npy')[:validation_size]
    validation_y = validation_y.tolist()
    for i in range(len(validation_y)):
        validation_y[i] = [validation_y[i][1]]
    validation_y = np.array(validation_y, dtype='float32')
    learning_rate = 0.001
    hidden_size = 100
    total_input = 2
    batch_size = 300
    n = 2
    middle = 1
    dropout = 0
    skipping = 0
    filepath = './custom_models/test_model'

    layers = []
    for i in range(n):
        layers.append(ChaosTransformLayer((0, mxlen), hidden_size, mxlen, 10 + i, learning_rate=learning_rate / 1, total_input=total_input, skipping=skipping))
    layers.append(ChaosTransformLayer((0, mxlen), hidden_size, 1, 10 + n, learning_rate=learning_rate / 1, total_input=total_input, dropout=dropout, skipping=skipping))
    dense_output = FeedForwardLayerDouble(hidden_size, hidden_size, hidden_size, 4, learning_rate=learning_rate)
    print(f'''
    learning_rate = {learning_rate}
    hidden_size = {hidden_size}
    total_input = {total_input}
    batch_size = {batch_size}
    n = {n}
    middle = {middle}
    dropout = {dropout}
    training_samples = {batches_selected * batch_size}
    validation_size = {validation_size}
    skipping = {skipping}
    ''')
    #for i in range(n + 1):
    #    layers[i].load_model(filepath)
    loss_graph_x = []
    loss_graph_y = []
    accuracy_graph_x = []
    accuracy_graph_y = []
    avg_loss = [0 for i in range(100)]
    avg_accuracy = [0 for i in range(100)]
    threshold_loss = 0
    sequence_order_padding = []
    for i in range(mxlen):
        sequence_order_padding.append([i / mxlen])
    sequence_order_padding = tf.convert_to_tensor(sequence_order_padding, dtype='float32')
    sequence_mxlen_padding = []
    for i in range(mxlen):
        sequence_mxlen_padding.append([i / mxlen, len(all_tokens) / (len(all_tokens) + 1)])
    sequence_mxlen_padding = tf.convert_to_tensor(sequence_mxlen_padding, dtype='float32')

    def pad_arrays(X, P):
        padded_X = []
        for i in range(len(X)):
            padded_X.append(tf.concat((X[i], P[len(X[i]):]), axis=0))
        return tf.convert_to_tensor(padded_X, dtype='float32')
    
    def make_categorical(Y):
        categorical = []
        for i in range(len(Y)):
            categorical.append(tf.cast(tf.equal(tf.range(len(all_tokens) + 1, dtype='int32'),
                                                tf.broadcast_to(tf.convert_to_tensor(int(tf.round(Y[i])), dtype='int32'), (len(all_tokens) + 1, ))), dtype='float32'))
        return tf.convert_to_tensor(categorical, dtype='float32')

    def training_loop(epoch):
        start_time = time.time()
        loss = 0
        accuracy = 0
        for i in range(batches_selected):
            X = np.load(f'/home/user/Desktop/batches/training_batches_x_{i}.npy', mmap_mode='r')
            X = tf.convert_to_tensor(X, dtype='float32')
            Y = np.load(f'/home/user/Desktop/batches/training_batches_y_{i}.npy', mmap_mode='r')
            Y = tf.convert_to_tensor(Y, dtype='float32')
            Y = Y[:, 1:]
            Y = tf.reshape(Y, (-1, 1))
            Y_pred_chaos = [0]
            skip_X = [0]
            Y_pred_chaos[0], skip_X[0] = layers[0].forward(X, new_X=1, middle=middle)
            skip_X_orig = tf.concat((tf.zeros((X.shape[0], X.shape[1], 1)), X[:, :, 1:]), axis=2)
            Y_pred_chaos_with_order = []
            Y_pred_chaos_with_order.append(tf.concat((tf.broadcast_to(sequence_order_padding, (Y_pred_chaos[-1].shape[0], Y_pred_chaos[-1].shape[1], 1)),
                                                      tf.reshape(Y_pred_chaos[-1], (Y_pred_chaos[-1].shape[0], Y_pred_chaos[-1].shape[1], 1))), axis=2))
            for j in range(1, len(layers) - 1):
                Y_pred_chaos.append(0)
                skip_X.append(0)
                Y_pred_chaos[-1], skip_X[-1] = layers[j].forward(Y_pred_chaos_with_order[-1], new_X=1, middle=middle, skip_X=skip_X_orig)
                Y_pred_chaos_with_order.append(tf.concat((tf.broadcast_to(sequence_order_padding, (Y_pred_chaos[-1].shape[0], Y_pred_chaos[-1].shape[1], 1)),
                                                      tf.reshape(Y_pred_chaos[-1], (Y_pred_chaos[-1].shape[0], Y_pred_chaos[-1].shape[1], 1))), axis=2))
            Y_pred_final, _ = layers[-1].forward(Y_pred_chaos_with_order[-1], new_X=1, output=1, skip_X=skip_X_orig)
            loss = (loss * i + dense_output.compute_loss(Y_pred_final, Y)) / (i + 1)
            accuracy = (accuracy * i + dense_output.compute_accuracy(Y_pred_final, Y)) / (i + 1)
            if math.isnan(loss):
                return 1 / 0
            dA2 = (Y_pred_final - Y) / batch_size
            dA2 = layers[-1].backward(Y_pred_chaos_with_order[-1], Y_pred_final, dA2, new_X=1, output=1, skip_X=skip_X_orig)#, skip_X=skip_X[-1])
            for j in range(len(Y_pred_chaos) - 1, 0, -1):
                dA2 = layers[j].backward(Y_pred_chaos_with_order[j - 1], Y_pred_chaos[j], dA2, new_X=1, middle=middle, skip_X=skip_X_orig)#, skip_X=skip_X[i - 1])
            dA2 = layers[0].backward(X, Y_pred_chaos[0], dA2, new_X=1, middle=middle, skip_X=skip_X_orig)
        avg_loss[epoch % 100] = loss
        avg_accuracy[epoch % 100] = accuracy
        val_accuracy = 0
        if not (loss < threshold_loss or accuracy > 0.98):
            print(f'epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Time: {time.time() - start_time}')
            print(f'average loss (per 100): {sum(avg_loss) / 100}, average accuracy: {sum(avg_accuracy) / 100}')
        if (epoch % 100 < 20 or epoch % 100 == 0 or loss < threshold_loss or accuracy > 0.98 or 1) and len(validation_x) > 0:
            X = validation_x
            X = tf.convert_to_tensor(X)
            Y = validation_y
            Y_pred_chaos = [0]
            skip_X = [0]
            Y_pred_chaos[0], skip_X[0] = layers[0].forward(X, new_X=1, middle=middle, training=0)
            skip_X_orig = tf.concat((tf.zeros((X.shape[0], X.shape[1], 1)), X[:, :, 1:]), axis=2)
            Y_pred_chaos_with_order = []
            Y_pred_chaos_with_order.append(tf.concat((tf.broadcast_to(sequence_order_padding, (Y_pred_chaos[-1].shape[0], Y_pred_chaos[-1].shape[1], 1)),
                                                      tf.reshape(Y_pred_chaos[-1], (Y_pred_chaos[-1].shape[0], Y_pred_chaos[-1].shape[1], 1))), axis=2))
            for j in range(1, len(layers) - 1):
                Y_pred_chaos.append(0)
                skip_X.append(0)
                Y_pred_chaos[-1], skip_X[-1] = layers[j].forward(Y_pred_chaos_with_order[-1], new_X=1, middle=middle, skip_X=skip_X_orig, training=0)
                Y_pred_chaos_with_order.append(tf.concat((tf.broadcast_to(sequence_order_padding, (Y_pred_chaos[-1].shape[0], Y_pred_chaos[-1].shape[1], 1)),
                                                      tf.reshape(Y_pred_chaos[-1], (Y_pred_chaos[-1].shape[0], Y_pred_chaos[-1].shape[1], 1))), axis=2))
            Y_pred_final, _ = layers[-1].forward(Y_pred_chaos_with_order[-1], new_X=1, output=1, skip_X=skip_X_orig, training=0)
            val_loss = dense_output.compute_loss(Y_pred_final, Y)
            val_accuracy = dense_output.compute_accuracy(Y_pred_final, Y)
            print(f'    epoch: {epoch}, validation loss: {val_loss}, validation accuracy: {val_accuracy}')
        if epoch % 100 == 0 or (loss < threshold_loss or accuracy > 0.98) and 0 or val_accuracy > 0.98 or 1:
            for i in range(len(layers)):
                layers[i].save_model(filepath)
            dense_output.save_model(filepath)
        loss_graph_x.append(epoch)
        loss_graph_y.append(loss)
        accuracy_graph_x.append(epoch)
        accuracy_graph_y.append(accuracy)
        return (loss, accuracy, val_accuracy)
    
    def print_memory_usage():
        process = psutil.Process()
        print(f"Memory Usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    
    
    for epoch in range(1,30001):
        logdir = f'./logs/memtrace_{epoch}_'
        #print(len(tf.compat.v1.get_default_graph().get_operations()))
        print_memory_usage()
        #tf.profiler.experimental.start(logdir)
        loss, accuracy, val_accuracy = training_loop(epoch)
        np.empty((0,))
        gc.collect()
        #tf.keras.backend.clear_session()
        #tf.profiler.experimental.stop()
        if (loss < threshold_loss or accuracy > 0.999) and 0 or val_accuracy > 0.98:
            break
    figure, axis = plt.subplots(1, 2)
    axis[0].plot(loss_graph_x, loss_graph_y)
    axis[0].set_title('loss')
    axis[1].plot(accuracy_graph_x, accuracy_graph_y)
    axis[1].set_title('accuracy')
    plt.show()

def train_simple_imdb():
    global mxlen
    batches_selected = 5
    validation_size = 500
    validation_x = np.load('/home/user/Desktop/datasets/imdb_50k_x_5.npy')[:validation_size]
    validation_y = np.load('/home/user/Desktop/datasets/imdb_50k_y_5.npy')[:validation_size]
    learning_rate = 0.001
    hidden_size = 100
    batch_size = 300
    n = 4
    num_y = 200
    dropout = 0.5
    layers = []
    layers.append(FeedForwardLayer(mxlen, hidden_size, 0, learning_rate=learning_rate))
    for i in range(n - 2):
        layers.append(FeedForwardLayer(hidden_size, hidden_size, i + 1, learning_rate=learning_rate))
    layers.append(FeedForwardLayer(hidden_size, mxlen, n, learning_rate=learning_rate))
    print(f'''
    learning_rate = {learning_rate}
    hidden_size = {hidden_size}
    batch_size = {batch_size}
    n = {n}
    num_y = {num_y}
    dropout = {dropout}
    training_samples = {batches_selected * batch_size}
    validation_size = {validation_size}
    ''')

    def generate_Y(num_y, shape, Y_optimal):
        distribution = tf.random.normal((num_y, shape[0], shape[1]), mean=0.0, stddev=0.01)
        Y = distribution + Y_optimal
        return Y
    
    for i in range(batches_selected):
        X = np.load(f'/home/user/Desktop/batches/training_batches_x_{i}.npy', mmap_mode='r')
        Y = np.load(f'/home/user/Desktop/batches/training_batches_y_{i}.npy', mmap_mode='r')
        X = tf.convert_to_tensor(X, dtype='float32')
        X = X[:, :, 1]
        Y = tf.convert_to_tensor(Y, dtype='float32')
        Y = Y[:, 1:]
        shape = X.shape
        y_true_classes = tf.cast(tf.broadcast_to(Y, shape), dtype='int32')
        mask_0 = tf.cast(tf.equal(tf.round(y_true_classes), 0), dtype='float32')
        mask_1 = tf.cast(tf.equal(tf.round(y_true_classes), 1), dtype='float32')
        distribution_0 = tf.random.normal((shape[0], shape[1]), mean=-0.1, stddev=0.01)
        distribution_1 = tf.random.normal((shape[0], shape[1]), mean=0.1, stddev=0.01)
        np.save(f'/home/user/Desktop/batches/ys_optimal_{i}.npy', mask_0 * distribution_0 + mask_1 * distribution_1)

    def training_epoch(epoch):
        start_time = time.time()
        threshold_loss = 0.05
        loss = 0
        accuracy = 0
        
        @tf.autograph.experimental.do_not_convert
        def batch_forward(X, Y):
            Y_pred = [X]
            for i in range(n):
                Y_pred.append(0)
                Y_pred[-1] = layers[i].forward(Y_pred[-2], output=int(i == n - 1))
            return layers[-1].compute_loss(Y_pred[-1], Y, mean=0)
        
        for i in range(batches_selected):
            X = np.load(f'/home/user/Desktop/batches/training_batches_x_{i}.npy', mmap_mode='r')
            Y = np.load(f'/home/user/Desktop/batches/training_batches_y_{i}.npy', mmap_mode='r')
            Y_optimal = np.load(f'/home/user/Desktop/batches/ys_optimal_{i}.npy', mmap_mode='r')
            X = tf.convert_to_tensor(X, dtype='float32')
            X = X[:, :, 1]
            Y = tf.convert_to_tensor(Y, dtype='float32')
            Y = Y[:, 1:]
            Y_classes = Y
            Y_samples = generate_Y(num_y, X.shape, Y_optimal)
            Y_samples_orig = copy.deepcopy(Y_samples)
            Y_samples_shape = Y_samples.shape
            Y_samples = tf.reshape(Y_samples, (Y_samples.shape[0] * Y_samples.shape[1], Y_samples.shape[2]))
            X_tiled = tf.tile(tf.expand_dims(X, 1), [1, num_y, 1])
            X_reshaped = tf.reshape(X_tiled, (X_tiled.shape[0] * X_tiled.shape[1], X_tiled.shape[2]))
            loss_vals = batch_forward(X_reshaped, Y_samples)
            loss_vals = tf.reshape(loss_vals, (Y_samples_shape[0], Y_samples_shape[1]))
            loss_vals = tf.reduce_mean(loss_vals, axis=1)
            argmin_index = tf.math.argmin(loss_vals)
            Y = Y_samples_orig[argmin_index]
            np.save(f'/home/user/Desktop/batches/ys_optimal_{i}.npy', Y)
            Y_pred = [X]
            for j in range(n):
                Y_pred.append(0)
                Y_pred[-1]= layers[j].forward(Y_pred[-2], output=int(j == n - 1))
            loss = (loss * i + layers[0].compute_loss(Y_pred[-1], Y)) / (i + 1)
            accuracy = (accuracy * i + layers[0].compute_accuracy(Y_pred[-1], Y_classes)) / (i + 1)
            dA2 = Y_pred[-1] - Y
            for j in range(n - 1, -1, -1):
                dA2 = layers[j].backward(Y_pred[j], Y_pred[j + 1], dA2, output=int(j == n - 1))
        if not (loss < threshold_loss or accuracy > 0.98):
            print(f'epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Time: {time.time() - start_time}')
        if (epoch % 100 < 20 or epoch % 100 == 0 or loss < threshold_loss or accuracy > 0.98 or 1) and len(validation_x) > 0:
            X = validation_x
            Y = validation_y
            X = tf.convert_to_tensor(X, dtype='float32')
            X = X[:, :, 1]
            Y = tf.convert_to_tensor(Y, dtype='float32')
            Y = Y[:, 1:]
            Y_classes = Y
            Y_pred = [X]
            for i in range(n):
                Y_pred.append(0)
                Y_pred[-1]= layers[i].forward(Y_pred[-2], output=int(i == n - 1))
            #val_loss = layers[0].compute_loss(Y_pred[-1], Y)
            val_accuracy = layers[0].compute_accuracy(Y_pred[-1], Y_classes)
            print(f'    epoch: {epoch}, validation loss: {1}, validation accuracy: {val_accuracy}')
    for i in range(1, 10001):
        training_epoch(i)



if 1:
    pass
    #train_representation_chaos_imdb()
    #train_chaos_words_imdb()
    train_simple_imdb()