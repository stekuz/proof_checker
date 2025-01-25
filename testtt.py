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

class BatchNormalization:
    def __init__(self, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = None 
        self.beta = None  
        self.running_mean = None
        self.running_var = None

    def initialize(self, input_shape):
        self.gamma = tf.ones(shape=(input_shape[-1],))
        self.beta = tf.zeros(shape=(input_shape[-1],))
        self.running_mean = tf.zeros(shape=(input_shape[-1],))
        self.running_var = tf.ones(shape=(input_shape[-1],))

    def forward(self, x, training=True):
        if self.gamma is None or self.beta is None:
            self.initialize(x.shape)

        if training:
            self.batch_mean = tf.math.reduce_mean(x, axis=0)
            self.batch_var = tf.math.reduce_variance(x, axis=0)

            self.running_mean = (self.momentum * self.running_mean + 
                                 (1 - self.momentum) * self.batch_mean)
            self.running_var = (self.momentum * self.running_var + 
                                (1 - self.momentum) * self.batch_var)
        else:
            self.batch_mean = self.running_mean
            self.batch_var = self.running_var

        self.x_normalized = (x - self.batch_mean) / tf.math.sqrt(self.batch_var + self.epsilon)

        out = self.gamma * self.x_normalized + self.beta
        return out
    
    def backward(self, dout):
        N, D = dout.shape
        
        dbeta = tf.math.reduce_sum(dout, axis=0)
        dgamma = tf.math.reduce_sum(dout * self.x_normalized, axis=0)

        dx_normalized = dout * self.gamma
        
        dvar = tf.math.reduce_sum(dx_normalized * (self.x_normalized), axis=0) * (-0.5) * ((self.batch_var + self.epsilon) ** (-1.5))
        dmean = tf.math.reduce_sum(dx_normalized, axis=0) * (-1 / tf.math.sqrt(self.batch_var + self.epsilon)) + dvar * tf.reduce_mean(-2 * (self.x_normalized), axis=0)

        dx = dx_normalized / tf.math.sqrt(self.batch_var + self.epsilon) + (dvar * 2 / N * (self.x_normalized)) + (dmean / N)

        return dx, dgamma, dbeta

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
        self.batch_norm1 = BatchNormalization()
        self.batch_norm2 = BatchNormalization()
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
        #self.A1 = tf.nn.relu(self.batch_norm1.forward(self.Z1))
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
        return tf.reduce_mean(abs(Y_true - Y_pred))

    def compute_accuracy(self, Y_pred, Y_true):
        Y_pred_classes = tf.round(Y_pred)
        return tf.reduce_mean(tf.cast(tf.equal(Y_true, Y_pred_classes), dtype='float32'))

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
        #dZ1_bn, _, _ = self.batch_norm1.backward(dZ1)
        dW1 = tf.linalg.matmul(tf.transpose(X), dZ1) / m 
        db1 = tf.math.reduce_sum(dZ1, axis=0, keepdims=True) / m 

        mnval = -1
        mxval = 1
        dW1 = tf.clip_by_value(dW1, mnval, mxval)
        db1 = tf.clip_by_value(db1, mnval, mxval)
        dW2 = tf.clip_by_value(dW2, mnval, mxval)
        db2 = tf.clip_by_value(db2, mnval, mxval)

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

class ChaosTransformLayer:
    def __init__(self, input_shape, hidden_size, output_size, index, learning_rate=0.001, total_input=10, batches_len=1):
        self.index = index
        self.total_input = total_input
        self.dense_input = []
        for i in range(self.total_input):
            self.dense_input.append(FeedForwardLayer(input_size=input_shape[1], hidden_size=hidden_size, output_size=hidden_size, index='ssoriginp' + str(index) + str(i), learning_rate=learning_rate))
        self.dense_combinator = FeedForwardLayer(input_size=self.total_input * hidden_size, hidden_size=self.total_input * hidden_size, output_size=output_size, index='sscomb' + str(index), learning_rate=learning_rate)
        #self.chaos_0 = np.random.rand(2, 2)
        self.chaos_10_9 = np.array([[10, 9], [1, 1]], dtype='float32')
        #self.chaos_10_9 = np.array([[1, 0], [0, 1]], dtype='float32')
        self.chaos_2_1 = np.array([[2, 1], [1, 1]], dtype='float32')
        self.mat_1 = np.array([[1, 0], [0, 1]], dtype='float32')
        self.chaos_1 = []
        self.mat_2 = np.array([[1, 0], [0, 1]], dtype='float32')
        self.chaos_2 = []
        for i in range(self.total_input):
            self.chaos_1.append(tf.convert_to_tensor(copy.deepcopy(self.mat_1), dtype='float32'))
            #self.chaos.append(tf.random.uniform((2, 2)))
            self.mat_1 = np.dot(self.mat_1, self.chaos_10_9)
            self.chaos_2.append(tf.convert_to_tensor(copy.deepcopy(self.mat_2), dtype='float32'))
            self.mat_2 = np.dot(self.mat_2, self.chaos_2_1)
        print(self.mat_1, self.mat_2)
        self.batches_len = batches_len
        self.X = [None] * self.batches_len

    def forward(self, X_original, double_matrix=0, new_X=0, batches_i=0):
        if isinstance(X_original, np.ndarray) == False:
            X_original = X_original.numpy()
        if self.X[batches_i] == None:
            X_1 = []
            X_2 = []
            X = []
            for i in range(self.total_input):
                X_1.append(copy.deepcopy(X_original))
                for j in range(len(X_1[i])):
                    X_1[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_1[i], tf.transpose(X_1[i][j]))) % 1
                X_1[i] = tf.squeeze(X_1[i][:, :, 1:])
                if double_matrix:
                    X_2.append(copy.deepcopy(X_original))
                    for j in range(len(X_2[i])):
                        X_2[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_2[i], tf.transpose(X_2[i][j]))) % 1
                    X_2[i] = tf.squeeze(X_2[i][:, :, 1:])
                    X.append((X_1[i] + X_2[i]) % 1)
                else:
                    X.append(X_1[i])
            self.X[batches_i] = X
        X = self.X[batches_i]
        if new_X:
            X_1 = []
            X_2 = []
            X = []
            for i in range(self.total_input):
                X_1.append(copy.deepcopy(X_original))
                for j in range(len(X_1[i])):
                    X_1[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_1[i], tf.transpose(X_1[i][j]))) % 1
                X_1[i] = tf.squeeze(X_1[i][:, :, 1:])
                if double_matrix:
                    X_2.append(copy.deepcopy(X_original))
                    for j in range(len(X_2[i])):
                        X_2[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_2[i], tf.transpose(X_2[i][j]))) % 1
                    X_2[i] = tf.squeeze(X_2[i][:, :, 1:])
                    X.append((X_1[i] + X_2[i]) % 1)
                else:
                    X.append(X_1[i])
        Y_pred = []
        self.A2 = []
        for i in range(self.total_input):
            self.A2.append(self.dense_input[i].forward(X[i]))
            Y_pred.append(self.A2[i])
        Y_pred_to_combinator = Y_pred[0]
        for i in range(1, self.total_input):
            Y_pred_to_combinator = tf.concat((Y_pred_to_combinator, Y_pred[i]), axis=1)
        Y_pred_final = self.dense_combinator.forward(Y_pred_to_combinator)
        return Y_pred_final
    
    def backward(self, X_original, A2, dA2, double_matrix=0, new_X=0, batches_i=0):
        if isinstance(X_original, np.ndarray) == False:
            X_original = X_original.numpy()
        if self.X[batches_i] == None:
            X_1 = []
            X_2 = []
            X = []
            for i in range(self.total_input):
                X_1.append(copy.deepcopy(X_original))
                for j in range(len(X_1[i])):
                    X_1[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_1[i], tf.transpose(X_1[i][j]))) % 1
                X_1[i] = tf.squeeze(X_1[i][:, :, 1:])
                if double_matrix:
                    X_2.append(copy.deepcopy(X_original))
                    for j in range(len(X_2[i])):
                        X_2[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_2[i], tf.transpose(X_2[i][j]))) % 1
                    X_2[i] = tf.squeeze(X_2[i][:, :, 1:])
                    X.append((X_1[i] + X_2[i]) % 1)
                else:
                    X.append(X_1[i])
            self.X[batches_i] = X
        X = self.X[batches_i]
        if new_X:
            X_1 = []
            X_2 = []
            X = []
            for i in range(self.total_input):
                X_1.append(copy.deepcopy(X_original))
                for j in range(len(X_1[i])):
                    X_1[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_1[i], tf.transpose(X_1[i][j]))) % 1
                X_1[i] = tf.squeeze(X_1[i][:, :, 1:])
                if double_matrix:
                    X_2.append(copy.deepcopy(X_original))
                    for j in range(len(X_2[i])):
                        X_2[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_2[i], tf.transpose(X_2[i][j]))) % 1
                    X_2[i] = tf.squeeze(X_2[i][:, :, 1:])
                    X.append((X_1[i] + X_2[i]) % 1)
                else:
                    X.append(X_1[i])
        self.A1 = self.dense_input[0].A1
        for i in range(1, self.total_input):
            self.A1 = tf.concat((self.A1, self.dense_input[i].A1), axis=1)
        dA2_final = self.dense_combinator.backward(self.A1, A2, dA2)

        dA2 = self.dense_input[0].backward(X[0], self.A2[0], dA2_final[:, i * self.dense_input[0].W1.shape[1]:(i + 1) * self.dense_input[0].W1.shape[1]])
        for i in range(1, self.total_input):
            dA2 += self.dense_input[i].backward(X[i], self.A2[i], dA2_final[:, i * self.dense_input[i].W1.shape[1]:(i + 1) * self.dense_input[i].W1.shape[1]])
        dA2 = tf.convert_to_tensor(dA2, dtype='float32')
        return dA2
    
    def save_model(self, filepath):
        for i in range(self.total_input):
            self.dense_input[i].save_model(filepath)
        self.dense_combinator.save_model(filepath)
    
    def load_model(self, filepath):
        for i in range(self.total_input):
            self.dense_input[i].load_model(filepath)
        self.dense_combinator.load_model(filepath)

all_tokens = {}
all_tokens_list = []

def prepare_data_imdb():
    data_raw = []
    with open('/home/user/Desktop/datasets/IMDB Dataset.csv') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            if i and len(row) == 2:
                data_raw.append(row)
            i += 1
            if i == 3000:
                break
    mxlen = 0
    for row in data_raw:
        mxlen = max(mxlen, len(row[0]))
        for token in row[0]:
            all_tokens[token] = 1
    with open('./all_tokens.txt', 'w') as f:
        for token in all_tokens:
            f.write(token + '\n')
    with open('./mxlen.txt', 'w') as f:
        f.write(str(mxlen))
    mxlen = int(open('./mxlen.txt', 'r').readlines()[0])
    all_tokens_list = open('./all_tokens.txt', 'r').readlines()
    all_tokens_list = [token[:-1] for token in all_tokens_list]
    all_tokens_list = sorted(all_tokens_list)
    print(all_tokens_list)
    for i in range(len(all_tokens_list)):
        all_tokens[all_tokens_list[i]] = i
    X = []
    Y = []
    for i in range(len(data_raw)):
        X.append([])
        for j in range(len(data_raw[i][0])):
            X[i].append([1, all_tokens[data_raw[i][0][j]] / (len(all_tokens) + 1)])
        nowlen = len(X[i])
        for j in range(mxlen - nowlen):
            X[i].append([1, len(all_tokens) / (len(all_tokens) + 1)])
        for j in range(mxlen):
            X[i][j][0] = j / mxlen
        if data_raw[i][1] == 'positive':
            Y.append([1])
        else:
            Y.append([0])
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')
    np.save('/home/user/Desktop/datasets/imdb_train_x.npy', X)
    np.save('/home/user/Desktop/datasets/imdb_train_y.npy', Y)

#prepare_data_imdb()

mxlen = int(open('./mxlen.txt', 'r').readlines()[0])
all_tokens_list = open('./all_tokens.txt', 'r').readlines()
all_tokens_list = [token[:-1] for token in all_tokens_list]
all_tokens_list = sorted(all_tokens_list)
print(all_tokens_list)
for i in range(len(all_tokens_list)):
    all_tokens[all_tokens_list[i]] = i
        
def train_model_chaos_batches():
    samples_selected = 500
    validation_split = 100
    X = np.load('/home/user/Desktop/datasets/imdb_train_x.npy')[:samples_selected]
    Y = np.load('/home/user/Desktop/datasets/imdb_train_y.npy')[:samples_selected]
    validation_x = np.load('/home/user/Desktop/datasets/imdb_train_x.npy')[samples_selected:samples_selected + validation_split]
    validation_y = np.load('/home/user/Desktop/datasets/imdb_train_y.npy')[samples_selected:samples_selected + validation_split]

    learning_rate = 0.001
    hidden_size = 3
    total_input = 13
    batch_size = len(X)
    X_shape = X.shape
    X = X.tolist()
    batches_x = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
    chaostranform1 = ChaosTransformLayer(X_shape, hidden_size, hidden_size, 1, learning_rate=learning_rate / 100, total_input=total_input, batches_len=len(batches_x))
    lastlen = len(batches_x[-1])
    for i in range(batch_size - lastlen):
        batches_x[-1].append(batches_x[-1][-1])
    batches_x = np.array(batches_x, dtype='float32')
    Y = Y.tolist()
    batches_y = [Y[i:i + batch_size] for i in range(0, len(X), batch_size)]
    lastlen = len(batches_y[-1])
    for i in range(batch_size - lastlen):
        batches_y[-1].append(batches_y[-1][-1])
    batches_y = np.array(batches_y, dtype='float32')
    #chaostranform1 = ChaosTransformLayer(X.shape, hidden_size, hidden_size, 1, learning_rate=learning_rate / 100, total_input=total_input)
    #hidden_size *= total_input
    dense_middle1 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayer(hidden_size, hidden_size, 1, 4, learning_rate=learning_rate, dropout=0)
    filepath = './custom_models_2/test_model'
    loss_graph_x = []
    loss_graph_y = []
    accuracy_graph_x = []
    accuracy_graph_y = []
    def trainint_loop(epoch):
        start_time = time.time()
        loss = 0
        accuracy = 0
        for i in range(len(batches_x)):
            Y_pred_chaos = chaostranform1.forward(batches_x[i], double_matrix=1, batches_i=i)
            Y_pred_middle1 = dense_middle1.forward(Y_pred_chaos)
            Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
            Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
            Y_pred_final = dense_output.forward(Y_pred_middle3, output=1)
            loss = (loss * i + dense_output.compute_loss(Y_pred_final, batches_y[i])) / (i + 1)
            accuracy = (accuracy * i + dense_output.compute_accuracy(Y_pred_final, batches_y[i])) / (i + 1)
            dA2 = dense_output.backward(Y_pred_middle3, Y_pred_final, Y_pred_final - batches_y[i], output=1)
            dA2 = dense_middle3.backward(Y_pred_middle2, Y_pred_middle3, dA2)
            dA2 = dense_middle2.backward(Y_pred_middle1, Y_pred_middle2, dA2)
            dA2 = dense_middle1.backward(Y_pred_chaos, Y_pred_middle1, dA2)
            dA2 = chaostranform1.backward(batches_x[i], Y_pred_chaos, dA2, double_matrix=1, batches_i=i)
        validation_loss = 1
        print(f'Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Time: {time.time() - start_time}')
        if 1 and (epoch%100==0 or (loss < 0.01 or accuracy > 0.9) or epoch%100 < 1):
            Y_pred_chaos = chaostranform1.forward(validation_x, double_matrix=1, new_X=1)
            Y_pred_middle1 = dense_middle1.forward(Y_pred_chaos)
            Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
            Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
            Y_pred_final = dense_output.forward(Y_pred_middle3, output=1)
            validation_loss = dense_output.compute_loss(Y_pred_final, validation_y)
            validation_accuracy = dense_output.compute_accuracy(Y_pred_final, validation_y)
            print(f'validation loss: {validation_loss}, validation accuracy: {validation_accuracy}')
        if epoch%100==0 or (loss < 0.01 or accuracy > 0.9):
            chaostranform1.save_model(filepath)
            dense_middle1.save_model(filepath)
            dense_middle2.save_model(filepath)
            dense_middle3.save_model(filepath)
            dense_output.save_model(filepath)
        loss_graph_x.append(epoch)
        loss_graph_y.append(loss)
        accuracy_graph_x.append(epoch)
        accuracy_graph_y.append(accuracy)
        return (loss, accuracy, validation_loss)

    for epoch in range(1,10001):
        loss, accuracy, validation_loss = trainint_loop(epoch)
        if loss < 0.01 or accuracy > 0.9 or validation_loss < 0.2:
            break
    figure, axis = plt.subplots(1, 2)
    axis[0].plot(loss_graph_x, loss_graph_y)
    axis[0].set_title('loss')
    axis[1].plot(accuracy_graph_x, accuracy_graph_y)
    axis[1].set_title('accuracy')
    plt.show()

def train_model_simple():
    samples_selected = 500
    validation_split = 100
    X = np.load('/home/user/Desktop/datasets/imdb_train_x.npy')[:samples_selected]
    Y = np.load('/home/user/Desktop/datasets/imdb_train_y.npy')[:samples_selected]
    validation_x = np.load('/home/user/Desktop/datasets/imdb_train_x.npy')[samples_selected:samples_selected + validation_split]
    validation_y = np.load('/home/user/Desktop/datasets/imdb_train_y.npy')[samples_selected:samples_selected + validation_split]
    X = X.tolist()
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = X[i][j][1]
    X = np.array(X, dtype='float32')
    validation_x = validation_x.tolist()
    for i in range(len(validation_x)):
        for j in range(len(validation_x[i])):
            validation_x[i][j] = validation_x[i][j][1]
    validation_x = np.array(validation_x, dtype='float32')
    learning_rate = 0.0001
    hidden_size = 3
    total_input = 13
    chaostranform1 = FeedForwardLayer(X.shape[1], hidden_size * total_input, hidden_size * total_input, 0, learning_rate=learning_rate)
    #hidden_size *= total_input
    dense_middle1 = FeedForwardLayer(hidden_size * total_input, hidden_size * total_input, hidden_size, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayer(hidden_size, hidden_size, mxlen, 4, learning_rate=learning_rate)
    filepath = './custom_models/test_model'
    loss_graph_x = []
    loss_graph_y = []
    accuracy_graph_x = []
    accuracy_graph_y = []
    avg_loss = [0 for i in range(100)]
    avg_accuracy = [0 for i in range(100)]
    threshold_loss = 0.03
    def trainint_loop(epoch):
        start_time = time.time()
        Y_pred_chaos = chaostranform1.forward(X)
        Y_pred_middle1 = dense_middle1.forward(Y_pred_chaos)
        Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
        Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
        Y_pred_final = dense_output.forward(Y_pred_middle3, output=1)
        loss = dense_output.compute_loss(Y_pred_final, Y)
        avg_loss[epoch % 100] = loss
        accuracy = dense_output.compute_accuracy(Y_pred_final, Y)
        avg_accuracy[epoch % 100] = accuracy
        dA2 = dense_output.backward(Y_pred_middle3, Y_pred_final, Y_pred_final - Y, output=1)
        dA2 = dense_middle3.backward(Y_pred_middle2, Y_pred_middle3, dA2)
        dA2 = dense_middle2.backward(Y_pred_middle1, Y_pred_middle2, dA2)
        dA2 = dense_middle1.backward(Y_pred_chaos, Y_pred_middle1, dA2)
        dA2 = chaostranform1.backward(X, Y_pred_chaos, dA2)
        val_accuracy = 0
        print(f'epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Time: {time.time() - start_time}')
        print(f'average loss (per 100): {sum(avg_loss) / 100}, average accuracy: {sum(avg_accuracy) / 100}')
        if (epoch % 500 < 0 or epoch % 100 == 0 or loss < threshold_loss or accuracy > 0.98) and len(validation_x) > 0:
            Y_pred_chaos = chaostranform1.forward(validation_x)
            Y_pred_middle1 = dense_middle1.forward(Y_pred_chaos)
            Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
            Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
            Y_pred_final = dense_output.forward(Y_pred_middle3, output=1)
            val_loss = dense_output.compute_loss(Y_pred_final, validation_y)    
            val_accuracy = dense_output.compute_accuracy(Y_pred_final, validation_y)
            print(f'validation loss: {val_loss}, validation accuracy: {val_accuracy}')
        if epoch % 100 == 0 or loss < threshold_loss or accuracy > 0.98:
            chaostranform1.save_model(filepath)
            dense_middle1.save_model(filepath)
            dense_middle2.save_model(filepath)
            dense_middle3.save_model(filepath)
            dense_output.save_model(filepath)
        loss_graph_x.append(epoch)
        loss_graph_y.append(loss)
        accuracy_graph_x.append(epoch)
        accuracy_graph_y.append(accuracy)
        return (loss, accuracy, val_accuracy)

    for epoch in range(1,2001):
        loss, accuracy, val_accuracy = trainint_loop(epoch)
        if loss < threshold_loss or accuracy > 0.98:
            break
    figure, axis = plt.subplots(1, 2)
    axis[0].plot(loss_graph_x, loss_graph_y)
    axis[0].set_title('loss')
    axis[1].plot(accuracy_graph_x, accuracy_graph_y)
    axis[1].set_title('accuracy')
    plt.show()

def train_model_keras():
    samples_selected = 500 + 100
    validation_split = 100
    X = np.load('/home/user/Desktop/datasets/imdb_train_x.npy')[:samples_selected]
    Y = np.load('/home/user/Desktop/datasets/imdb_train_y.npy')[:samples_selected]
    validation_x = np.load('/home/user/Desktop/datasets/imdb_train_x.npy')[samples_selected:samples_selected + validation_split]
    validation_y = np.load('/home/user/Desktop/datasets/imdb_train_y.npy')[samples_selected:samples_selected + validation_split]
    X = X.tolist()
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = X[i][j][1]
    X = np.array(X, dtype='float32')
    validation_x = validation_x.tolist()
    for i in range(len(validation_x)):
        for j in range(len(validation_x[i])):
            validation_x[i][j] = validation_x[i][j][1]
    validation_x = np.array(validation_x, dtype='float32')
    Y = Y.tolist()
    for i in range(len(Y)):
        Y[i] = [1 - Y[i][0], Y[i][0]]
    Y = np.array(Y, dtype='float32')
    validation_y = validation_y.tolist()
    for i in range(len(validation_y)):
        validation_y[i] = [1 - validation_y[i][0], validation_y[i][0]]
    validation_y = np.array(validation_y, dtype='float32')
    input_shape = (X.shape[1], 1, )
    inputs = keras.Input(shape=input_shape)
    conv1 = layers.Conv1D(3, kernel_size=3, activation='relu')(inputs)
    flatten1 = layers.Flatten()(inputs)
    dense2 = layers.Dense(3, activation='relu')(flatten1)
    dense3 = layers.Dense(3, activation='relu')(dense2)
    flatten2 = layers.Flatten()(dense3)
    dropout = layers.Dropout(0)(flatten2)
    outputs = layers.Dense(2, activation='softmax')(dropout)
    #model = keras.models.load_model('./models_trained/checker_keras.ckpt.keras')
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    batch_size = 1
    epochs = 300
    optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8)
    #optimizer = keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss='mae', metrics=['accuracy'])
    model.summary()
    cp_callback = keras.callbacks.ModelCheckpoint(filepath='./models_trained/checker_keras_2.ckpt.keras')
    model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=1/6, callbacks=[cp_callback])
if 1:
    #print(mxlen)
    train_model_chaos_batches()
    #train_model_simple()
    #train_model_keras()

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

special_chars = ['„', '”', '"', '\'', '.', ',', '/', '\\', ' ', '   ', '*', ';', ':', '`', '[', ']', '{', '}', '!', '?', '\n', '$']

common_words=[
'water','away','good','want','over','how','did','man',
'going','where','would','or','took','school','think',
'home','who','didn','ran','know','bear','can','again',
'cat','long','things','new','after','wanted','eat',
'everyone','our','two','has','yes','play','take',
'thought','dog','well','find','more','i','ll',
'round','tree','magic','shouted','us','other',
'food','fox','through','way','been','stop',
'must','red','door','right','sea','these',
'began','boy','animals','never','next',
'first','work','lots','need','that',
'baby','fish','gave','mouse',
'something','bed','may','still','found','live',
'say','soon','night','narrator','small','car',
'couldn','t','three','head','king','town',
'I','ve','around','every','garden','fast',
'only','many','laughed','let','much',
'suddenly','told','another','great','why',
'cried','keep','room','last','jumped',
'because','even','am','before',
'gran','clothes',
'tell','key','fun','place','mother',
'sat','boat','window','sleep',
'feet','morning','queen',
'each','book','its',
'green','different',
'let','girl',
'which','inside',
'run','any',
'under','hat',
'snow','air',
'trees','bad',
'tea','top',
'eyes','fell',
'friends','box',
'dark','grandad',
'there','looking',
'end','than',
'best','better','hot','sun',
'across','gone','hard',
'floppy','really','wind',
'wish','eggs',
'once','please',
'thing','stopped',
'ever','miss',
'most','cold',
'park','lived',
'birds','duck',
'horse','rabbit',
'white','coming',
'he','s',
'river','liked',
'giant','looks',
'use','along',
'plants','dragon',
'pulled','we',
're','fly',
'grow','make'
]

#for sentences from 200 words:
#chaotic loss = 40 at 5500 epoch
#non chaotic loss didn't cross 40 over 10000 epochs

#for sentences from 70 words:
#chaotic loss >100 but accuracy >0.94 at 4600 epoch, also it can translate unseen examples
#non chaotic loss >200 and accuracy <80 over 10000 epochs

#2500 epochs for 0.98 accuracy for one matrix with random multiples
#3100 epochs for 0.98 accuracy for two matrices with random multiples
#3300 epochs for 0.98 accuracy for one matrix without random multiples

#chaotic transformer can achieve 0.99 of accuracy on the training set
#chaotic transformer can achieve 0.33 of true pseudo accuracy (the absolute deviation of the first translated word from the true one is less than 10)
#only dense layers can achieve 0.5 of true pseudo accuracy but can not achieve 0.9 of accuracy on the training set
#chaotic transformer also can achieve 0.5 of true pseudo accuracy but can achieve 0.99 of accuracy on the training set

all_tokens = {}

data_train = []
mxlen = 0

data_raw = sentence_pairs
#print(data_raw)

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
    with open('./mxlen.txt', 'w') as f:
        f.write(str(mxlen))
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
    for i in range(len(X)):
        nowlen = len(X[i])
        for j in range(mxlen - nowlen):
            X[i].append([1, len(all_tokens) / (len(all_tokens) + 1)])
        for j in range(mxlen):
            X[i][j][0] = j / mxlen
        nowlen = len(Y[i])
        for j in range(mxlen - nowlen):
            Y[i].append(len(all_tokens))
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')

    np.save('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy', X)
    np.save('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_y.npy', Y)

#data_preparation_common_seq2seq()

def data_preparation_common():
    global mxlen
    data_raw = sentence_pairs

    for i in range(len(data_raw)):
        en = []
        de = []
        now = ''
        for j in data_raw[i][0]:
            if j in special_chars:
                if now != '':
                    en.append(now)
                en.append(j)
                now = ''
            else:
                now += j.lower()
        if now != '':
            en.append(now)
            now = ''
        for j in data_raw[i][1]:
            if j in special_chars:
                if now != '':
                    de.append(now)
                de.append(j)
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
        mxlen = max([mxlen, len(de) + 10, len(en) + 10])
    with open('./all_tokens.txt', 'w') as f:
        for i in all_tokens:
            f.write(i + '\n')

    all_tokens_list = open('./all_tokens.txt', 'r').readlines()
    all_tokens_list = [token[:-1] for token in all_tokens_list]
    all_tokens_list = sorted(all_tokens_list)
    for i in range(len(all_tokens_list)):
        all_tokens[all_tokens_list[i]] = i
    
    for i in range(len(data_train)):
        for k in range(2):
            for j in range(len(data_train[i][k])):
                data_train[i][k][j] = [1, all_tokens[data_train[i][k][j]] / (len(all_tokens) + 1)]

    
    X = []
    Y = []
    for i in range(len(data_train)):
        to_add = []
        for j in range(len(data_train[i][0])):
            X.append(data_train[i][1] + to_add)
            Y.append(data_train[i][0][j][1])
            to_add.append([1, data_train[i][0][j][1]])
    mxlen *= 2
    for i in range(len(X)):
        nowlen = len(X[i])
        for j in range(mxlen - nowlen):
            X[i].append([1, len(all_tokens) / (len(all_tokens) + 1)])
        for j in range(mxlen):
            X[i][j][0] = j / mxlen
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')

    np.save('/home/user/Desktop/datasets/translation_en_de_train_common_x.npy', X)
    np.save('/home/user/Desktop/datasets/translation_en_de_train_common_y.npy', Y)

#data_preparation_common()

def data_preparation_wmt():
    global mxlen
    data_raw = []

    with open('/home/user/Desktop/datasets/wmt14_translate_de-en_train.csv') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            if i and len(row) == 2:
                data_raw.append(row)
            i += 1
            if i == 10000:
                break
    for i in range(len(data_raw)):
        de = []
        en = []
        now = ''
        check = 1
        for j in data_raw[i][0]:
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
        for j in data_raw[i][1]:
            if j in special_chars:
                if now != '':
                    if not now in common_words:
                        check = 0 
                    en.append(now + j)
                #en.append(j)
                now = ''
            else:
                now += j.lower()
        if now != '':
            if not now in common_words:
                check = 0 
            en.append(now)
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
    with open('./mxlen.txt', 'w') as f:
        f.write(str(mxlen))

#data_preparation_wmt()
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

def preprocess_translation_wmt():
    global mxlen

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
    for i in range(len(X)):
        nowlen = len(X[i])
        for j in range(mxlen - nowlen):
            X[i].append([1, len(all_tokens) / (len(all_tokens) + 1)])
        for j in range(mxlen):
            X[i][j][0] = j / mxlen
        nowlen = len(Y[i])
        for j in range(mxlen - nowlen):
            Y[i].append(len(all_tokens))
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')

    np.save('/home/user/Desktop/datasets/translation_en_de_train_wmt_x.npy', X)
    np.save('/home/user/Desktop/datasets/translation_en_de_train_wmt_y.npy', Y)

#preprocess_translation_wmt()

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

tf.random.set_seed(42)

class BatchNormalization:
    def __init__(self, momentum=0.9, epsilon=1e-8):
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

class FeedForwardLayer:
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
        self.optimizer = AdamOptimizer(learning_rate=learning_rate / 10, epsilon=1e-8)
        self.batch_norm1 = BatchNormalization()
        self.batch_norm2 = BatchNormalization()
        self.batch_norm1.initialize((hidden_size, ))
        self.batch_norm2.initialize((output_size, ))
        #self.optimizer = SGD(learning_rate=learning_rate)
        #self.optimizer = AdaBelief(learning_rate=learning_rate, epsilon=1e-16)
        self.optimizer.initialize({
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
        })

    def forward(self, X, output=0, training=1):
        self.Z1 = tf.linalg.matmul(X, self.W1) + self.b1
        self.B1 = self.batch_norm1.forward(self.Z1, training=training)
        self.A1 = tf.nn.relu(self.B1)
        self.Z2 = tf.linalg.matmul(self.A1, self.W2) + self.b2
        #self.learning_rate += random.uniform(-self.learning_rate / 9, self.learning_rate / 10)
        #self.learning_rate = max(self.learning_rate, 0.00001)
        self.mx = tf.math.reduce_max(abs(self.Z2)) + 1e-10
        #epsilon = (1 / (self.optimizer.t + 1)) * 0
        #self.noise_matrix = tf.convert_to_tensor(np.diag(np.random.uniform(1 - epsilon, 1 + epsilon, self.Z2.shape[1])), dtype='float32')
        #self.noise_matrix = tf.random.uniform(self.Z2.shape, 1 - epsilon, 1 + epsilon)
        if output:
            #return dropout_mask * tf.nn.softmax(self.Z2)
            #return tf.linalg.matmul(self.Z2, self.noise_matrix)
            self.B2 = self.batch_norm2.forward(self.Z2, training=training)
            self.A2 = tf.nn.sigmoid(self.B2)
            return self.A2
        else:
            #return tf.linalg.matmul(tf.nn.relu(self.Z2), self.noise_matrix)
            self.B2 = self.batch_norm2.forward(self.Z2, training=training)
            self.A2 = tf.nn.relu(self.B2)
            return self.A2 

    def compute_loss(self, Y_pred, Y_true):
        #epsilon = 1e-8
        #loss = -tf.reduce_mean(tf.reduce_sum(Y_true * tf.math.log(Y_pred + epsilon), axis=-1))
        loss = tf.reduce_sum(tf.math.sqrt(abs(Y_true - Y_pred))) / (Y_pred.shape[0] * Y_pred.shape[1])
        l2_reg_cost = 0*(self.l2_lambda / 2) * (tf.reduce_sum(tf.square(self.W1)) + tf.reduce_sum(tf.square(self.W2)))
        return loss + l2_reg_cost

    def compute_accuracy(self, Y_pred, Y_true):
        #y_pred_classes = tf.argmax(Y_pred, axis=-1)
        #y_true_classes = tf.argmax(Y_true, axis=-1)
        y_pred_classes = tf.cast(tf.round(Y_pred * (len(all_tokens) + 1)), dtype='int32')
        y_true_classes = tf.cast(tf.round(Y_true * (len(all_tokens) + 1)), dtype='int32')
        threshold = 1
        correct_predictions = abs(y_true_classes - y_pred_classes) < threshold
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def backward(self, X, A2, dA2, output=0):
        m = dA2.shape[1] 
        if output:
            #dZ2 = dA2 * tf.nn.softmax(A2)
            #dZ2 = tf.linalg.matmul(dA2, self.noise_matrix) #* tf.cast(A2 > 0, dtype='float32')
            dZ2 = dA2 * tf.nn.sigmoid(A2) * (1 - tf.nn.sigmoid(A2))
            dB2 = self.batch_norm2.backward(self.Z2, dZ2)
        else:
            #dZ2 = tf.linalg.matmul(dA2, self.noise_matrix) * tf.cast(A2 > 0, dtype='float32')
            dZ2 = dA2 * tf.cast(A2 > 0, dtype='float32')
            dB2 = self.batch_norm2.backward(self.Z2, dZ2)
        dW2 = tf.linalg.matmul(tf.transpose(self.A1), dB2) / m + (self.l2_lambda * self.W2) / m
        db2 = tf.math.reduce_sum(dB2, axis=0, keepdims=True) / m 

        
        dA1 = tf.linalg.matmul(dZ2, tf.transpose(self.W2))
        m = dA1.shape[1]
        dZ1 = dA1 * tf.cast(self.A1 > 0, dtype='float32')
        dB1 = self.batch_norm1.backward(self.Z1, dZ1)
        dW1 = tf.linalg.matmul(tf.transpose(X), dB1) / m + (self.l2_lambda * self.W1) / m
        db1 = tf.math.reduce_sum(dB1, axis=0, keepdims=True) / m 
        dA0 = tf.linalg.matmul(dB1, tf.transpose(self.W1))

        mnval = -1
        mxval = 1
        dW1 = tf.clip_by_value(dW1, mnval, mxval)
        db1 = tf.clip_by_value(db1, mnval, mxval)
        dW2 = tf.clip_by_value(dW2, mnval, mxval)
        db2 = tf.clip_by_value(db2, mnval, mxval)
        #print(tf.math.reduce_max(abs(dW1)))

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

class ChaosTransformLayer:
    def __init__(self, input_shape, hidden_size, output_size, index, learning_rate=0.001, total_input=10, batches_len=1):
        self.index = index
        self.total_input = total_input
        self.dense_input = []
        for i in range(self.total_input):
            self.dense_input.append(FeedForwardLayer(input_size=input_shape[1], hidden_size=hidden_size, output_size=hidden_size, index='ssoriginp' + str(index) + str(i), learning_rate=learning_rate))
        self.dense_combinator = FeedForwardLayer(input_size=self.total_input * hidden_size, hidden_size=self.total_input * hidden_size, output_size=output_size, index='sscomb' + str(index), learning_rate=learning_rate)
        #self.chaos_0 = np.random.rand(2, 2)
        self.chaos_10_9 = np.array([[5, 2], [7, 3]], dtype='float32')
        #self.chaos_10_9 = np.array([[1, 0], [0, 1]], dtype='float32')
        self.chaos_2_1 = np.array([[2, 1], [1, 1]], dtype='float32')
        self.mat_1 = np.array([[1, 0], [0, 1]], dtype='float32')
        self.chaos_1 = []
        self.mat_2 = np.array([[1, 0], [0, 1]], dtype='float32')
        self.chaos_2 = []
        for i in range(self.total_input):
            self.chaos_1.append(tf.convert_to_tensor(copy.deepcopy(self.mat_1), dtype='float32'))
            #self.chaos.append(tf.random.uniform((2, 2)))
            self.mat_1 = np.dot(self.mat_1, self.chaos_10_9)
            self.chaos_2.append(tf.convert_to_tensor(copy.deepcopy(self.mat_2), dtype='float32'))
            self.mat_2 = np.dot(self.mat_2, self.chaos_2_1)
        print(self.mat_1, self.mat_2)
        self.batches_len = batches_len
        self.X = [None] * self.batches_len

    def forward(self, X_original, double_matrix=0, new_X=0, batches_i=0, training=1):
        if isinstance(X_original, np.ndarray) == False:
            X_original = X_original.numpy()
        if self.X[batches_i] == None:
            X_1 = []
            X_2 = []
            X = []
            for i in range(self.total_input):
                X_1.append(copy.deepcopy(X_original))
                for j in range(len(X_1[i])):
                    X_1[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_1[i], tf.transpose(X_1[i][j]))) % 1
                X_1[i] = tf.squeeze(X_1[i][:, :, 1:])
                if double_matrix:
                    X_2.append(copy.deepcopy(X_original))
                    for j in range(len(X_2[i])):
                        X_2[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_2[i], tf.transpose(X_2[i][j]))) % 1
                    X_2[i] = tf.squeeze(X_2[i][:, :, 1:])
                    X.append((X_1[i] + X_2[i]) % 1)
                else:
                    X.append(X_1[i])
            self.X[batches_i] = X
        X = self.X[batches_i]
        if new_X:
            X_1 = []
            X_2 = []
            X = []
            for i in range(self.total_input):
                X_1.append(copy.deepcopy(X_original))
                for j in range(len(X_1[i])):
                    X_1[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_1[i], tf.transpose(X_1[i][j]))) % 1
                X_1[i] = tf.squeeze(X_1[i][:, :, 1:])
                if double_matrix:
                    X_2.append(copy.deepcopy(X_original))
                    for j in range(len(X_2[i])):
                        X_2[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_2[i], tf.transpose(X_2[i][j]))) % 1
                    X_2[i] = tf.squeeze(X_2[i][:, :, 1:])
                    X.append((X_1[i] + X_2[i]) % 1)
                else:
                    X.append(X_1[i])
        Y_pred = []
        self.A2 = []
        for i in range(self.total_input):
            self.A2.append(self.dense_input[i].forward(X[i], training=training))
            Y_pred.append(self.A2[i])
        Y_pred_to_combinator = Y_pred[0]
        for i in range(1, self.total_input):
            Y_pred_to_combinator = tf.concat((Y_pred_to_combinator, Y_pred[i]), axis=1)
        Y_pred_final = self.dense_combinator.forward(Y_pred_to_combinator, training=training)
        return Y_pred_final
    
    def backward(self, X_original, A2, dA2, double_matrix=0, new_X=0, batches_i=0):
        if isinstance(X_original, np.ndarray) == False:
            X_original = X_original.numpy()
        if self.X[batches_i] == None:
            X_1 = []
            X_2 = []
            X = []
            for i in range(self.total_input):
                X_1.append(copy.deepcopy(X_original))
                for j in range(len(X_1[i])):
                    X_1[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_1[i], tf.transpose(X_1[i][j]))) % 1
                X_1[i] = tf.squeeze(X_1[i][:, :, 1:])
                if double_matrix:
                    X_2.append(copy.deepcopy(X_original))
                    for j in range(len(X_2[i])):
                        X_2[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_2[i], tf.transpose(X_2[i][j]))) % 1
                    X_2[i] = tf.squeeze(X_2[i][:, :, 1:])
                    X.append((X_1[i] + X_2[i]) % 1)
                else:
                    X.append(X_1[i])
            self.X[batches_i] = X
        X = self.X[batches_i]
        if new_X:
            X_1 = []
            X_2 = []
            X = []
            for i in range(self.total_input):
                X_1.append(copy.deepcopy(X_original))
                for j in range(len(X_1[i])):
                    X_1[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_1[i], tf.transpose(X_1[i][j]))) % 1
                X_1[i] = tf.squeeze(X_1[i][:, :, 1:])
                if double_matrix:
                    X_2.append(copy.deepcopy(X_original))
                    for j in range(len(X_2[i])):
                        X_2[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_2[i], tf.transpose(X_2[i][j]))) % 1
                    X_2[i] = tf.squeeze(X_2[i][:, :, 1:])
                    X.append((X_1[i] + X_2[i]) % 1)
                else:
                    X.append(X_1[i])
        self.A1 = self.dense_input[0].A1
        for i in range(1, self.total_input):
            self.A1 = tf.concat((self.A1, self.dense_input[i].A1), axis=1)
        dA2_final = self.dense_combinator.backward(self.A1, A2, dA2)

        dA2 = self.dense_input[0].backward(X[0], self.A2[0], dA2_final[:, i * self.dense_input[0].W1.shape[1]:(i + 1) * self.dense_input[0].W1.shape[1]])
        for i in range(1, self.total_input):
            dA2 += self.dense_input[i].backward(X[i], self.A2[i], dA2_final[:, i * self.dense_input[i].W1.shape[1]:(i + 1) * self.dense_input[i].W1.shape[1]])
        dA2 = tf.convert_to_tensor(dA2, dtype='float32')
        return dA2
    
    def save_model(self, filepath):
        for i in range(self.total_input):
            self.dense_input[i].save_model(filepath)
        self.dense_combinator.save_model(filepath)
    
    def load_model(self, filepath):
        for i in range(self.total_input):
            self.dense_input[i].load_model(filepath)
        self.dense_combinator.load_model(filepath)

def train_model_words():
    samples_selected = 1000
    X = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy')
    print(len(X))
    '''for i in range(len(X)):
        res = ''
        for j in range(len(X[i])):
            index = round(X[i][j][1] * (len(all_tokens) + 1))
            if index < len(all_tokens):
                res += all_tokens_list[index]
        print(res)'''
    Y = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_y.npy')
    '''Y = Y.tolist()
    for i in range(len(Y)):
        Yi = Y[i]
        Y[i] = [0] * (len(all_tokens) + 1)
        Y[i][round(Yi * (len(all_tokens) + 1))] = 1
        #print(i, all_tokens_list[round(Yi * (len(all_tokens) + 1))], Y[i][0])
    Y = np.array(Y, dtype='float32')'''
    perm = np.random.permutation(len(X))
    X = X[perm]
    Y = Y[perm]
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            Y[i][j] /= len(all_tokens) + 1
    learning_rate = 0.0001
    hidden_size = 100
    total_input = 11
    batch_size = len(X) // 3
    chaostranform1 = ChaosTransformLayer(X.shape, hidden_size, hidden_size * total_input, 1, learning_rate=learning_rate / 100, total_input=total_input, batches_len=(len(X) + batch_size - 1) // batch_size)
    X = X.tolist()
    batches_x = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
    lastlen = len(batches_x[-1])
    for i in range(batch_size - lastlen):
        batches_x[-1].append(batches_x[-1][-1])
    batches_x = np.array(batches_x, dtype='float32')
    Y = Y.tolist()
    batches_y = [Y[i:i + batch_size] for i in range(0, len(X), batch_size)]
    lastlen = len(batches_y[-1])
    for i in range(batch_size - lastlen):
        batches_y[-1].append(batches_y[-1][-1])
    batches_y = np.array(batches_y, dtype='float32')
    print(batches_x.shape, batches_y.shape, mxlen)
    hidden_size *= total_input
    dense_middle1 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayer(hidden_size, hidden_size, mxlen, 4, learning_rate=learning_rate)
    filepath = './custom_models_2/test_model'
    loss_graph_x = []
    loss_graph_y = []
    accuracy_graph_x = []
    accuracy_graph_y = []
    avg_loss = [0 for i in range(100)]
    avg_accuracy = [0 for i in range(100)]
    threshold_loss = 0.03
    def trainint_loop(epoch):
        start_time = time.time()
        loss = 0
        accuracy = 0
        for i in range(len(batches_x)):
            Y_pred_chaos = chaostranform1.forward(batches_x[i], double_matrix=0, batches_i=i)
            Y_pred_middle1 = dense_middle1.forward(Y_pred_chaos)
            Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
            Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
            Y_pred_final = dense_output.forward(Y_pred_middle3, output=1)
            loss = (loss * i + dense_output.compute_loss(Y_pred_final, batches_y[i])) / (i + 1)
            accuracy = (accuracy * i + dense_output.compute_accuracy(Y_pred_final, batches_y[i])) / (i + 1)
            dA2 = dense_output.backward(Y_pred_middle3, Y_pred_final, (Y_pred_final - batches_y[i]) / batch_size, output=1)
            dA2 = dense_middle3.backward(Y_pred_middle2, Y_pred_middle3, dA2)
            dA2 = dense_middle2.backward(Y_pred_middle1, Y_pred_middle2, dA2)
            dA2 = dense_middle1.backward(Y_pred_chaos, Y_pred_middle1, dA2)
            dA2 = chaostranform1.backward(batches_x[i], Y_pred_chaos, dA2, double_matrix=0, batches_i=i)
        validation_accuracy= 0
        avg_loss[epoch % 100] = loss
        avg_accuracy[epoch % 100] = accuracy
        if not (loss < 0.01 or accuracy > 0.9):
            print(f'Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Time: {time.time() - start_time}')
            print(f'avg loss: {sum(avg_loss) / 100}, avg accuracy: {sum(avg_accuracy) / 100}')
        if epoch % 100 == 0 or (loss < 0.01 or accuracy > 0.9) or epoch % 500 < 20:
            Y_pred_chaos = chaostranform1.forward(validation_x, double_matrix=0, new_X=1, training=0)
            Y_pred_middle1 = dense_middle1.forward(Y_pred_chaos, training=0)
            Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1, training=0)
            Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2, training=0)
            Y_pred_final = dense_output.forward(Y_pred_middle3, output=1, training=0)
            validation_loss = dense_output.compute_loss(Y_pred_final, validation_y)
            validation_accuracy = dense_output.compute_accuracy(Y_pred_final, validation_y)
            print(f'    epoch: {epoch}, validation loss: {validation_loss}, validation accuracy: {validation_accuracy}')
        if epoch % 100 == 0 or (loss < 0.01 or accuracy > 0.9) and 0 or validation_accuracy > 0.8:
            chaostranform1.save_model(filepath)
            dense_middle1.save_model(filepath)
            dense_middle2.save_model(filepath)
            dense_middle3.save_model(filepath)
            dense_output.save_model(filepath)
        loss_graph_x.append(epoch)
        loss_graph_y.append(loss)
        accuracy_graph_x.append(epoch)
        accuracy_graph_y.append(accuracy)
        return (loss, accuracy, validation_accuracy)

    for epoch in range(1,30001):
        loss, accuracy, val_accuracy = trainint_loop(epoch)
        if (loss < threshold_loss or accuracy > 0.98) and 0 or val_accuracy > 0.8:
            break
    figure, axis = plt.subplots(1, 2)
    axis[0].plot(loss_graph_x, loss_graph_y)
    axis[0].set_title('loss')
    axis[1].plot(accuracy_graph_x, accuracy_graph_y)
    axis[1].set_title('accuracy')
    plt.show()

def train_model_simple():
    samples_selected = 1000
    X = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy')
    print(len(X))
    X = X.tolist()
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = X[i][j][1]
    X = np.array(X, dtype='float32')
    print(X.shape)
    global validation_x
    validation_x = validation_x.tolist()
    for i in range(len(validation_x)):
        for j in range(len(validation_x[i])):
            validation_x[i][j] = validation_x[i][j][1]
    validation_x = np.array(validation_x, dtype='float32')
    Y = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_y.npy')
    '''Y = Y.tolist()
    for i in range(len(Y)):
        Yi = Y[i]
        Y[i] = [0] * (len(all_tokens) + 1)
        Y[i][round(Yi * (len(all_tokens) + 1))] = 1
        #print(i, all_tokens_list[round(Yi * (len(all_tokens) + 1))], Y[i][0])
    Y = np.array(Y, dtype='float32')'''
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            Y[i][j] /= len(all_tokens) + 1
    print(X.shape, Y.shape)
    learning_rate = 0.0001
    hidden_size = 100
    total_input = 11
    batch_size = 32
    chaostranform1 = FeedForwardLayer(X.shape[1], hidden_size * total_input, hidden_size * total_input, 0, learning_rate=learning_rate)
    hidden_size *= total_input
    dense_middle1 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayer(hidden_size, hidden_size, mxlen, 4, learning_rate=learning_rate)
    X = X.tolist()
    batches_x = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
    lastlen = len(batches_x[-1])
    for i in range(batch_size - lastlen):
        batches_x[-1].append(batches_x[-1][-1])
    batches_x = np.array(batches_x, dtype='float32')
    Y = Y.tolist()
    batches_y = [Y[i:i + batch_size] for i in range(0, len(X), batch_size)]
    lastlen = len(batches_y[-1])
    for i in range(batch_size - lastlen):
        batches_y[-1].append(batches_y[-1][-1])
    batches_y = np.array(batches_y, dtype='float32')
    print(batches_x.shape, batches_y.shape, mxlen)
    filepath = './custom_models/test_model'
    loss_graph_x = []
    loss_graph_y = []
    accuracy_graph_x = []
    accuracy_graph_y = []
    avg_loss = [0 for i in range(100)]
    avg_accuracy = [0 for i in range(100)]
    threshold_loss = 0.03
    def trainint_loop(epoch):
        start_time = time.time()
        loss = 0
        accuracy = 0
        for i in range(len(batches_x)):
            Y_pred_chaos = chaostranform1.forward(batches_x[i])
            Y_pred_middle1 = dense_middle1.forward(Y_pred_chaos)
            Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
            Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
            Y_pred_final = dense_output.forward(Y_pred_middle3, output=1)
            loss = (loss * i + dense_output.compute_loss(Y_pred_final, batches_y[i])) / (i + 1)
            accuracy = (accuracy * i + dense_output.compute_accuracy(Y_pred_final, batches_y[i])) / (i + 1)
            dA2 = dense_output.backward(Y_pred_middle3, Y_pred_final, (Y_pred_final - batches_y[i]) / 1, output=1)
            dA2 = dense_middle3.backward(Y_pred_middle2, Y_pred_middle3, dA2)
            dA2 = dense_middle2.backward(Y_pred_middle1, Y_pred_middle2, dA2)
            dA2 = dense_middle1.backward(Y_pred_chaos, Y_pred_middle1, dA2)
            dA2 = chaostranform1.backward(batches_x[i], Y_pred_chaos, dA2)
        validation_accuracy= 0
        avg_loss[epoch % 100] = loss
        avg_accuracy[epoch % 100] = accuracy
        print(f'Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Time: {time.time() - start_time}')
        print(f'avg loss: {sum(avg_loss) / 100}, avg accuracy: {sum(avg_accuracy) / 100}')
        '''if epoch % 100 == 0 or (loss < 0.01 or accuracy > 0.9) or epoch % 500 < 20:
            Y_pred_chaos = chaostranform1.forward(validation_x, double_matrix=0, new_X=1)
            Y_pred_middle1 = dense_middle1.forward(Y_pred_chaos)
            Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
            Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
            Y_pred_final = dense_output.forward(Y_pred_middle3, output=1)
            validation_loss = dense_output.compute_loss(Y_pred_final, validation_y)
            validation_accuracy = dense_output.compute_accuracy(Y_pred_final, validation_y)
            print(f'epoch: {epoch}, validation loss: {validation_loss}, validation accuracy: {validation_accuracy}')'''
        if epoch % 100 == 0 or (loss < 0.01 or accuracy > 0.9) and 0 or validation_accuracy > 0.8:
            chaostranform1.save_model(filepath)
            dense_middle1.save_model(filepath)
            dense_middle2.save_model(filepath)
            dense_middle3.save_model(filepath)
            dense_output.save_model(filepath)
        loss_graph_x.append(epoch)
        loss_graph_y.append(loss)
        accuracy_graph_x.append(epoch)
        accuracy_graph_y.append(accuracy)
        return (loss, accuracy, validation_accuracy)

    for epoch in range(1,30001):
        loss, accuracy, val_accuracy = trainint_loop(epoch)
        if loss < threshold_loss or accuracy > 0.98:
            break
    figure, axis = plt.subplots(1, 2)
    axis[0].plot(loss_graph_x, loss_graph_y)
    axis[0].set_title('loss')
    axis[1].plot(accuracy_graph_x, accuracy_graph_y)
    axis[1].set_title('accuracy')
    plt.show()


content = ''

def use_model_words():
    samples_selected = 0
    X = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_x.npy')[:samples_selected]
    for i in range(len(X)):
        res = ''
        for j in range(len(X[i])):
            index = round(X[i][j][1] * (len(all_tokens) + 1))
            if index < len(all_tokens):
                res += all_tokens_list[index]
        print(res)
    print(len(X))
    samples_selected = 1
    X_full = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_x.npy')[:samples_selected]
    
    input_shape = X_full.shape
    print(input_shape)
    learning_rate = 0.01
    chaostranform1 = ChaosTransformLayer(input_shape, 100, 100, 1, learning_rate=learning_rate / 10, total_input=13)
    dense_middle1 = FeedForwardLayer(100, 100, 100, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayer(100, 100, 100, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayer(100, 100, 100, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayer(100, 100, 2, 4, learning_rate=learning_rate)
    filepath = './custom_models_3/test_model'
    #filepath = './custom_models/test_model'
    chaostranform1.load_model(filepath)
    dense_middle1.load_model(filepath)
    dense_middle2.load_model(filepath)
    dense_middle3.load_model(filepath)
    dense_output.load_model(filepath)
    
    n = 10
    global content
    global mxlen
    mxlen *= 2
    print(mxlen)
    for _ in range(n):
        X = [[]]
        now = ''
        en = []
        for j in content:
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
        for i in range(len(en)):
            X[0].append([1, all_tokens[en[i]] / (len(all_tokens) + 1)])
        nowlen = len(X[0])
        
        for i in range(mxlen - nowlen):
            X[0].append([1, len(all_tokens) / (len(all_tokens) + 1)])
        for i in range(mxlen):
            X[0][i][0] = i / mxlen

        X = np.array(X, dtype='float32')
        X = np.concatenate((X, X), axis=0)
        Y_pred_ss = chaostranform1.forward(X, double_matrix=0, new_X=1)
        Y_pred_middle1 = dense_middle1.forward(Y_pred_ss)
        Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
        Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
        Y_pred_final = dense_output.forward(Y_pred_middle3, output=1)
        Y_pred_final = Y_pred_final.numpy()
        mxi = 0
        Y_sorted = []
        for i in range(len(Y_pred_final[0])):
            Y_sorted.append([Y_pred_final[0][i], i])
        Y_sorted = sorted(Y_sorted)
        for i in range(len(Y_sorted) - 10, len(Y_sorted)):
            if Y_sorted[i][1] < len(all_tokens):
                print(_, all_tokens_list[Y_sorted[i][1]], Y_sorted[i][0])
        if len(en) > 2:
            if en[-1] == all_tokens_list[Y_sorted[-1][1]] or en[-2] == all_tokens_list[Y_sorted[-1][1]]:
                if en[-2] == all_tokens_list[Y_sorted[-2][1]] and Y_sorted[-3][1] != len(all_tokens_list):
                    content += all_tokens_list[Y_sorted[-3][1]]
                else:
                    content += all_tokens_list[Y_sorted[-2][1]]
            else:
                content += all_tokens_list[Y_sorted[-1][1]]
        else:
            content += all_tokens_list[Y_sorted[-1][1]]
        print(content)

def use_model_seq2seq():
    samples_selected = 0
    X = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy')[:samples_selected]
    for i in range(len(X)):
        res = ''
        for j in range(len(X[i])):
            index = round(X[i][j][1] * (len(all_tokens) + 1))
            if index < len(all_tokens):
                res += all_tokens_list[index]
        print(res)
    print(len(X))
    samples_selected = 1000
    X_full = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy')[:samples_selected]
    Y = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_y.npy')[:samples_selected]
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            Y[i][j] /= len(all_tokens) + 1
    
    input_shape = X_full.shape
    print(input_shape)
    learning_rate = 0.01
    chaostranform1 = ChaosTransformLayer(input_shape, 100, 100, 1, learning_rate=learning_rate / 10, total_input=13)
    dense_middle1 = FeedForwardLayer(100, 100, 100, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayer(100, 100, 100, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayer(100, 100, 100, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayer(100, 100, 2, 4, learning_rate=learning_rate)
    filepath = './custom_models/test_model'
    #filepath = './custom_models/test_model'
    chaostranform1.load_model(filepath)
    dense_middle1.load_model(filepath)
    dense_middle2.load_model(filepath)
    dense_middle3.load_model(filepath)
    dense_output.load_model(filepath)
    
    n = 10
    global content
    global mxlen
    #mxlen *= 2
    X = [[]]
    now = ''
    en = []
    for j in content:
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
    for i in range(len(en)):
        X[0].append([1, all_tokens[en[i]] / (len(all_tokens) + 1)])
    nowlen = len(X[0])
    
    for i in range(mxlen - nowlen):
        X[0].append([1, len(all_tokens) / (len(all_tokens) + 1)])
    for i in range(mxlen):
        X[0][i][0] = i / mxlen
    X = np.array(X, dtype='float32')
    X = np.concatenate((X, X), axis=0)
    Y_pred_ss = chaostranform1.forward(X, double_matrix=0, new_X=1)
    Y_pred_middle1 = dense_middle1.forward(Y_pred_ss)
    Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
    Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
    Y_pred_final = dense_output.forward(Y_pred_middle3, output=1)
    #print(f'loss: {dense_output.compute_loss(Y_pred_final, Y)}')
    Y_pred_final = Y_pred_final.numpy()
    print(Y_pred_final.shape)
    answer = ''
    for i in Y_pred_final[0]:
        j = round(i * (len(all_tokens) + 1))
        res = ''
        for k in range(j - 10, j + 10):
            if k < len(all_tokens_list):
                res += all_tokens_list[k] + ' '
        if j < len(all_tokens_list):
            answer += all_tokens_list[j]
        print(res)
    print(answer)

def train_model_keras():
    global mxlen
    X = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy')
    X = X.tolist()
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = X[i][j][1]
    X = np.array(X, dtype='float32')
    Y = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_y.npy')
    print(X.shape)
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            Y[i][j] /= len(all_tokens) + 1
    input_shape = (X.shape[1], 1, )
    inputs = keras.Input(shape=input_shape)
    lstm1 = layers.LSTM(1000, activation='relu', return_sequences=True)(inputs)
    lstm2 = layers.LSTM(100, activation='relu', return_sequences=False)(lstm1)
    #dense1 = layers.Dense(32, activation='relu')(flatten1)
    dense2 = layers.Dense(100, activation='relu')(lstm2)
    dense3 = layers.Dense(100, activation='relu')(dense2)
    flatten2 = layers.Flatten()(dense3)
    #layers.LSTM(200, activation='relu', return_sequences=False),
    dropout = layers.Dropout(0)(flatten2)
    outputs = layers.Dense(mxlen, activation='linear')(dropout)
    #model = keras.models.load_model('./models_trained/checker_keras.ckpt.keras')
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    batch_size = 16
    epochs = 200
    optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8)
    #optimizer = keras.optimizers.SGD(learning_rate=0.1)
    def custom_accuracy(Y_true, Y_pred):
        y_pred_classes = tf.round(Y_pred * (len(all_tokens) + 1))
        y_true_classes = tf.round(Y_true * (len(all_tokens) + 1))
        correct_predictions = tf.equal(y_true_classes, y_pred_classes)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    def custom_loss(Y_true, Y_pred):
        loss = tf.reduce_mean(tf.reduce_sum(tf.math.sqrt(abs(Y_true - Y_pred))))
        return loss
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=[custom_accuracy])
    model.summary()
    cp_callback = keras.callbacks.ModelCheckpoint(filepath='./models_trained/checker_keras_2.ckpt.keras')
    if 1:
        model.fit(X, Y, batch_size=batch_size, epochs=epochs, callbacks=[cp_callback])
    model = keras.models.load_model('./models_trained/checker_keras_2.ckpt.keras', custom_objects={'loss': custom_loss, 'metrics':[custom_accuracy]})
    global content
    #mxlen *= 2
    X = [[]]
    now = ''
    en = []
    for j in content:
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
    for i in range(len(en)):
        X[0].append([1, all_tokens[en[i]] / (len(all_tokens) + 1)])
    nowlen = len(X[0])
    
    for i in range(mxlen - nowlen):
        X[0].append([1, len(all_tokens) / (len(all_tokens) + 1)])
    for i in range(mxlen):
        X[0][i][0] = i / mxlen
    X = np.array(X, dtype='float32')
    X = X.tolist()
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = X[i][j][1]
    X = np.array(X, dtype='float32')
    print(model.predict(X))
    Y_pred_final = model.predict(X)
    print(Y_pred_final.shape)
    answer = ''
    for i in Y_pred_final[0]:
        j = round(i * (len(all_tokens) + 1))
        res = ''
        for k in range(j - 10, j + 10):
            if k < len(all_tokens_list):
                res += all_tokens_list[k] + ' '
        if j < len(all_tokens_list):
            answer += all_tokens_list[j]
        print(res)
    print(answer)

evaluation_pairs = [
('go', 'gehe'),
('time', 'zeit'),
#('great', 'großartiger'),
('year', 'jahr'),
#('thing', 'sache'),
('way', 'weg'),
('long', 'lange'),
('day', 'tag'),
('man', 'mann'),
('last', 'letzte'),
('old', 'alter'),
('important', 'wichtiger'),
('be', 'sei'),
('good', 'gut'),
('a', 'eine'),
('person', 'person'),
('the', 'der'),
('first', 'erste'),
('do', 'mache'),
('it', 'es'),
('say', 'sage'),
('new', 'neue'),
('get', 'bekomme'),
('make', 'mache'),
('go', 'gehe'),
('little', 'kleine'),
('is', 'ist'),
('i', 'ich'),
('am', 'bin'),
('you', 'du'),
('are', 'bist'),
('he', 'er'),
('she', 'sie'),
('we', 'wir'),
('it is a new day', 'es ist eine neuer tag')
]

evaluation_pairs = [
('have', 'haben'),
('time', 'zeit'),
('year', 'jahr'),
('way', 'weg'),
('day', 'tag'),
('man', 'mann'),
('be a new man', 'sei eine neuer mann'),
('man is a man', 'mann ist ein mann'),
('be a good day', 'sei ein guter tag'),
('a man', 'eine mann'),
('it is a good day', 'es ist ein guter tag'),
('it is a new day', 'es ist ein neuer tag'),
]

evaluation_pairs = [(pair[0] + ' ', pair[1] + ' ') for pair in evaluation_pairs]

#evaluation_pairs += sentence_pairs

#_1: 1604
#_2: 1265, 0.265
#_3: 1497

def evaluate_model():
    samples_selected = 1
    X_full = np.load('/home/user/Desktop/datasets/translation_en_de_anki_x.npy')[:samples_selected]
    Y = np.load('/home/user/Desktop/datasets/translation_en_de_anki_y.npy')[:samples_selected]
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            Y[i][j] /= len(all_tokens) + 1
    
    input_shape = X_full.shape
    #print(X_full)
    print(input_shape)
    learning_rate = 0.01
    chaostranform1 = ChaosTransformLayer(input_shape, 100, 100, 1, learning_rate=learning_rate / 10, total_input=11)
    #chaostranform1 = FeedForwardLayer(input_shape[1], 100, 100, 0, learning_rate=learning_rate)
    dense_middle1 = FeedForwardLayer(100, 100, 100, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayer(100, 100, 100, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayer(100, 100, 100, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayer(100, 100, 2, 4, learning_rate=learning_rate)
    filepath = './custom_models_2/test_model'
    #filepath = './custom_models/test_model'
    chaostranform1.load_model(filepath)
    dense_middle1.load_model(filepath)
    dense_middle2.load_model(filepath)
    dense_middle3.load_model(filepath)
    dense_output.load_model(filepath)
    Ys = []
    for i in dense_output.W1:
        for j in i:
            Ys.append(j)
    plt.hist(Ys, bins=100)
    plt.show()
    global content
    global mxlen
    value = 0
    for n in range(len(evaluation_pairs)):
        content = evaluation_pairs[n][0]
        translation = evaluation_pairs[n][1]
        #print(mxlen, content)
        #mxlen *= 2
        X = [[]]
        now = ''
        en = []
        de = []
        for j in content:
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
        for j in translation:
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
        #print('en', en, len(en))
        for i in range(len(en)):
            X[0].append([1, all_tokens[en[i]] / (len(all_tokens) + 1)])
            #X[0].append(all_tokens[en[i]] / (len(all_tokens) + 1))
        nowlen = len(X[0])
        #print(X)

        for i in range(mxlen - nowlen):
            X[0].append([1, len(all_tokens) / (len(all_tokens) + 1)])
            #X[0].append(len(all_tokens) / (len(all_tokens) + 1))

        for i in range(mxlen):
            X[0][i][0] = i / mxlen
            pass
        X = np.array(X, dtype='float32')
        Y = validation_y[n]
        #print(X)
        X = np.concatenate((X, X), axis=0)
        Y_pred_ss = chaostranform1.forward(X, new_X=1)
        #Y_pred_middle1 = dense_middle1.forward(Y_pred_ss)
        #Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
        #Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
        Y_pred_final = dense_output.forward(Y_pred_ss, output=1)
        print(f'loss: {dense_output.compute_loss(Y_pred_final, Y)}, accuracy: {dense_output.compute_accuracy(Y_pred_final, Y)}')
        Y_pred_final = Y_pred_final.numpy()
        #print(Y_pred_final.shape)
        d = 0
        for i in range(len(de)):
            d += abs(round(Y_pred_final[0][i] * (len(all_tokens) + 1)) - all_tokens[de[i]])
        value += d * d
        #if d < 20:
        #    value += 1
        answer = ''
        for i in Y_pred_final[0]:
            j = round(i * (len(all_tokens) + 1))
            res = ''
            for k in range(j - 10, j + 10):
                if k < len(all_tokens_list) and k >= 0:
                    res += all_tokens_list[k] + ' '
            if j < len(all_tokens_list) and j >= 0:
                answer += all_tokens_list[j]
        print(evaluation_pairs[n][0] + ': ' + answer + ' ' + str(d))
    print(value / len(evaluation_pairs))

if 1:
    train_model_words()
    #train_model_simple()
elif 1:
    #use_model_words()
    #use_model_seq2seq()
    evaluate_model()
elif 1:
    train_model_keras()

'''
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

special_chars = ['„', '”', '"', '\'', '.', ',', '/', '\\', ' ', '   ', '*', ';', ':', '`', '[', ']', '{', '}', '!', '?', '\n', '$']

common_words=[
'water','away','good','want','over','how','did','man',
'going','where','would','or','took','school','think',
'home','who','didn','ran','know','bear','can','again',
'cat','long','things','new','after','wanted','eat',
'everyone','our','two','has','yes','play','take',
'thought','dog','well','find','more','i','ll',
'round','tree','magic','shouted','us','other',
'food','fox','through','way','been','stop',
'must','red','door','right','sea','these',
'began','boy','animals','never','next',
'first','work','lots','need','that',
'baby','fish','gave','mouse',
'something','bed','may','still','found','live',
'say','soon','night','narrator','small','car',
'couldn','t','three','head','king','town',
'I','ve','around','every','garden','fast',
'only','many','laughed','let','much',
'suddenly','told','another','great','why',
'cried','keep','room','last','jumped',
'because','even','am','before',
'gran','clothes',
'tell','key','fun','place','mother',
'sat','boat','window','sleep',
'feet','morning','queen',
'each','book','its',
'green','different',
'let','girl',
'which','inside',
'run','any',
'under','hat',
'snow','air',
'trees','bad',
'tea','top',
'eyes','fell',
'friends','box',
'dark','grandad',
'there','looking',
'end','than',
'best','better','hot','sun',
'across','gone','hard',
'floppy','really','wind',
'wish','eggs',
'once','please',
'thing','stopped',
'ever','miss',
'most','cold',
'park','lived',
'birds','duck',
'horse','rabbit',
'white','coming',
'he','s',
'river','liked',
'giant','looks',
'use','along',
'plants','dragon',
'pulled','we',
're','fly',
'grow','make'
]

#for sentences from 200 words:
#chaotic loss = 40 at 5500 epoch
#non chaotic loss didn't cross 40 over 10000 epochs

#for sentences from 70 words:
#chaotic loss >100 but accuracy >0.94 at 4600 epoch, also it can translate unseen examples
#non chaotic loss >200 and accuracy <80 over 10000 epochs

#2500 epochs for 0.98 accuracy for one matrix with random multiples
#3100 epochs for 0.98 accuracy for two matrices with random multiples
#3300 epochs for 0.98 accuracy for one matrix without random multiples

#chaotic transformer can achieve 0.99 of accuracy on the training set
#chaotic transformer can achieve 0.33 of true pseudo accuracy (the absolute deviation of the first translated word from the true one is less than 10)
#only dense layers can achieve 0.5 of true pseudo accuracy but can not achieve 0.9 of accuracy on the training set
#chaotic transformer also can achieve 0.5 of true pseudo accuracy but can achieve 0.99 of accuracy on the training set

all_tokens = {}

data_train = []
mxlen = 0

data_raw = sentence_pairs
#print(data_raw)

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
    with open('./mxlen.txt', 'w') as f:
        f.write(str(mxlen))
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
    for i in range(len(X)):
        nowlen = len(X[i])
        for j in range(mxlen - nowlen):
            X[i].append([1, len(all_tokens) / (len(all_tokens) + 1)])
        for j in range(mxlen):
            X[i][j][0] = j / mxlen
        nowlen = len(Y[i])
        for j in range(mxlen - nowlen):
            Y[i].append(len(all_tokens))
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')

    np.save('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy', X)
    np.save('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_y.npy', Y)

#data_preparation_common_seq2seq()

def data_preparation_common():
    global mxlen
    data_raw = sentence_pairs

    for i in range(len(data_raw)):
        en = []
        de = []
        now = ''
        for j in data_raw[i][0]:
            if j in special_chars:
                if now != '':
                    en.append(now)
                en.append(j)
                now = ''
            else:
                now += j.lower()
        if now != '':
            en.append(now)
            now = ''
        for j in data_raw[i][1]:
            if j in special_chars:
                if now != '':
                    de.append(now)
                de.append(j)
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
        mxlen = max([mxlen, len(de) + 10, len(en) + 10])
    with open('./all_tokens.txt', 'w') as f:
        for i in all_tokens:
            f.write(i + '\n')

    all_tokens_list = open('./all_tokens.txt', 'r').readlines()
    all_tokens_list = [token[:-1] for token in all_tokens_list]
    all_tokens_list = sorted(all_tokens_list)
    for i in range(len(all_tokens_list)):
        all_tokens[all_tokens_list[i]] = i
    
    for i in range(len(data_train)):
        for k in range(2):
            for j in range(len(data_train[i][k])):
                data_train[i][k][j] = [1, all_tokens[data_train[i][k][j]] / (len(all_tokens) + 1)]

    
    X = []
    Y = []
    for i in range(len(data_train)):
        to_add = []
        for j in range(len(data_train[i][0])):
            X.append(data_train[i][1] + to_add)
            Y.append(data_train[i][0][j][1])
            to_add.append([1, data_train[i][0][j][1]])
    mxlen *= 2
    for i in range(len(X)):
        nowlen = len(X[i])
        for j in range(mxlen - nowlen):
            X[i].append([1, len(all_tokens) / (len(all_tokens) + 1)])
        for j in range(mxlen):
            X[i][j][0] = j / mxlen
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')

    np.save('/home/user/Desktop/datasets/translation_en_de_train_common_x.npy', X)
    np.save('/home/user/Desktop/datasets/translation_en_de_train_common_y.npy', Y)

#data_preparation_common()

def data_preparation_wmt():
    global mxlen
    data_raw = []

    with open('/home/user/Desktop/datasets/wmt14_translate_de-en_train.csv') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            if i and len(row) == 2:
                data_raw.append(row)
            i += 1
            if i == 10000:
                break
    for i in range(len(data_raw)):
        de = []
        en = []
        now = ''
        check = 1
        for j in data_raw[i][0]:
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
        for j in data_raw[i][1]:
            if j in special_chars:
                if now != '':
                    if not now in common_words:
                        check = 0 
                    en.append(now + j)
                #en.append(j)
                now = ''
            else:
                now += j.lower()
        if now != '':
            if not now in common_words:
                check = 0 
            en.append(now)
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
    with open('./mxlen.txt', 'w') as f:
        f.write(str(mxlen))

#data_preparation_wmt()
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

def preprocess_translation_wmt():
    global mxlen

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
    for i in range(len(X)):
        nowlen = len(X[i])
        for j in range(mxlen - nowlen):
            X[i].append([1, len(all_tokens) / (len(all_tokens) + 1)])
        for j in range(mxlen):
            X[i][j][0] = j / mxlen
        nowlen = len(Y[i])
        for j in range(mxlen - nowlen):
            Y[i].append(len(all_tokens))
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')

    np.save('/home/user/Desktop/datasets/translation_en_de_train_wmt_x.npy', X)
    np.save('/home/user/Desktop/datasets/translation_en_de_train_wmt_y.npy', Y)

#preprocess_translation_wmt()

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

tf.random.set_seed(42)

class FeedForwardLayer:
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
        self.optimizer = AdamOptimizer(learning_rate=learning_rate / 10, epsilon=1e-8)
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
        dropout_mask = tf.cast(tf.random.uniform(self.Z1.shape) >= self.dropout, dtype='float32')
        self.A1 = self.A1 * dropout_mask
        self.Z2 = tf.linalg.matmul(self.A1, self.W2) + self.b2
        #self.learning_rate += random.uniform(-self.learning_rate / 9, self.learning_rate / 10)
        #self.learning_rate = max(self.learning_rate, 0.00001)
        self.mx = tf.math.reduce_max(abs(self.Z2)) + 1e-10
        epsilon = (1 / (self.optimizer.t + 1)) * 0
        #self.noise_matrix = tf.convert_to_tensor(np.diag(np.random.uniform(1 - epsilon, 1 + epsilon, self.Z2.shape[1])), dtype='float32')
        self.noise_matrix = tf.random.uniform(self.Z2.shape, 1 - epsilon, 1 + epsilon)
        if output:
            #return dropout_mask * tf.nn.softmax(self.Z2)
            #return tf.linalg.matmul(self.Z2, self.noise_matrix)
            return tf.nn.sigmoid(self.Z2) * self.noise_matrix
        else:
            #return tf.linalg.matmul(tf.nn.relu(self.Z2), self.noise_matrix)
            return tf.nn.relu(self.Z2) * self.noise_matrix

    def compute_loss(self, Y_pred, Y_true):
        #epsilon = 1e-8
        #loss = -tf.reduce_mean(tf.reduce_sum(Y_true * tf.math.log(Y_pred + epsilon), axis=-1))
        loss = tf.reduce_sum(tf.math.sqrt(abs(Y_true - Y_pred))) / (Y_pred.shape[0] * Y_pred.shape[1])
        l2_reg_cost = 0*(self.l2_lambda / 2) * (tf.reduce_sum(tf.square(self.W1)) + tf.reduce_sum(tf.square(self.W2)))
        return loss + l2_reg_cost

    def compute_accuracy(self, Y_pred, Y_true):
        #y_pred_classes = tf.argmax(Y_pred, axis=-1)
        #y_true_classes = tf.argmax(Y_true, axis=-1)
        y_pred_classes = tf.cast(tf.round(Y_pred * (len(all_tokens) + 1)), dtype='int32')
        y_true_classes = tf.cast(tf.round(Y_true * (len(all_tokens) + 1)), dtype='int32')
        threshold = 1
        correct_predictions = abs(y_true_classes - y_pred_classes) < threshold
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def backward(self, X, A2, dA2, output=0):
        m = dA2.shape[1] 
        #self.noise_matrix = tf.linalg.diag(1.0 / tf.linalg.diag_part(self.noise_matrix))
        self.noise_matrix = 1.0 / self.noise_matrix
        if output:
            #dZ2 = dA2 * tf.nn.softmax(A2)
            #dZ2 = tf.linalg.matmul(dA2, self.noise_matrix) #* tf.cast(A2 > 0, dtype='float32')
            dZ2 = dA2 * self.noise_matrix * tf.nn.sigmoid(A2) * (1 - tf.nn.sigmoid(A2))
        else:
            #dZ2 = tf.linalg.matmul(dA2, self.noise_matrix) * tf.cast(A2 > 0, dtype='float32')
            dZ2 = dA2 * self.noise_matrix * tf.cast(A2 > 0, dtype='float32')
        dW2 = tf.linalg.matmul(tf.transpose(self.A1), dZ2) / m + (self.l2_lambda * self.W2) / m
        db2 = tf.math.reduce_sum(dZ2, axis=0, keepdims=True) / m 

        
        dA1 = tf.linalg.matmul(dZ2, tf.transpose(self.W2))
        m = dA1.shape[1]
        dZ1 = dA1 * tf.cast(self.A1 > 0, dtype='float32')
        dW1 = tf.linalg.matmul(tf.transpose(X), dZ1) / m + (self.l2_lambda * self.W1) / m
        db1 = tf.math.reduce_sum(dZ1, axis=0, keepdims=True) / m 
        dA0 = tf.linalg.matmul(dZ1, tf.transpose(self.W1))

        mnval = -1
        mxval = 1
        dW1 = tf.clip_by_value(dW1, mnval, mxval)
        db1 = tf.clip_by_value(db1, mnval, mxval)
        dW2 = tf.clip_by_value(dW2, mnval, mxval)
        db2 = tf.clip_by_value(db2, mnval, mxval)
        #print(tf.math.reduce_max(abs(dW1)))

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

class ChaosTransformLayer:
    def __init__(self, input_shape, hidden_size, output_size, index, learning_rate=0.001, total_input=10, dropout=0):
        self.index = index
        self.total_input = total_input
        self.dense_input = []
        self.dense_middle = []
        for i in range(self.total_input):
            self.dense_input.append(FeedForwardLayer(input_size=input_shape[1], hidden_size=hidden_size, output_size=hidden_size, index='ssoriginp' + str(index) + str(i), learning_rate=learning_rate))
            self.dense_middle.append(FeedForwardLayer(input_size=hidden_size, hidden_size=hidden_size, output_size=hidden_size, index='ssorigmid' + str(index) + str(i), learning_rate=learning_rate))
        self.dense_combinator = FeedForwardLayer(input_size=self.total_input * hidden_size, hidden_size=self.total_input * hidden_size, output_size=output_size, index='sscomb' + str(index), learning_rate=learning_rate, dropout=dropout)
        #self.chaos_0 = np.random.rand(2, 2)
        self.chaos_10_9 = np.array([[5, 2], [7, 3]], dtype='float32')
        #self.chaos_10_9 = np.array([[1, 0], [0, 1]], dtype='float32')
        self.chaos_2_1 = np.array([[2, 1], [1, 1]], dtype='float32')
        self.mat_1 = np.array([[1, 0], [0, 1]], dtype='float32')
        self.chaos_1 = []
        self.mat_2 = np.array([[1, 0], [0, 1]], dtype='float32')
        self.chaos_2 = []
        for i in range(self.total_input):
            self.chaos_1.append(tf.convert_to_tensor(copy.deepcopy(self.mat_1), dtype='float32'))
            #self.chaos.append(tf.random.uniform((2, 2)))
            self.mat_1 = np.dot(self.mat_1, self.chaos_10_9)
            self.chaos_2.append(tf.convert_to_tensor(copy.deepcopy(self.mat_2), dtype='float32'))
            self.mat_2 = np.dot(self.mat_2, self.chaos_2_1)
        print(self.mat_1, self.mat_2)
        self.X = None

    def attention(self, Q, K, V):
        return tf.linalg.matmul(tf.nn.softmax(tf.linalg.matmul(Q, tf.transpose(K)) / (K.shape[1])** 0.5), V)
    
    def forward(self, X_original, double_matrix=0, new_X=0):
        if self.X == None:
            X_1 = []
            X_2 = []
            X = []
            for i in range(self.total_input):
                X_1.append(copy.deepcopy(X_original))
                for j in range(len(X_1[i])):
                    X_1[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_1[i], X_1[i][j].T)) % 1
                X_1[i] = tf.squeeze(X_1[i][:, :, 1:])
                if double_matrix:
                    X_2.append(copy.deepcopy(X_original))
                    for j in range(len(X_2[i])):
                        X_2[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_2[i], X_2[i][j].T)) % 1
                    X_2[i] = tf.squeeze(X_2[i][:, :, 1:])
                    X.append((X_1[i] + X_2[i]) % 1)
                else:
                    X.append(X_1[i])
            self.X = X
        X = self.X
        if new_X:
            X_1 = []
            X_2 = []
            X = []
            for i in range(self.total_input):
                X_1.append(copy.deepcopy(X_original))
                for j in range(len(X_1[i])):
                    X_1[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_1[i], X_1[i][j].T)) % 1
                X_1[i] = tf.squeeze(X_1[i][:, :, 1:])
                if double_matrix:
                    X_2.append(copy.deepcopy(X_original))
                    for j in range(len(X_2[i])):
                        X_2[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_2[i], X_2[i][j].T)) % 1
                    X_2[i] = tf.squeeze(X_2[i][:, :, 1:])
                    X.append((X_1[i] + X_2[i]) % 1)
                else:
                    X.append(X_1[i])
        Y_pred = []
        self.A2inp = []
        self.A2mid = []
        for i in range(self.total_input):
            self.A2inp.append(self.dense_input[i].forward(X[i]))
            #self.A2inp[i] = self.attention(self.A2inp[i], self.A2inp[i], self.A2inp[i])
            self.A2mid.append(self.dense_middle[i].forward(self.A2inp[i]))
            Y_pred.append(self.A2mid[i])
        Y_pred_to_combinator = Y_pred[0]
        for i in range(1, self.total_input):
            Y_pred_to_combinator = tf.concat((Y_pred_to_combinator, Y_pred[i]), axis=1)
        Y_pred_final = self.dense_combinator.forward(Y_pred_to_combinator)
        return Y_pred_final
    
    def backward(self, X_original, A2, dA2, double_matrix=0, new_X=0):
        if self.X == None:
            X_1 = []
            X_2 = []
            X = []
            for i in range(self.total_input):
                X_1.append(copy.deepcopy(X_original))
                for j in range(len(X_1[i])):
                    X_1[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_1[i], X_1[i][j].T)) % 1
                X_1[i] = tf.squeeze(X_1[i][:, :, 1:])
                if double_matrix:
                    X_2.append(copy.deepcopy(X_original))
                    for j in range(len(X_2[i])):
                        X_2[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_2[i], X_2[i][j].T)) % 1
                    X_2[i] = tf.squeeze(X_2[i][:, :, 1:])
                    X.append((X_1[i] + X_2[i]) % 1)
                else:
                    X.append(X_1[i])
            self.X = X
        X = self.X
        if new_X:
            X_1 = []
            X_2 = []
            X = []
            for i in range(self.total_input):
                X_1.append(copy.deepcopy(X_original))
                for j in range(len(X_1[i])):
                    X_1[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_1[i], X_1[i][j].T)) % 1
                X_1[i] = tf.squeeze(X_1[i][:, :, 1:])
                if double_matrix:
                    X_2.append(copy.deepcopy(X_original))
                    for j in range(len(X_2[i])):
                        X_2[i][j] = tf.transpose(tf.linalg.matmul(self.chaos_2[i], X_2[i][j].T)) % 1
                    X_2[i] = tf.squeeze(X_2[i][:, :, 1:])
                    X.append((X_1[i] + X_2[i]) % 1)
                else:
                    X.append(X_1[i])
        self.A1 = self.dense_middle[0].A1
        for i in range(1, self.total_input):
            self.A1 = tf.concat((self.A1, self.dense_middle[i].A1), axis=1)
        dA2_final = self.dense_combinator.backward(self.A1, A2, dA2)

        for i in range(1, self.total_input):
            dA2 = self.dense_middle[i].backward(self.A2inp[i], self.A2mid[i], dA2_final[:, i * self.dense_input[i].W1.shape[1]:(i + 1) * self.dense_input[i].W1.shape[1]])
            dA2 = self.dense_input[i].backward(X[i], self.A2inp[i], dA2)
        dA2 = tf.convert_to_tensor(dA2, dtype='float32')
        return dA2
    
    def save_model(self, filepath):
        np.save('./chaos_transform_matrix.npy', self.chaos_1)
        for i in range(self.total_input):
            self.dense_input[i].save_model(filepath)
            self.dense_middle[i].save_model(filepath)
        self.dense_combinator.save_model(filepath)
    
    def load_model(self, filepath):
        self.chaos_1 = np.load('./chaos_transform_matrix.npy')
        for i in range(self.total_input):
            self.dense_input[i].load_model(filepath)
            self.dense_middle[i].load_model(filepath)
        self.dense_combinator.load_model(filepath)

def train_model_words():
    samples_selected = 1000
    X = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy')
    print(len(X))
    Y = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_y.npy')
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            Y[i][j] /= len(all_tokens) + 1
    chaostranform1 = ChaosTransformLayer(X.shape, hidden_size, hidden_size * total_input, 1, learning_rate=learning_rate / 100, total_input=total_input)
    batch_size = len(X)
    X_shape = X.shape
    X = X.tolist()
    batches_x = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
    lastlen = len(batches_x[-1])
    for i in range(batch_size - lastlen):
        batches_x[-1].append(batches_x[-1][-1])
    batches_x = np.array(batches_x, dtype='float32')
    Y = Y.tolist()
    batches_y = [Y[i:i + batch_size] for i in range(0, len(X), batch_size)]
    lastlen = len(batches_y[-1])
    for i in range(batch_size - lastlen):
        batches_y[-1].append(batches_y[-1][-1])
    batches_y = np.array(batches_y, dtype='float32')
    print(X.shape, Y.shape, mxlen)
    learning_rate = 0.001
    hidden_size = 100
    total_input = 11
    hidden_size *= total_input
    dense_middle1 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayer(hidden_size, hidden_size, mxlen, 4, learning_rate=learning_rate)
    filepath = './custom_models_2/test_model'
    loss_graph_x = []
    loss_graph_y = []
    accuracy_graph_x = []
    accuracy_graph_y = []
    avg_loss = [0 for i in range(100)]
    avg_accuracy = [0 for i in range(100)]
    threshold_loss = 0.03
    def trainint_loop(epoch):
        start_time = time.time()
        Y_pred_chaos = chaostranform1.forward(X, double_matrix=0)
        #Y_pred_middle1 = dense_middle1.forward(Y_pred_chaos)
        #Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
        #Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
        Y_pred_final = dense_output.forward(Y_pred_chaos, output=1)
        loss = dense_output.compute_loss(Y_pred_final, Y)
        avg_loss[epoch % 100] = loss
        accuracy = dense_output.compute_accuracy(Y_pred_final, Y)
        avg_accuracy[epoch % 100] = accuracy
        dA2 = dense_output.backward(Y_pred_chaos, Y_pred_final, Y_pred_final - Y, output=1)
        #dA2 = dense_middle3.backward(Y_pred_middle2, Y_pred_middle3, dA2)
        #dA2 = dense_middle2.backward(Y_pred_middle1, Y_pred_middle2, dA2)
        #dA2 = dense_middle1.backward(Y_pred_chaos, Y_pred_middle1, dA2)
        dA2 = chaostranform1.backward(X, Y_pred_chaos, dA2, double_matrix=0)
        val_accuracy = 0
        if not (loss < threshold_loss or accuracy > 0.98):
            print(f'epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Time: {time.time() - start_time}')
            print(f'average loss (per 100): {sum(avg_loss) / 100}, average accuracy: {sum(avg_accuracy) / 100}')
        if (epoch % 500 < 20 or epoch % 100 == 0 or loss < threshold_loss or accuracy > 0.98) and len(validation_x) > 0:
            Y_pred_chaos = chaostranform1.forward(validation_x, double_matrix=0, new_X=1)
            #Y_pred_middle1 = dense_middle1.forward(Y_pred_chaos)
            #Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
            #Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
            Y_pred_final = dense_output.forward(Y_pred_chaos, output=1)
            val_loss = dense_output.compute_loss(Y_pred_final, validation_y)
            val_accuracy = dense_output.compute_accuracy(Y_pred_final, validation_y)
            print(f'    epoch: {epoch}, validation loss: {val_loss}, validation accuracy: {val_accuracy}')
        if epoch % 100 == 0 or (loss < threshold_loss or accuracy > 0.98) and 0 or val_accuracy > 0.8:
            chaostranform1.save_model(filepath)
            dense_middle1.save_model(filepath)
            dense_middle2.save_model(filepath)
            dense_middle3.save_model(filepath)
            dense_output.save_model(filepath)
        loss_graph_x.append(epoch)
        loss_graph_y.append(loss)
        accuracy_graph_x.append(epoch)
        accuracy_graph_y.append(accuracy)
        return (loss, accuracy, val_accuracy)

    for epoch in range(1,30001):
        loss, accuracy, val_accuracy = trainint_loop(epoch)
        if (loss < threshold_loss or accuracy > 0.98) and 0 or val_accuracy > 0.8:
            break
    figure, axis = plt.subplots(1, 2)
    axis[0].plot(loss_graph_x, loss_graph_y)
    axis[0].set_title('loss')
    axis[1].plot(accuracy_graph_x, accuracy_graph_y)
    axis[1].set_title('accuracy')
    plt.show()

def train_model_simple():
    samples_selected = 1000
    X = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy')
    print(len(X))
    X = X.tolist()
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = X[i][j][1]
    X = np.array(X, dtype='float32')
    print(X.shape)
    global validation_x
    validation_x = validation_x.tolist()
    for i in range(len(validation_x)):
        for j in range(len(validation_x[i])):
            validation_x[i][j] = validation_x[i][j][1]
    validation_x = np.array(validation_x, dtype='float32')
    Y = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_y.npy')
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            Y[i][j] /= len(all_tokens) + 1
    print(X.shape, Y.shape)
    learning_rate = 0.0001
    hidden_size = 100
    total_input = 11
    chaostranform1 = FeedForwardLayer(X.shape[1], hidden_size * total_input, hidden_size * total_input, 0, learning_rate=learning_rate)
    hidden_size *= total_input
    dense_middle1 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayer(hidden_size, hidden_size, mxlen, 4, learning_rate=learning_rate)
    filepath = './custom_models/test_model'
    loss_graph_x = []
    loss_graph_y = []
    accuracy_graph_x = []
    accuracy_graph_y = []
    avg_loss = [0 for i in range(100)]
    avg_accuracy = [0 for i in range(100)]
    threshold_loss = 0.03
    def trainint_loop(epoch):
        start_time = time.time()
        Y_pred_chaos = chaostranform1.forward(X)
        Y_pred_middle1 = dense_middle1.forward(Y_pred_chaos)
        Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
        Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
        Y_pred_final = dense_output.forward(Y_pred_middle3, output=1)
        loss = dense_output.compute_loss(Y_pred_final, Y)
        avg_loss[epoch % 100] = loss
        accuracy = dense_output.compute_accuracy(Y_pred_final, Y)
        avg_accuracy[epoch % 100] = accuracy
        dA2 = dense_output.backward(Y_pred_middle3, Y_pred_final, Y_pred_final - Y, output=1)
        dA2 = dense_middle3.backward(Y_pred_middle2, Y_pred_middle3, dA2)
        dA2 = dense_middle2.backward(Y_pred_middle1, Y_pred_middle2, dA2)
        dA2 = dense_middle1.backward(Y_pred_chaos, Y_pred_middle1, dA2)
        dA2 = chaostranform1.backward(X, Y_pred_chaos, dA2)
        val_accuracy = 0
        print(f'epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Time: {time.time() - start_time}')
        print(f'average loss (per 100): {sum(avg_loss) / 100}, average accuracy: {sum(avg_accuracy) / 100}')
        if (epoch % 500 < 20 or epoch % 100 == 0 or loss < threshold_loss or accuracy > 0.98) and len(validation_x) > 0:
            Y_pred_chaos = chaostranform1.forward(validation_x)
            Y_pred_middle1 = dense_middle1.forward(Y_pred_chaos)
            Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
            Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
            Y_pred_final = dense_output.forward(Y_pred_middle3, output=1)
            val_loss = dense_output.compute_loss(Y_pred_final, validation_y)    
            val_accuracy = dense_output.compute_accuracy(Y_pred_final, validation_y)
            print(f'validation loss: {val_loss}, validation accuracy: {val_accuracy}')
        if epoch % 100 == 0 or loss < threshold_loss or accuracy > 0.98:
            chaostranform1.save_model(filepath)
            dense_middle1.save_model(filepath)
            dense_middle2.save_model(filepath)
            dense_middle3.save_model(filepath)
            dense_output.save_model(filepath)
        loss_graph_x.append(epoch)
        loss_graph_y.append(loss)
        accuracy_graph_x.append(epoch)
        accuracy_graph_y.append(accuracy)
        return (loss, accuracy, val_accuracy)

    for epoch in range(1,30001):
        loss, accuracy, val_accuracy = trainint_loop(epoch)
        if loss < threshold_loss or accuracy > 0.98:
            break
    figure, axis = plt.subplots(1, 2)
    axis[0].plot(loss_graph_x, loss_graph_y)
    axis[0].set_title('loss')
    axis[1].plot(accuracy_graph_x, accuracy_graph_y)
    axis[1].set_title('accuracy')
    plt.show()


content = ''

def use_model_words():
    samples_selected = 0
    X = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_x.npy')[:samples_selected]
    for i in range(len(X)):
        res = ''
        for j in range(len(X[i])):
            index = round(X[i][j][1] * (len(all_tokens) + 1))
            if index < len(all_tokens):
                res += all_tokens_list[index]
        print(res)
    print(len(X))
    samples_selected = 1
    X_full = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_x.npy')[:samples_selected]
    
    input_shape = X_full.shape
    print(input_shape)
    learning_rate = 0.01
    chaostranform1 = ChaosTransformLayer(input_shape, 100, 100, 1, learning_rate=learning_rate / 10, total_input=13)
    dense_middle1 = FeedForwardLayer(100, 100, 100, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayer(100, 100, 100, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayer(100, 100, 100, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayer(100, 100, 2, 4, learning_rate=learning_rate)
    filepath = './custom_models_3/test_model'
    #filepath = './custom_models/test_model'
    chaostranform1.load_model(filepath)
    dense_middle1.load_model(filepath)
    dense_middle2.load_model(filepath)
    dense_middle3.load_model(filepath)
    dense_output.load_model(filepath)
    
    n = 10
    global content
    global mxlen
    mxlen *= 2
    print(mxlen)
    for _ in range(n):
        X = [[]]
        now = ''
        en = []
        for j in content:
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
        for i in range(len(en)):
            X[0].append([1, all_tokens[en[i]] / (len(all_tokens) + 1)])
        nowlen = len(X[0])
        
        for i in range(mxlen - nowlen):
            X[0].append([1, len(all_tokens) / (len(all_tokens) + 1)])
        for i in range(mxlen):
            X[0][i][0] = i / mxlen

        X = np.array(X, dtype='float32')
        X = np.concatenate((X, X), axis=0)
        Y_pred_ss = chaostranform1.forward(X, double_matrix=0, new_X=1)
        Y_pred_middle1 = dense_middle1.forward(Y_pred_ss)
        Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
        Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
        Y_pred_final = dense_output.forward(Y_pred_middle3, output=1)
        Y_pred_final = Y_pred_final.numpy()
        mxi = 0
        Y_sorted = []
        for i in range(len(Y_pred_final[0])):
            Y_sorted.append([Y_pred_final[0][i], i])
        Y_sorted = sorted(Y_sorted)
        for i in range(len(Y_sorted) - 10, len(Y_sorted)):
            if Y_sorted[i][1] < len(all_tokens):
                print(_, all_tokens_list[Y_sorted[i][1]], Y_sorted[i][0])
        if len(en) > 2:
            if en[-1] == all_tokens_list[Y_sorted[-1][1]] or en[-2] == all_tokens_list[Y_sorted[-1][1]]:
                if en[-2] == all_tokens_list[Y_sorted[-2][1]] and Y_sorted[-3][1] != len(all_tokens_list):
                    content += all_tokens_list[Y_sorted[-3][1]]
                else:
                    content += all_tokens_list[Y_sorted[-2][1]]
            else:
                content += all_tokens_list[Y_sorted[-1][1]]
        else:
            content += all_tokens_list[Y_sorted[-1][1]]
        print(content)

def use_model_seq2seq():
    samples_selected = 0
    X = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy')[:samples_selected]
    for i in range(len(X)):
        res = ''
        for j in range(len(X[i])):
            index = round(X[i][j][1] * (len(all_tokens) + 1))
            if index < len(all_tokens):
                res += all_tokens_list[index]
        print(res)
    print(len(X))
    samples_selected = 1000
    X_full = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy')[:samples_selected]
    Y = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_y.npy')[:samples_selected]
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            Y[i][j] /= len(all_tokens) + 1
    
    input_shape = X_full.shape
    print(input_shape)
    learning_rate = 0.01
    chaostranform1 = ChaosTransformLayer(input_shape, 100, 100, 1, learning_rate=learning_rate / 10, total_input=13)
    dense_middle1 = FeedForwardLayer(100, 100, 100, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayer(100, 100, 100, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayer(100, 100, 100, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayer(100, 100, 2, 4, learning_rate=learning_rate)
    filepath = './custom_models/test_model'
    #filepath = './custom_models/test_model'
    chaostranform1.load_model(filepath)
    dense_middle1.load_model(filepath)
    dense_middle2.load_model(filepath)
    dense_middle3.load_model(filepath)
    dense_output.load_model(filepath)
    
    n = 10
    global content
    global mxlen
    #mxlen *= 2
    X = [[]]
    now = ''
    en = []
    for j in content:
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
    for i in range(len(en)):
        X[0].append([1, all_tokens[en[i]] / (len(all_tokens) + 1)])
    nowlen = len(X[0])
    
    for i in range(mxlen - nowlen):
        X[0].append([1, len(all_tokens) / (len(all_tokens) + 1)])
    for i in range(mxlen):
        X[0][i][0] = i / mxlen
    X = np.array(X, dtype='float32')
    X = np.concatenate((X, X), axis=0)
    Y_pred_ss = chaostranform1.forward(X, double_matrix=0, new_X=1)
    Y_pred_middle1 = dense_middle1.forward(Y_pred_ss)
    Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
    Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
    Y_pred_final = dense_output.forward(Y_pred_middle3, output=1)
    #print(f'loss: {dense_output.compute_loss(Y_pred_final, Y)}')
    Y_pred_final = Y_pred_final.numpy()
    print(Y_pred_final.shape)
    answer = ''
    for i in Y_pred_final[0]:
        j = round(i * (len(all_tokens) + 1))
        res = ''
        for k in range(j - 10, j + 10):
            if k < len(all_tokens_list):
                res += all_tokens_list[k] + ' '
        if j < len(all_tokens_list):
            answer += all_tokens_list[j]
        print(res)
    print(answer)

def train_model_keras():
    global mxlen
    X = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy')
    X = X.tolist()
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = X[i][j][1]
    X = np.array(X, dtype='float32')
    Y = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_y.npy')
    print(X.shape)
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            Y[i][j] /= len(all_tokens) + 1
    input_shape = (X.shape[1], 1, )
    inputs = keras.Input(shape=input_shape)
    lstm1 = layers.LSTM(1000, activation='relu', return_sequences=True)(inputs)
    lstm2 = layers.LSTM(100, activation='relu', return_sequences=False)(lstm1)
    #dense1 = layers.Dense(32, activation='relu')(flatten1)
    dense2 = layers.Dense(100, activation='relu')(lstm2)
    dense3 = layers.Dense(100, activation='relu')(dense2)
    flatten2 = layers.Flatten()(dense3)
    #layers.LSTM(200, activation='relu', return_sequences=False),
    dropout = layers.Dropout(0)(flatten2)
    outputs = layers.Dense(mxlen, activation='linear')(dropout)
    #model = keras.models.load_model('./models_trained/checker_keras.ckpt.keras')
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    batch_size = 16
    epochs = 200
    optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8)
    #optimizer = keras.optimizers.SGD(learning_rate=0.1)
    def custom_accuracy(Y_true, Y_pred):
        y_pred_classes = tf.round(Y_pred * (len(all_tokens) + 1))
        y_true_classes = tf.round(Y_true * (len(all_tokens) + 1))
        correct_predictions = tf.equal(y_true_classes, y_pred_classes)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    def custom_loss(Y_true, Y_pred):
        loss = tf.reduce_mean(tf.reduce_sum(tf.math.sqrt(abs(Y_true - Y_pred))))
        return loss
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=[custom_accuracy])
    model.summary()
    cp_callback = keras.callbacks.ModelCheckpoint(filepath='./models_trained/checker_keras_2.ckpt.keras')
    if 1:
        model.fit(X, Y, batch_size=batch_size, epochs=epochs, callbacks=[cp_callback])
    model = keras.models.load_model('./models_trained/checker_keras_2.ckpt.keras', custom_objects={'loss': custom_loss, 'metrics':[custom_accuracy]})
    global content
    #mxlen *= 2
    X = [[]]
    now = ''
    en = []
    for j in content:
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
    for i in range(len(en)):
        X[0].append([1, all_tokens[en[i]] / (len(all_tokens) + 1)])
    nowlen = len(X[0])
    
    for i in range(mxlen - nowlen):
        X[0].append([1, len(all_tokens) / (len(all_tokens) + 1)])
    for i in range(mxlen):
        X[0][i][0] = i / mxlen
    X = np.array(X, dtype='float32')
    X = X.tolist()
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = X[i][j][1]
    X = np.array(X, dtype='float32')
    print(model.predict(X))
    Y_pred_final = model.predict(X)
    print(Y_pred_final.shape)
    answer = ''
    for i in Y_pred_final[0]:
        j = round(i * (len(all_tokens) + 1))
        res = ''
        for k in range(j - 10, j + 10):
            if k < len(all_tokens_list):
                res += all_tokens_list[k] + ' '
        if j < len(all_tokens_list):
            answer += all_tokens_list[j]
        print(res)
    print(answer)

evaluation_pairs = [
('go', 'gehe'),
('time', 'zeit'),
#('great', 'großartiger'),
('year', 'jahr'),
#('thing', 'sache'),
('way', 'weg'),
('long', 'lange'),
('day', 'tag'),
('man', 'mann'),
('last', 'letzte'),
('old', 'alter'),
('important', 'wichtiger'),
('be', 'sei'),
('good', 'gut'),
('a', 'eine'),
('person', 'person'),
('the', 'der'),
('first', 'erste'),
('do', 'mache'),
('it', 'es'),
('say', 'sage'),
('new', 'neue'),
('get', 'bekomme'),
('make', 'mache'),
('go', 'gehe'),
('little', 'kleine'),
('is', 'ist'),
('i', 'ich'),
('am', 'bin'),
('you', 'du'),
('are', 'bist'),
('he', 'er'),
('she', 'sie'),
('we', 'wir'),
('it is a new day', 'es ist eine neuer tag')
]

evaluation_pairs = [
('have', 'haben'),
('time', 'zeit'),
('year', 'jahr'),
('way', 'weg'),
('day', 'tag'),
('man', 'mann'),
('be a new man', 'sei eine neuer mann'),
('man is a man', 'mann ist ein mann'),
('be a good day', 'sei ein guter tag'),
('a man', 'eine mann'),
('it is a good day', 'es ist ein guter tag'),
('it is a new day', 'es ist ein neuer tag'),
]

evaluation_pairs = [(pair[0] + ' ', pair[1] + ' ') for pair in evaluation_pairs]

#evaluation_pairs += sentence_pairs

#_1: 1604
#_2: 1265, 0.265
#_3: 1497

def evaluate_model():
    samples_selected = 1
    X_full = np.load('/home/user/Desktop/datasets/translation_en_de_anki_x.npy')[:samples_selected]
    Y = np.load('/home/user/Desktop/datasets/translation_en_de_anki_y.npy')[:samples_selected]
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            Y[i][j] /= len(all_tokens) + 1
    
    input_shape = X_full.shape
    #print(X_full)
    print(input_shape)
    learning_rate = 0.01
    chaostranform1 = ChaosTransformLayer(input_shape, 100, 100, 1, learning_rate=learning_rate / 10, total_input=11)
    #chaostranform1 = FeedForwardLayer(input_shape[1], 100, 100, 0, learning_rate=learning_rate)
    dense_middle1 = FeedForwardLayer(100, 100, 100, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayer(100, 100, 100, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayer(100, 100, 100, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayer(100, 100, 2, 4, learning_rate=learning_rate)
    filepath = './custom_models_2/test_model'
    #filepath = './custom_models/test_model'
    chaostranform1.load_model(filepath)
    dense_middle1.load_model(filepath)
    dense_middle2.load_model(filepath)
    dense_middle3.load_model(filepath)
    dense_output.load_model(filepath)
    Ys = []
    for i in dense_output.W1:
        for j in i:
            Ys.append(j)
    plt.hist(Ys, bins=100)
    plt.show()
    global content
    global mxlen
    value = 0
    for n in range(len(evaluation_pairs)):
        content = evaluation_pairs[n][0]
        translation = evaluation_pairs[n][1]
        #print(mxlen, content)
        #mxlen *= 2
        X = [[]]
        now = ''
        en = []
        de = []
        for j in content:
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
        for j in translation:
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
        #print('en', en, len(en))
        for i in range(len(en)):
            X[0].append([1, all_tokens[en[i]] / (len(all_tokens) + 1)])
            #X[0].append(all_tokens[en[i]] / (len(all_tokens) + 1))
        nowlen = len(X[0])
        #print(X)

        for i in range(mxlen - nowlen):
            X[0].append([1, len(all_tokens) / (len(all_tokens) + 1)])
            #X[0].append(len(all_tokens) / (len(all_tokens) + 1))

        for i in range(mxlen):
            X[0][i][0] = i / mxlen
            pass
        X = np.array(X, dtype='float32')
        Y = validation_y[n]
        #print(X)
        X = np.concatenate((X, X), axis=0)
        Y_pred_ss = chaostranform1.forward(X, new_X=1)
        #Y_pred_middle1 = dense_middle1.forward(Y_pred_ss)
        #Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
        #Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
        Y_pred_final = dense_output.forward(Y_pred_ss, output=1)
        print(f'loss: {dense_output.compute_loss(Y_pred_final, Y)}, accuracy: {dense_output.compute_accuracy(Y_pred_final, Y)}')
        Y_pred_final = Y_pred_final.numpy()
        #print(Y_pred_final.shape)
        d = 0
        for i in range(len(de)):
            d += abs(round(Y_pred_final[0][i] * (len(all_tokens) + 1)) - all_tokens[de[i]])
        value += d * d
        #if d < 20:
        #    value += 1
        answer = ''
        for i in Y_pred_final[0]:
            j = round(i * (len(all_tokens) + 1))
            res = ''
            for k in range(j - 10, j + 10):
                if k < len(all_tokens_list) and k >= 0:
                    res += all_tokens_list[k] + ' '
            if j < len(all_tokens_list) and j >= 0:
                answer += all_tokens_list[j]
        print(evaluation_pairs[n][0] + ': ' + answer + ' ' + str(d))
    print(value / len(evaluation_pairs))

if 1:
    train_model_words()
    #train_model_simple()
elif 1:
    #use_model_words()
    #use_model_seq2seq()
    evaluate_model()
elif 1:
    train_model_keras()
'''