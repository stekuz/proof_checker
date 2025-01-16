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
        return float(tf.reduce_mean(abs(tf.cast(tf.round(Y_pred * (len(all_tokens) + 1)), dtype='float32') - 
                                      tf.cast(tf.round(Y_true * (len(all_tokens) + 1)), dtype='float32'))).numpy())

    def compute_accuracy(self, Y_pred, Y_true):
        #y_pred_classes = tf.argmax(Y_pred, axis=-1)
        #y_true_classes = tf.argmax(Y_true, axis=-1)
        y_pred_classes = tf.cast(tf.round(Y_pred * (len(all_tokens) + 1)), dtype='int32')
        y_true_classes = tf.cast(tf.round(Y_true * (len(all_tokens) + 1)), dtype='int32')
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

class ChaosTransformLayer:
    def __init__(self, input_shape, hidden_size, output_size, index, learning_rate=0.001, total_input=10):
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
        self.X = None

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
        self.A2 = []
        for i in range(self.total_input):
            self.A2.append(self.dense_input[i].forward(X[i]))
            Y_pred.append(self.A2[i])
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

class Encoder:
    def __init__(self, input_shape, hidden_size, output_size, learning_rate=0.001, total_input=10, double_matrix=0):
        self.chaos_transform = ChaosTransformLayer(input_shape, hidden_size, hidden_size, 'enc0', learning_rate=learning_rate / 100, total_input=total_input)
        self.middle = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 'enc1', learning_rate=learning_rate)
        self.output = FeedForwardLayer(hidden_size, hidden_size, output_size, 'enc2', learning_rate=learning_rate)
        self.double_matrix = double_matrix

    def forward(self, X, training=1):
        if training:
            self.Y_pred_chaos = self.chaos_transform.forward(X, double_matrix=self.double_matrix, new_X=0)
            self.Y_pred_middle = self.middle.forward(self.Y_pred_chaos)
            self.Y_pred_final = self.output.forward(self.Y_pred_middle)
            return self.Y_pred_final
        else:
            self.Y_pred_chaos = self.chaos_transform.forward(X, double_matrix=self.double_matrix, new_X=1)
            self.Y_pred_middle = self.middle.forward(self.Y_pred_chaos)
            self.Y_pred_final = self.output.forward(self.Y_pred_middle)
            return self.Y_pred_final
        
    def backward(self, X, dA2):
        dA2 = self.output.backward(self.Y_pred_middle, self.Y_pred_final, dA2)
        dA2 = self.middle.backward(self.Y_pred_chaos, self.Y_pred_middle, dA2)
        dA2 = self.chaos_transform.backward(X, self.Y_pred_chaos, dA2, double_matrix=self.double_matrix)
    
    def save_model(self, filepath):
        self.chaos_transform.save_model(filepath)
        self.middle.save_model(filepath)
        self.output.save_model(filepath)

    def load_model(self, filepath):
        self.chaos_transform.load_model(filepath)
        self.middle.load_model(filepath)
        self.output.load_model(filepath)

class Decoder:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.middle = FeedForwardLayer(input_size, hidden_size, hidden_size, 'dec0', learning_rate=learning_rate)
        self.output = FeedForwardLayer(hidden_size, hidden_size, output_size, 'dec1', learning_rate=learning_rate)

    def forward(self, X):
        self.Y_pred_middle = self.middle.forward(X)
        self.Y_pred_final = self.output.forward(self.Y_pred_middle)
        return self.Y_pred_final
    
    def backward(self, X, dA2):
        dA2 = self.output.backward(self.Y_pred_middle, self.Y_pred_final, dA2)
        dA2 = self.middle.backward(X, self.Y_pred_middle, dA2)
        return dA2
    
    def save_model(self, filepath):
        self.middle.save_model(filepath)
        self.output.save_model(filepath)

    def load_model(self, filepath):
        self.middle.load_model(filepath)
        self.output.load_model(filepath)

data_train = []
with open('/home/user/Desktop/datasets/gsm8k_train.jsonl', 'r') as f:
    for line in f:
        data_train.append(json.loads(line))

alphabet = 'abcdefghijklmnopqrstuvwxyz'
alphabet += alphabet.upper()
all_tokens = {}

def preprocess_gsm8k():
    mxlen = 0
    X = []
    Y = []
    for i in range(1000):
        question = data_train[i]['question']
        answer = data_train[i]['answer']
        question_tokens = []
        answer_tokens = []
        now = ''
        for j in question:
            if j not in alphabet:
                if now != '':
                    question_tokens.append(now)
                    all_tokens[now] = 1
                    now = ''
                all_tokens[j] = 1
                question_tokens.append(j)
            else:
                now += j.lower()
        if now != '':
            question_tokens.append(now)
            now = ''
        for j in answer:
            if j not in alphabet:
                if now != '':
                    answer_tokens.append(now)
                    all_tokens[now] = 1
                    now = ''
                all_tokens[j] = 1
                question_tokens.append(j)
            else:
                now += j.lower()
        if now != '':
            answer_tokens.append(now)
            now = ''
        X.append(question_tokens)
        Y.append(answer_tokens)
        mxlen = max([mxlen, len(question_tokens) + 10, len(answer_tokens) + 10])
    with open('./mxlen_gsm8k.txt', 'w') as f:
        f.write(str(mxlen))
    with open('./all_tokens_gsm8k.txt', 'w') as f:
        for token in all_tokens:
            f.write(token + '\n')
    all_tokens_list = open('./all_tokens_gsm8k.txt', 'r').readlines()
    all_tokens_list = [token[:-1] for token in all_tokens_list]
    all_tokens_list = sorted(all_tokens_list)
    for i in range(len(all_tokens_list)):
        all_tokens[all_tokens_list[i]] = i
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = [1, all_tokens[X[i][j]] / (len(all_tokens) + 1)]
        nowlen = len(X[i])
        for j in range(mxlen - nowlen):
            X[i].append([1, len(all_tokens) / (len(all_tokens) + 1)])
        for j in range(mxlen):
            X[i][j][0] = j / mxlen
        for j in range(len(Y[i])):
            Y[i][j] = [1, all_tokens[Y[i][j]] / (len(all_tokens) + 1)]
        nowlen = len(Y[i])
        for j in range(mxlen - nowlen):
            Y[i].append([1, len(all_tokens) / (len(all_tokens) + 1)])
        for j in range(mxlen):
            Y[i][j][0] = j / mxlen
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')
    np.save('/home/user/Desktop/datasets/gsm8k_train_chaos_x.npy', X)
    np.save('/home/user/Desktop/datasets/gsm8k_train_chaos_y.npy', Y)

#preprocess_gsm8k()

mxlen = int(open('./mxlen_gsm8k.txt', 'r').readlines()[0])
all_tokens_list = open('./all_tokens_gsm8k.txt', 'r').readlines()
all_tokens_list = [token[:-1] for token in all_tokens_list]
all_tokens_list = sorted(all_tokens_list)
for i in range(len(all_tokens_list)):
    all_tokens[all_tokens_list[i]] = i
print(all_tokens_list)

def train_model_chaos_detection():
    samples_selected = 100
    content = 'Adam bougth 3 apples with the price of $7 per apple. How many dollars Adam spent on apples in total? Adam spent $7*3=<<7*3=21>>$21 in total\n#### 21'
    mxlen = len(content) + 10
    X = [[]]
    for i in range(len(content)):
        X[0].append([1, alphabet.find(content[i]) / (len(alphabet) + 1)])
    for i in range(mxlen - len(content)):
        X[0].append([1, len(alphabet) / (len(alphabet) + 1)])
    for i in range(mxlen):
        X[0][i][0] = i / mxlen
    print('now')
    X = [X[0]]
    to_change = []
    for i in range(len(X[0])):
        index = round(X[0][i][1] * (len(alphabet) + 1))
        if index < len(alphabet):
            if alphabet[index] in '0123456789':
                to_change.append(alphabet[index])
    to_change = list(set(to_change))
    for i in range(samples_selected):
        X.append(copy.deepcopy(X[0]))
        change_num = random.randint(1, 5)
        change_from = []
        change_to = []
        for k in range(change_num):
            change_from.append(to_change[random.randint(0, len(to_change) - 1)])
        change_from = list(set(change_from))
        for k in range(change_num):
            change_to.append(str(random.randint(0, 9)))
            for k2 in range(1):
                if random.randint(0, 9) < 2:
                    change_to[k] += str(random.randint(0, 9))
        X_add = []
        for j in range(len(X[-1])):
            index = round(X[-1][j][1] * (len(alphabet) + 1))
            if index < len(alphabet):
                if alphabet[index] in change_from:
                    k = change_from.index(alphabet[index])
                    for k2 in range(len(change_to[k])):
                        X_add.append([1, alphabet.find(change_to[k][k2]) / (len(alphabet) + 1)])
                else:
                    X_add.append(X[-1][j])
            else:
                X_add.append(X[-1][j])
        X_add = X_add[:mxlen]
        for j in range(len(X_add)):
            X_add[j][0] = j / len(X_add)
        X[-1] = copy.deepcopy(X_add)
    for i in range(samples_selected + 1):
        X.append(copy.deepcopy(X[i]))
        threshold = 0.12
        for j in range(len(X[0])):
            index = round(X[-1][j][1] * (len(alphabet) + 1))
            if index < len(alphabet):
                if alphabet[index] in '0123456789' and random.uniform(0, 1) < threshold:
                    X[-1][j][1] = alphabet.find('0123456789'[random.randint(0,9)]) / (len(alphabet) + 1)
    Y = [[1]] * (samples_selected + 1) + [[0]] * (samples_selected + 1)
    X = np.array(X, dtype='float32')
    for i in range(len(X)):
        res = ''
        for j in range(len(X[i])):
            index = round(X[i][j][1] * (len(alphabet) + 1))
            if index < len(alphabet):
                res += alphabet[index]
        print(res)
    Y = np.array(Y, dtype='float32')

def train_model_chaos_generation():
    samples_selected = 200
    X = np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_x.npy')[:samples_selected]
    Y = np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_y.npy')[:samples_selected]
    Y = Y.tolist()
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            Y[i][j] = Y[i][j][1]
    Y = np.array(Y, dtype='float32')
    Y_toencode = Y.tolist()
    for i in range(len(Y_toencode)):
        for j in range(len(Y_toencode[i])):
            Y_toencode[i][j] = [j / len(Y_toencode[i]), Y_toencode[i][j]]
    Y_toencode = np.array(Y_toencode, dtype='float32')
    X_true_decoded = X.tolist()
    for i in range(len(X_true_decoded)):
        for j in range(len(X_true_decoded[i])):
            X_true_decoded[i][j] = X_true_decoded[i][j][1]
    X_true_decoded = np.array(X_true_decoded, dtype='float32')
    Y = X_true_decoded
    hidden_size = 100
    total_input = 13
    output_size = 100
    learning_rate = 0.0001
    x_encoder = Encoder(X.shape, hidden_size, output_size, learning_rate=learning_rate, total_input=total_input, double_matrix=1)
    #y_encoder = Encoder(Y.shape, hidden_size, output_size, learning_rate=learning_rate, total_input=total_input, double_matrix=1)
    decoder = Decoder(output_size, output_size, X.shape[1], learning_rate=learning_rate)
    filepath = './custom_models/test_model'
    loss_graph_x = []
    loss_graph_y = []
    accuracy_graph_x = []
    accuracy_graph_y = []
    avg_loss = [0 for i in range(100)]
    avg_accuracy = [0 for i in range(100)]
    def trainint_loop(epoch):
        start_time = time.time()
        X_encoded = x_encoder.forward(X)
        Y_decoded = decoder.forward(X_encoded)
        loss = decoder.output.compute_loss(Y_decoded, Y)
        accuracy = decoder.output.compute_accuracy(Y_decoded, Y)
        avg_loss[epoch % 100] = loss
        avg_accuracy[epoch % 100] = accuracy
        dA2 = decoder.backward(X_encoded, Y_decoded * (len(all_tokens) + 1) - 
                                      Y * (len(all_tokens) + 1))
        dA2x = x_encoder.backward(X, dA2)
        print(f'Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Time: {time.time() - start_time}')
        print(f'Average loss: {sum(avg_loss) / 100}, Average accuracy: {sum(avg_accuracy) / 100}')
        if epoch%100==0 or loss < 0.05 or accuracy > 0.8:
            x_encoder.save_model(filepath)
            decoder.save_model(filepath)
        loss_graph_x.append(epoch)
        loss_graph_y.append(loss)
        accuracy_graph_x.append(epoch)
        accuracy_graph_y.append(accuracy)
        return (loss, accuracy)

    for epoch in range(1,30001):
        loss, accuracy = trainint_loop(epoch)
        if loss < 0.05 or accuracy > 0.8:
            break
    figure, axis = plt.subplots(1, 2)
    axis[0].plot(loss_graph_x, loss_graph_y)
    axis[0].set_title('loss')
    axis[1].plot(accuracy_graph_x, accuracy_graph_y)
    axis[1].set_title('accuracy')
    plt.show()

content = {"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
 "answer": "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10"}
content = content['question']

def use_model_chaos_generation():
    filepath = './custom_models/test_model'
    input_shape = np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_x.npy')[:2].shape
    X = []
    content_tokens = []
    now = ''
    for j in content:
        if j not in alphabet:
            if now != '':
                content_tokens.append(now)
                all_tokens[now] = 1
                now = ''
            all_tokens[j] = 1
            content_tokens.append(j)
        else:
            now += j.lower()
    if now != '':
        content_tokens.append(now)
        now = ''
    X.append(content_tokens)
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = [1, all_tokens[X[i][j]] / (len(all_tokens) + 1)]
        nowlen = len(X[i])
        for j in range(mxlen - nowlen):
            X[i].append([1, len(all_tokens) / (len(all_tokens) + 1)])
        for j in range(mxlen):
            X[i][j][0] = j / mxlen
    X = np.array(X, dtype='float32')
    X = np.concatenate((X, X), axis=0)
    hidden_size = 100
    total_input = 13
    output_size = 50
    learning_rate = 0.001
    x_encoder = Encoder(X.shape, hidden_size, output_size, learning_rate=learning_rate, total_input=total_input, double_matrix=1)
    #encoder = Encoder(Y.shape, hidden_size, output_size, learning_rate=learning_rate, total_input=total_input, double_matrix=1)
    decoder = Decoder(output_size, output_size, X.shape[1], learning_rate=learning_rate)
    x_encoder.load_model(filepath)
    decoder.load_model(filepath)
    X_encoded = x_encoder.forward(X, training=0)
    Y_decoded = decoder.forward(X_encoded)
    print(Y_decoded.shape)
    res = ''
    for i in range(len(Y_decoded[0])):
        index = round(tf.math.round(Y_decoded[0][i] * (len(all_tokens) + 1)).numpy())
        if index < len(all_tokens):
            res += all_tokens_list[index]
    print(res)

if 1:
    print(mxlen)
    train_model_chaos_generation()
elif 1:
    use_model_chaos_generation()