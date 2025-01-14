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
        
def train_model_chaos():
    samples_selected = 500
    validation_split = 100
    X = np.load('/home/user/Desktop/datasets/imdb_train_x.npy')[:samples_selected]
    Y = np.load('/home/user/Desktop/datasets/imdb_train_y.npy')[:samples_selected]
    validation_x = np.load('/home/user/Desktop/datasets/imdb_train_x.npy')[samples_selected:samples_selected + validation_split]
    validation_y = np.load('/home/user/Desktop/datasets/imdb_train_y.npy')[samples_selected:samples_selected + validation_split]

    learning_rate = 0.001
    hidden_size = 4
    total_input = 13
    chaostranform1 = ChaosTransformLayer(X.shape, hidden_size, hidden_size, 1, learning_rate=learning_rate / 100, total_input=total_input)
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
        Y_pred_chaos = chaostranform1.forward(X, double_matrix=1)
        Y_pred_middle1 = dense_middle1.forward(Y_pred_chaos)
        Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
        Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
        Y_pred_final = dense_output.forward(Y_pred_middle3, output=1)
        loss = dense_output.compute_loss(Y_pred_final, Y)
        accuracy = dense_output.compute_accuracy(Y_pred_final, Y)
        dA2 = dense_output.backward(Y_pred_middle3, Y_pred_final, Y_pred_final - Y, output=1)
        dA2 = dense_middle3.backward(Y_pred_middle2, Y_pred_middle3, dA2)
        dA2 = dense_middle2.backward(Y_pred_middle1, Y_pred_middle2, dA2)
        dA2 = dense_middle1.backward(Y_pred_chaos, Y_pred_middle1, dA2)
        dA2 = chaostranform1.backward(X, Y_pred_chaos, dA2, double_matrix=1)
        validation_loss = 1
        print(f'Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Time: {time.time() - start_time}')
        if epoch%100==0 or (loss < 0.01 or accuracy > 0.9) or epoch%100 < 1:
            Y_pred_chaos = chaostranform1.forward(validation_x, double_matrix=1, new_X=1)
            Y_pred_middle1 = dense_middle1.forward(Y_pred_chaos)
            Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
            Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
            Y_pred_final = dense_output.forward(Y_pred_middle3, output=1)
            validation_loss = dense_output.compute_loss(Y_pred_final, validation_y)
            validation_accuracy = dense_output.compute_accuracy(Y_pred_final, validation_y)
            print(f'validation loss: {validation_loss}, validation accuracy: {validation_accuracy}')
        if epoch%100==0 or (loss < 0.01 or accuracy > 0.99):
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
    hidden_size = 4
    total_input = 13
    chaostranform1 = FeedForwardLayer(X.shape[1], hidden_size, hidden_size * total_input, 0, learning_rate=learning_rate)
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
    conv1 = layers.Conv1D(10, kernel_size=3, activation='relu')(inputs)
    flatten1 = layers.Flatten()(conv1)
    #dense2 = layers.Dense(3, activation='relu')(flatten1)
    #dense3 = layers.Dense(3, activation='relu')(dense2)
    flatten2 = layers.Flatten()(flatten1)
    dropout = layers.Dropout(0)(flatten2)
    outputs = layers.Dense(2, activation='softmax')(dropout)
    #model = keras.models.load_model('./models_trained/checker_keras.ckpt.keras')
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    batch_size = 1
    epochs = 1000
    optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8)
    #optimizer = keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss='mae', metrics=['accuracy'])
    model.summary()
    cp_callback = keras.callbacks.ModelCheckpoint(filepath='./models_trained/checker_keras_2.ckpt.keras')
    model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=1/6, callbacks=[cp_callback])

if 1:
    #print(mxlen)
    #train_model_chaos()#even with only chaos_transform+output
    #train_model_simple()
    train_model_keras()