#to start 
#   conda activate tf-gpu
#   python main.py

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

sys.setrecursionlimit(100000)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#gpus = tf.config.list_physical_devices('GPU')
#tf.config.set_logical_device_configuration(
#        gpus[0],
#        [tf.config.LogicalDeviceConfiguration(memory_limit=3000)])

data = json.load(open('/home/user/Desktop/proofs_dataset/number_theory.json', 'r'))
data_train = []
with open('/home/user/Desktop/datasets/gsm8k_train.jsonl', 'r') as f:
    for line in f:
        data_train.append(json.loads(line))
all_tokens = {}

data_raw = open('/home/user/Desktop/datasets/logiqa_train.txt', 'r').readlines()
data_X = []
data_Y = []
alphabet = ''

for i in data_raw:
    alphabet += i
alphabet = ''.join(sorted(list(set(alphabet))))


def all_tokens_nt():
    for theorem in data['theorems']:
        for statement in theorem['contents']:
            for character in statement:
                all_tokens[character] = 1
        for proof in theorem['proofs']:
            for statement in proof['contents']:
                for character in statement:
                    all_tokens[character] = 1

def all_tokens_gsm8k():
    for sample in data_train:
        for char in sample['question']:
            all_tokens[char] = 1
    for sample in data_train:
        for char in sample['answer']:
            all_tokens[char] = 1

'''all_tokens_gsm8k()
#all_tokens_nt()

alphabet = ''
for token in all_tokens:
    alphabet += token'''

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

    def initialize(self, params):
        return

    def update(self, params, grads):
        result = copy.deepcopy(params)
        for key in params:
            result[key] += self.learning_rate * grads[key]
        return result

class AdaBelief:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-16):
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
        result = {}

        for key in params:
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            grad_diff = grads[key] - self.m[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * tf.square(grad_diff)

            m_hat = self.m[key] / (1 - tf.pow(self.beta1, self.t))
            v_hat = self.v[key] / (1 - tf.pow(self.beta2, self.t))

            result[key] = params[key] - (self.learning_rate * m_hat) / (tf.sqrt(v_hat) + self.epsilon)

        return result

class ScaledDotProductAttention:
    def __init__(self):
        pass

    def forward(self, query, key, value, mask=None):
        matmul_qk = tf.linalg.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(key)[-1], dtype='float32')
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9) 

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.linalg.matmul(attention_weights, value)
        return output, attention_weights

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
        self.attention_layer = ScaledDotProductAttention()
        self.optimizer = AdamOptimizer(learning_rate=learning_rate, epsilon=1e-7)
        #self.optimizer = SGD(learning_rate=learning_rate)
        #self.optimizer = AdaBelief(learning_rate=learning_rate, epsilon=1e-16)
        self.optimizer.initialize({
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
        })

    def forward(self, X, output=0, attention=0):
        self.Z1 = tf.linalg.matmul(X, self.W1) + self.b1
        self.A1 = tf.nn.relu(self.Z1)
        if attention:
            attention_output, _ = self.attention_layer.forward(self.A1, self.A1, self.A1)
            self.Z2 = tf.linalg.matmul(attention_output, self.W2) + self.b2
        else:
            self.Z2 = tf.linalg.matmul(self.A1, self.W2) + self.b2
        #self.learning_rate += random.uniform(-self.learning_rate / 9, self.learning_rate / 10)
        #self.learning_rate = max(self.learning_rate, 0.00001)
        dropout_mask = tf.cast(tf.random.uniform(self.Z2.shape) >= self.dropout, dtype='float32')
        if output:
            return dropout_mask * tf.nn.sigmoid(self.Z2)
        else:
            return dropout_mask * tf.nn.relu(self.Z2)

    def compute_loss(self, Y_pred, Y_true):
        return tf.reduce_mean(abs(Y_true - Y_pred))

    def compute_accuracy(self, Y_pred, Y_true):
        Y_pred_classes = tf.round(Y_pred)
        return tf.reduce_mean(tf.cast(tf.equal(Y_true, Y_pred_classes), dtype='float32'))

    def backward(self, X, A2, dA2, output=0):
        m = dA2.shape[0] 
        if output:
            dZ2 = dA2 * (tf.nn.sigmoid(A2) * (1 - tf.nn.sigmoid(A2)))
        else:
            dZ2 = dA2 * tf.cast(A2 > 0, dtype='float32')
        dW2 = tf.linalg.matmul(tf.transpose(self.A1), dZ2) / m
        db2 = tf.math.reduce_sum(dZ2, axis=0, keepdims=True) / m 

        dA1 = tf.linalg.matmul(dZ2, tf.transpose(self.W2))
        dZ1 = dA1 * tf.cast(self.A1 > 0, dtype='float32')
        dW1 = tf.linalg.matmul(tf.transpose(X), dZ1) / m 
        db1 = tf.math.reduce_sum(dZ1, axis=0, keepdims=True) / m 

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
        self.attention = ScaledDotProductAttention()
        self.dense_input = []
        for i in range(self.total_input):
            self.dense_input.append(FeedForwardLayer(input_size=input_shape[1], hidden_size=hidden_size, output_size=hidden_size, index='ssoriginp' + str(index) + str(i), learning_rate=learning_rate))
        self.dense_combinator = FeedForwardLayer(input_size=self.total_input * hidden_size, hidden_size=self.total_input * hidden_size, output_size=output_size, index='sscomb' + str(index), learning_rate=learning_rate)
        #self.chaos_0 = np.random.rand(2, 2)
        self.chaos_10_9 = np.array([[10, 9], [1, 1]], dtype='float32')
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

    def forward(self, X_original, double_matrix=0):
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
    
    def backward(self, X_original, A2, dA2, double_matrix=0):
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


def keras_example_model_xor():
    X = [[0, 0], [0, 1], [1, 0], [1, 1]] * 100
    Y = [[0], [1], [1], [0]] * 100
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')

    model = keras.Sequential([
        layers.Input((1, 2)),
        layers.Dense(5, activation='relu'),
        layers.Dense(5, activation='relu'),
        layers.Dense(1, activation='relu')
    ])

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    model.fit(X, Y, batch_size=1, epochs=10)

def train_model_symbols_keras():
    X_full = np.load('/home/user/Desktop/datasets/checker_train_x_flatten.npy') / (len(alphabet) + 1)
    def preprocess_chaos():
        X = []
        for i in range(len(X_full)):
            print(i)
            X.append([])
            for j in range(len(X_full[i])):
                X[i].append([j / len(X_full[i]), X_full[i][j]])
        X = np.array(X, dtype='float32')
        np.save('/home/user/Desktop/datasets/checker_train_x_chaos.npy', X)
    X = np.load('/home/user/Desktop/datasets/checker_train_x_chaos.npy')
    chaos = np.array([[2, 1], [1, 1]], dtype='uint64')
    for i in range(4):
        chaos = np.dot(chaos, chaos)
    chaos = np.array(chaos, dtype='float32')
    for i in range(len(X)):
        X[i] = np.dot(chaos, X[i].T).T
    X = np.reshape(X, (X.shape[0], 2 * X.shape[1], 1))
    Y = np.load('/home/user/Desktop/datasets/checker_train_y.npy')
    if 0:
        X_full = np.load('/home/user/Desktop/datasets/checker_train_x_chaos.npy')[:200]
        X = []
        for i in X_full:
            x = []
            for j in i:
                x.append(j[1])
            X.append(x)
        X = np.array(X, dtype='float32')
        Y_full = np.load('/home/user/Desktop/datasets/checker_train_y.npy')[:200]
        Y = []
        for i in range(len(Y_full)):
            Y.append([Y_full[i][1]])
        Y = np.array(Y, dtype='float32')
    else:
        samples_selected = 100
        X = np.concatenate((np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_x.npy')[:samples_selected], np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_x.npy')[x_len:x_len + samples_selected]), axis=0)
        shape = X[0].shape
        content = 'Adam bougth 6 apples with the price of $2 per apple. How many dollars Adam spent on apples in total? Adam spent $2*6=<<2*6=12>>$12 in total\n#### 12'
        X = [[]]
        for i in range(len(content)):
            X[0].append([1, alphabet.find(content[i]) / (len(alphabet) + 1)])
        for i in range(shape[0] - len(content)):
            X[0].append([1, len(alphabet) / (len(alphabet) + 1)])
        for i in range(shape[0]):
            X[0][i][0] = i / shape[0]
        print('now')
        #print(X)
        Y = np.concatenate((np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_y.npy')[:samples_selected], np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_y.npy')[x_len:x_len + samples_selected]), axis=0)
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
            change_num = random.randint(1, 10)
            change_from = []
            change_to = []
            for k in range(change_num):
                change_from.append(to_change[random.randint(0, len(to_change) - 1)])
            change_from = list(set(change_from))
            for k in range(change_num):
                change_to.append(str(random.randint(0, 9)))
                for k2 in range(1):
                    if random.randint(0, 9) < 3:
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
            X_add = X_add[:shape[0]]
            for j in range(len(X_add)):
                X_add[j][0] = j / len(X_add)
            X[-1] = copy.deepcopy(X_add)
        for i in range(samples_selected + 1):
            X.append(copy.deepcopy(X[i]))
            threshold = 0.3
            for j in range(len(X[0])):
                index = round(X[-1][j][1] * (len(alphabet) + 1))
                if index < len(alphabet):
                    if alphabet[index] in '0123456789' and random.uniform(0, 1) < threshold:
                        X[-1][j][1] = alphabet.find('0123456789'[random.randint(0,9)]) / (len(alphabet) + 1)
        Y = [[0, 1]] * (samples_selected + 1) + [[1, 0]] * (samples_selected + 1)
        X = np.array(X, dtype='float32')
        Y = np.array(Y, dtype='float32')
        X_full = copy.deepcopy(X)
        X = []
        for i in X_full:
            x = []
            for j in i:
                x.append(j[1])
            X.append(x)
        X = np.array(X, dtype='float32')
        print('now')
        print(X.shape, X)

    input_shape = (X.shape[1], 1, )
    inputs = keras.Input(shape=input_shape)
    lstm1 = layers.LSTM(100, activation='relu', return_sequences=True)(inputs)
    lstm2 = layers.LSTM(100, activation='relu', return_sequences=False)(lstm1)
    #dense1 = layers.Dense(32, activation='relu')(flatten1)
    dense2 = layers.Dense(100, activation='relu')(lstm2)
    dense3 = layers.Dense(100, activation='relu')(dense2)
    flatten2 = layers.Flatten()(dense3)
    #layers.LSTM(200, activation='relu', return_sequences=False),
    dropout = layers.Dropout(0.1)(flatten2)
    outputs = layers.Dense(2, activation='softmax')(dropout)
    #model = keras.models.load_model('./models_trained/checker_keras.ckpt.keras')
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    batch_size = 16
    epochs = 300
    optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8)
    #optimizer = keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def preprocessing():
        X = []
        Y = []
        for theorem in data['theorems']:
            if theorem['id'] < 1000 or True:
                print(theorem['id'])
                for proof in theorem['proofs']:
                    x = []
                    mxlen = 0
                    for line in proof['contents']:
                        x.append([])
                        for character in line:
                            x[-1].append(alphabet.find(character))
                        mxlen = min(input_shape[0], len(x))
                        to_X = copy.deepcopy(x)
                        for i in range(len(to_X)):
                            for j in range(input_shape[1] - len(to_X[i])):
                                to_X[i].append(len(alphabet))
                        if len(to_X) > input_shape[0]:
                            to_X = to_X[-input_shape[0]:]
                        for i in range(input_shape[0] - mxlen):
                            to_X.append([len(alphabet)] * input_shape[1])
                        X.append(copy.deepcopy(to_X))
                        Y.append([0, 1])
                    x = copy.deepcopy(to_X)
                    for i in range(mxlen - 1):
                        if len(x[i]) > input_shape[1]:
                            print(len(x[i]))
                        to_X = []
                        for j in range(input_shape[0]):
                            if j == i:
                                continue
                            to_X.append(copy.deepcopy(x[j]))
                        to_X.append([len(alphabet)] * input_shape[1])
                        X.append(copy.deepcopy(to_X))
                        Y.append([1, 0])
        X = np.array(X, dtype='float32')
        Y = np.array(Y, dtype='float32')
        np.save('/home/user/Desktop/datasets/checker_train_x.npy', X)
        np.save('/home/user/Desktop/datasets/checker_train_y.npy', Y)
    #preprocessing()
    model.summary()
    cp_callback = keras.callbacks.ModelCheckpoint(filepath='./models_trained/checker_keras_2.ckpt.keras')
    model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[cp_callback])

def use_model_symbols_keras():
    model = keras.models.load_model('./models_trained/checker_keras_2.ckpt.keras')
    input_shape = (20, 2000, )
    X = []
    Y = []
    mxlen = 0
    x = []
    proof = {
        "contents": [
                        "Consider the [[Definition:Natural Numbers|natural numbers]] $\\N$ defined as the [[Definition:Naturally Ordered Semigroup|naturally ordered semigroup]] $\\struct {S, \\circ, \\preceq}$.",
                        "From its definition, $\\strusdsssdsggs {S, \\circ, \\precgssfggfeq}$ is [[Definition:Well-Ordered Set|well-ordered]] by $\\preceq$.",
                        "The result follows.",
                        "As $\\N_{\\ne 0} = \\N \\setsrgrgssdfsinus \\set 0$, by [[Set Difference is Subset]] $\\N_{\\ne 0} \\subseteq \\N$.",
                        "As $\\N$ is [[Definition:Welsgdsgfsgfgsl-Ordered Set|well-ordered]], by definition, every subset of $\\N$ has a [[Definition:Smallest Element|smallest element]].",
                        "{{qed}}"
                    ],
    }
    for line in proof['contents']:
        x.append([])
        for character in line:
            x[-1].append(alphabet.find(character))
        mxlen = min(input_shape[0], len(x))
        to_X = copy.deepcopy(x)
        for i in range(len(to_X)):
            for j in range(input_shape[1] - len(to_X[i])):
                to_X[i].append(len(alphabet))
        if len(to_X) > input_shape[0]:
            to_X = to_X[-input_shape[0]:]
        for i in range(input_shape[0] - mxlen):
            to_X.append([len(alphabet)] * input_shape[1])
        X.append(copy.deepcopy(to_X))
        Y.append([0, 1])
    print(model.predict(X))

def preprocess_gsm8k():
    X = []
    print(np.random.normal())
    mid = 0
    for sample in data_train[:100]:
        X.append([])
        mid += 1
        print('first', mid)
        for i in range(len(sample['question'])):
            X[-1].append([1, alphabet.find(sample['question'][i]) / (len(alphabet) + 1)])
        for i in range(len(sample['answer'])):
            X[-1].append([1, alphabet.find(sample['answer'][i]) / (len(alphabet) + 1)])
        X_add = []
        for k in range(10):
            change_from = str(random.randint(0,9))
            change_to = str(random.randint(0,9))
            X_add.append([])
            for i in range(len(X[-1])):
                index = round(X[-1][i][1] * (len(alphabet) + 1))
                if index < len(alphabet):
                    if alphabet[index] == change_from:
                        X_add[k].append([1, alphabet.find(change_to) / (len(alphabet) + 1)])
                    else:
                        X_add[k].append(X[-1][i])
        X += copy.deepcopy(X_add)
    mxlen = 0
    for i in X:
        mxlen = max(mxlen, len(i))
    for j in range(len(X)):
        lenx = len(X[j])
        for i in range(mxlen - lenx):
            X[j].append([1, len(alphabet) / (len(alphabet) + 1)])
        for i in range(mxlen):
            X[j][i][0] = i / mxlen
    x_len = len(X)
    for i in range(x_len):
        print('last', i)
        X.append(copy.deepcopy(X[i]))
        for j in range(mxlen):
            index = round(X[-1][j][1] * (len(alphabet) + 1))
            if index < len(alphabet):
                if alphabet[index] in '0123456789' and random.uniform(0, 1) < 0.1:
                    X[-1][j][1] = alphabet.find('0123456789'[random.randint(0,9)]) / (len(alphabet) + 1)
    X = np.array(X, dtype='float32')
    np.save('/home/user/Desktop/datasets/gsm8k_train_chaos_x.npy', X)
    Y = [[1]] * x_len + [[0]] * x_len
    Y = np.array(Y, dtype='float32')
    np.save('/home/user/Desktop/datasets/gsm8k_train_chaos_y.npy', Y)

#preprocess_gsm8k()
x_len = 1100

def pseudo_random_pairing(n, k):
    X = []
    for i in range(k):
        X.append([random.randint(0, n - 1), random.randint(0, n - 1)])
    X = np.array(X, dtype='int32')
    np.save('./custom_models_2/pseudo_random_pairing.npy', X)

#pseudo_random_pairing(1064, 1064)

def train_model_symbols():
    #the best result so far is total_input = 11, A = [[10, 9], [1, 1]], input_size = hidden_size
    #bad with two smart selection layers
    #bad with middle <<input-middle->combinator>> layers
    if 0:
        X = np.load('/home/user/Desktop/datasets/checker_train_x_chaos.npy')[:200]
        Y_full = np.load('/home/user/Desktop/datasets/checker_train_y.npy')[:200]
        Y = []
        for i in range(len(Y_full)):
            Y.append([Y_full[i][1]])
        Y = np.array(Y, dtype='float32')
    elif 0:
        samples_selected = 100
        X = np.concatenate((np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_x.npy')[:samples_selected], np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_x.npy')[x_len:x_len + samples_selected]), axis=0)
        shape = X[0].shape
        content = 'Adam bougth 6 apples with the price of $2 per apple. How many dollars Adam spent on apples in total? Adam spent $2*6=<<2*6=12>>$12 in total\n#### 12'
        X = [[]]
        for i in range(len(content)):
            X[0].append([1, alphabet.find(content[i]) / (len(alphabet) + 1)])
        for i in range(shape[0] - len(content)):
            X[0].append([1, len(alphabet) / (len(alphabet) + 1)])
        for i in range(shape[0]):
            X[0][i][0] = i / shape[0]
        print('now')
        #print(X)
        Y = np.concatenate((np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_y.npy')[:samples_selected], np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_y.npy')[x_len:x_len + samples_selected]), axis=0)
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
            change_num = random.randint(1, 10)
            change_from = []
            change_to = []
            for k in range(change_num):
                change_from.append(to_change[random.randint(0, len(to_change) - 1)])
            change_from = list(set(change_from))
            for k in range(change_num):
                change_to.append(str(random.randint(0, 9)))
                for k2 in range(1):
                    if random.randint(0, 9) < 3:
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
            X_add = X_add[:shape[0]]
            for j in range(len(X_add)):
                X_add[j][0] = j / len(X_add)
            X[-1] = copy.deepcopy(X_add)
        for i in range(samples_selected + 1):
            X.append(copy.deepcopy(X[i]))
            threshold = 0.3
            for j in range(len(X[0])):
                index = round(X[-1][j][1] * (len(alphabet) + 1))
                if index < len(alphabet):
                    if alphabet[index] in '0123456789' and random.uniform(0, 1) < threshold:
                        X[-1][j][1] = alphabet.find('0123456789'[random.randint(0,9)]) / (len(alphabet) + 1)
        Y = [[1]] * (samples_selected + 1) + [[0]] * (samples_selected + 1)
        X = np.array(X, dtype='float32')
        Y = np.array(Y, dtype='float32')
    elif 1:
        samples_selected = 10000
        X = np.load('/home/user/Desktop/datasets/logiqa_chaos_x.npy')[:samples_selected]
        Y = np.load('/home/user/Desktop/datasets/logiqa_chaos_y.npy')[:samples_selected]
    '''X = X.tolist()
    mxlen = len(X[0])
    random_pairs = np.load('./custom_models_2/pseudo_random_pairing.npy')
    for i in range(len(X)):
        for j in range(mxlen):
            r1 = random_pairs[j][0]
            r2 = random_pairs[j][1]
            X[i].append([(X[i][r1][0] + X[i][r2][0]) % 1, (X[i][r1][1] + X[i][r2][1]) % 1])
    X = np.array(X, dtype='float32')'''
    print(X.shape)
    '''for i in range(len(X)):
        res = ''
        for j in range(len(X[i])):
            index = round(X[i][j][1] * (len(alphabet) + 1))
            if index < len(alphabet):
                res += alphabet[index]
        print(res, Y[i])'''
    learning_rate = 0.001
    hidden_size = 100
    total_input = 13
    chaostranform1 = ChaosTransformLayer(X.shape, hidden_size, hidden_size, 1, learning_rate=learning_rate / 100, total_input=total_input)
    dense_middle1 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayer(hidden_size, hidden_size, 1, 4, learning_rate=learning_rate, dropout=0)
    filepath = './custom_models_2/test_model'
    loss_graph_x = []
    loss_graph_y = []
    def trainint_loop(epoch):
        start_time = time.time()
        Y_pred_chaos = chaostranform1.forward(X, double_matrix=0)
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
        dA2 = chaostranform1.backward(X, Y_pred_chaos, dA2)
        print(f'Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Time: {time.time() - start_time}')
        chaostranform1.save_model(filepath)
        dense_middle1.save_model(filepath)
        dense_middle2.save_model(filepath)
        dense_middle3.save_model(filepath)
        dense_output.save_model(filepath)
        loss_graph_x.append(epoch)
        loss_graph_y.append(loss)
        return (loss, accuracy)

    for epoch in range(1,10001):
        loss, accuracy = trainint_loop(epoch)
        if loss < 0.16 or accuracy > 0.98:
            break
    plt.plot(loss_graph_x, loss_graph_y)
    plt.show()

content = {"question": "Natalia sold clips to 54 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
           "answer": "Natalia sold 54/2 = <<54/2=27>>27 clips in May.\nNatalia sold 54+27 = <<54+27=81>>81 clips altogether in April and May.\n#### 81"}
content = content['question'] + content['answer']
content = 'Adam bougth 5 apples with the price of $2 per apple. How many dollars Adam spent on apples in total? Adam spent $2*5=<<2*5=10>>$10 in total\n#### 32'
content = list(content)
for i in range(len(content)):
    if content[i] in '0123456789':
        if random.uniform(0,1) < 0:
            content[i] = str(random.randint(0,9))
content = ''.join(content)
content = 'All Cantonese don\'t like chili, so all southerners don\'t like chili.\nWhich of the following can guarantee the above argument?\nC.Some Cantonese are southerners and all southerners are Cantonese'

#print(content)

def use_model_symbols():
    #X_full = np.load('/home/user/Desktop/datasets/checker_train_x_chaos.npy')
    #X_full = np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_x.npy')
    X_full = np.load('/home/user/Desktop/datasets/logiqa_chaos_x.npy')
    Y = np.load('/home/user/Desktop/datasets/checker_train_y.npy')
    for i in range(len(Y)):
        if Y[i][1] == 0:
            print(i)
            break
    input_shape = X_full.shape
    print(input_shape)
    learning_rate = 0.01
    chaostranform1 = ChaosTransformLayer(input_shape, 100, 100, 1, learning_rate=learning_rate / 10, total_input=13)
    dense_middle1 = FeedForwardLayer(100, 100, 100, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayer(100, 100, 100, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayer(100, 100, 100, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayer(100, 100, 2, 4, learning_rate=learning_rate)
    filepath = './custom_models_2/test_model'
    chaostranform1.load_model(filepath)
    dense_middle1.load_model(filepath)
    dense_middle2.load_model(filepath)
    dense_middle3.load_model(filepath)
    dense_output.load_model(filepath)
    

    X = [[]]
    for i in range(len(content)):
        X[0].append([1, alphabet.find(content[i]) / (len(alphabet) + 1)])
    for i in range(input_shape[1] - len(content)):
        X[0].append([1, len(alphabet) / (len(alphabet) + 1)])
    for i in range(input_shape[1]):
        X[0][i][0] = i / input_shape[1]
    '''random_pairs = np.load('./custom_models_2/pseudo_random_pairing.npy')
    mxlen = len(X[0])
    for i in range(len(X)):
        for j in range(mxlen):
            r1 = random_pairs[j][0]
            r2 = random_pairs[j][1]
            X[i].append([(X[i][r1][0] + X[i][r2][0]) % 1, (X[i][r1][1] + X[i][r2][1]) % 1])'''
    
    X = np.array(X, dtype='float32')
    X = np.concatenate((X, X), axis=0)
    Y_pred_ss = chaostranform1.forward(X, double_matrix=0)
    Y_pred_middle1 = dense_middle1.forward(Y_pred_ss)
    Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
    Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
    Y_pred_final = dense_output.forward(Y_pred_middle3, output=1)
    print(Y_pred_final)

def train_model_symbols_simple():
    if 0:
        X_full = np.load('/home/user/Desktop/datasets/checker_train_x_chaos.npy')[:200]
        X = []
        for i in X_full:
            x = []
            for j in i:
                x.append(j[1])
            X.append(x)
        X = np.array(X, dtype='float32')
        Y_full = np.load('/home/user/Desktop/datasets/checker_train_y.npy')[:200]
        Y = []
        for i in range(len(Y_full)):
            Y.append([Y_full[i][1]])
        Y = np.array(Y, dtype='float32')
    if 0:
        X = np.load('/home/user/Desktop/datasets/checker_train_x_chaos.npy')[:200]
        Y_full = np.load('/home/user/Desktop/datasets/checker_train_y.npy')[:200]
        Y = []
        for i in range(len(Y_full)):
            Y.append([Y_full[i][1]])
        Y = np.array(Y, dtype='float32')
    elif 0:
        samples_selected = 100
        X = np.concatenate((np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_x.npy')[:samples_selected], np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_x.npy')[x_len:x_len + samples_selected]), axis=0)
        shape = X[0].shape
        content = 'Adam bougth 6 apples with the price of $2 per apple. How many dollars Adam spent on apples in total? Adam spent $2*6=<<2*6=12>>$12 in total\n#### 12'
        X = [[]]
        for i in range(len(content)):
            X[0].append([1, alphabet.find(content[i]) / (len(alphabet) + 1)])
        for i in range(shape[0] - len(content)):
            X[0].append([1, len(alphabet) / (len(alphabet) + 1)])
        for i in range(shape[0]):
            X[0][i][0] = i / shape[0]
        print('now')
        #print(X)
        Y = np.concatenate((np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_y.npy')[:samples_selected], np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_y.npy')[x_len:x_len + samples_selected]), axis=0)
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
            change_num = random.randint(1, 10)
            change_from = []
            change_to = []
            for k in range(change_num):
                change_from.append(to_change[random.randint(0, len(to_change) - 1)])
            change_from = list(set(change_from))
            for k in range(change_num):
                change_to.append(str(random.randint(0, 9)))
                for k2 in range(1):
                    if random.randint(0, 9) < 3:
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
            X_add = X_add[:shape[0]]
            for j in range(len(X_add)):
                X_add[j][0] = j / len(X_add)
            X[-1] = copy.deepcopy(X_add)
        for i in range(samples_selected + 1):
            X.append(copy.deepcopy(X[i]))
            threshold = 0.3
            for j in range(len(X[0])):
                index = round(X[-1][j][1] * (len(alphabet) + 1))
                if index < len(alphabet):
                    if alphabet[index] in '0123456789' and random.uniform(0, 1) < threshold:
                        X[-1][j][1] = alphabet.find('0123456789'[random.randint(0,9)]) / (len(alphabet) + 1)
        Y = [[1]] * (samples_selected + 1) + [[0]] * (samples_selected + 1)
        X = np.array(X, dtype='float32')
        Y = np.array(Y, dtype='float32')
        X_full = copy.deepcopy(X)
        X = []
        for i in X_full:
            x = []
            for j in i:
                x.append(j[1])
            X.append(x)
        X = np.array(X, dtype='float32')
        print('now')
        #print(X)
        #Y = np.concatenate((np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_y.npy')[:samples_selected], np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_y.npy')[x_len:x_len + samples_selected]), axis=0)
    elif 1:
        samples_selected = 2000
        X = np.load('/home/user/Desktop/datasets/logiqa_chaos_x.npy')[:samples_selected]
        Y = np.load('/home/user/Desktop/datasets/logiqa_chaos_y.npy')[:samples_selected]
        X_full = copy.deepcopy(X)
        X = []
        for i in X_full:
            x = []
            for j in i:
                x.append(j[1])
            X.append(x)
        X = np.array(X, dtype='float32')
        print('now')
    hidden_size = 100
    learning_rate = 0.001
    dense_input = FeedForwardLayer(X.shape[1], hidden_size, hidden_size, 0, learning_rate=learning_rate)
    dense_middle1 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 3, learning_rate=learning_rate)
    dense_middle4 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 4, learning_rate=learning_rate)
    dense_output = FeedForwardLayer(hidden_size, hidden_size, 1, 5, learning_rate=learning_rate, dropout=0)

    loss_graph_x = []
    loss_graph_y = []
    def trainint_loop(epoch):
        start_time = time.time()
        Y_pred_input = dense_input.forward(X)
        #print(Y_pred_ss)
        Y_pred_middle1 = dense_middle1.forward(Y_pred_input)
        Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
        Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
        Y_pred_middle4 = dense_middle4.forward(Y_pred_middle3)
        Y_pred_final = dense_output.forward(Y_pred_middle4, output=1)
        #print(Y_pred_final)
        loss = dense_output.compute_loss(Y_pred_final, Y)
        accuracy = dense_output.compute_accuracy(Y_pred_final, Y)
        dA2 = dense_output.backward(Y_pred_middle4, Y_pred_final, Y_pred_final - Y, output=1)
        dA2 = dense_middle4.backward(Y_pred_middle3, Y_pred_middle4, dA2)
        dA2 = dense_middle3.backward(Y_pred_middle2, Y_pred_middle3, dA2)
        dA2 = dense_middle2.backward(Y_pred_middle1, Y_pred_middle2, dA2)
        dA2 = dense_middle1.backward(Y_pred_input, Y_pred_middle1, dA2)
        _ = dense_input.backward(X, Y_pred_input, dA2)
        print(f'Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Time: {time.time() - start_time}')
        dense_input.save_model('./custom_models/test_model_simple')
        dense_middle1.save_model('./custom_models/test_model_simple')
        dense_middle2.save_model('./custom_models/test_model_simple')
        dense_middle3.save_model('./custom_models/test_model_simple')
        dense_middle4.save_model('./custom_models/test_model_simple')
        dense_output.save_model('./custom_models/test_model_simple')
        loss_graph_x.append(epoch)
        loss_graph_y.append(loss)
        return (loss, accuracy)

    for epoch in range(1,501):
        loss, accuracy = trainint_loop(epoch)
        if loss < 0.05 or accuracy > 0.95:
            break
    plt.plot(loss_graph_x, loss_graph_y)
    plt.show()

def use_model_symbols_simple():
    if 0:
        X_full = np.load('/home/user/Desktop/datasets/checker_train_x_chaos.npy')[:200]
        X = []
        for i in X_full:
            x = []
            for j in i:
                x.append(j[1])
            X.append(x)
        X = np.array(X, dtype='float32')
        Y_full = np.load('/home/user/Desktop/datasets/checker_train_y.npy')[:200]
        Y = []
        for i in range(len(Y_full)):
            Y.append([Y_full[i][1]])
        Y = np.array(Y, dtype='float32')
    elif 0:
        samples_selected = 50
        X_full = np.concatenate((np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_x.npy')[:samples_selected], np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_x.npy')[len(data_train):len(data_train) + samples_selected]), axis=0)
        X = []
        for i in X_full:
            x = []
            for j in i:
                x.append(j[1])
            X.append(x)
        X = np.array(X, dtype='float32')
        print('now')
        #print(X)
        Y = np.concatenate((np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_y.npy')[:samples_selected], np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_y.npy')[len(data_train):len(data_train) + samples_selected]), axis=0)
    elif 1:
        #X_full = np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_x.npy')
        X = np.load('/home/user/Desktop/datasets/logiqa_chaos_x.npy')
    hidden_size = 100
    learning_rate = 0.001
    input_shape = X.shape
    dense_input = FeedForwardLayer(X.shape[1], hidden_size, hidden_size, 0, learning_rate=learning_rate)
    dense_middle1 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 3, learning_rate=learning_rate)
    dense_middle4 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 4, learning_rate=learning_rate)
    dense_output = FeedForwardLayer(hidden_size, hidden_size, 1, 5, learning_rate=learning_rate, dropout=0)
    filepath = './custom_models/test_model_simple'
    dense_input.load_model(filepath)
    dense_middle1.load_model(filepath)
    dense_middle2.load_model(filepath)
    dense_middle3.load_model(filepath)
    dense_middle4.load_model(filepath)
    dense_output.load_model(filepath)


    X = [[]]
    for i in range(len(content)):
        X[0].append(alphabet.find(content[i]) / (len(alphabet) + 1))
    for i in range(input_shape[1] - len(content)):
        X[0].append(len(alphabet) / (len(alphabet) + 1))
    
    X = np.array(X, dtype='float32')
    X = np.concatenate((X, X), axis=0)
    Y_pred_input = dense_input.forward(X)
    Y_pred_middle1 = dense_middle1.forward(Y_pred_input)
    Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
    Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
    Y_pred_middle4 = dense_middle3.forward(Y_pred_middle3)
    Y_pred_final = dense_output.forward(Y_pred_middle4, output=1)
    print(Y_pred_final)

if 0:
    train_model_symbols()
    #train_model_symbols_simple()
elif 1:
    use_model_symbols()
    use_model_symbols_simple()
#train_model_symbols_keras()
#use_model_symbols_keras()

def preprocess_logiqa():
    mxlen = 0
    for i in range(0, len(data_raw), 8):
        res = data_raw[i + 2] + data_raw[i + 3]
        res_new = [res, res, res, res]
        for j in range(4):
            res_new[j] += data_raw[i + 4 + j]
        Y_add = [[0], [0], [0], [0]]
        if data_raw[i + 1][0] == 'a':
            Y_add[0][0] = 1
        if data_raw[i + 1][0] == 'b':
            Y_add[1][0] = 1
        if data_raw[i + 1][0] == 'c':
            Y_add[2][0] = 1
        if data_raw[i + 1][0] == 'd':
            Y_add[3][0] = 1
        for j in range(4):
            mxlen = max(mxlen, len(res_new[j]) + 100)
            data_X.append([])
            for k in range(len(res_new[j])):
                data_X[-1].append([1, alphabet.find(res_new[j][k]) / (len(alphabet) + 1)])
            data_Y.append(Y_add[j])

    for i in range(len(data_X)):
        item_len = len(data_X[i])
        for j in range(mxlen - item_len):
            data_X[i].append([1, len(alphabet) / (len(alphabet) + 1)])
        for j in range(mxlen):
            data_X[i][j][0] = j / mxlen
    
    

    X = np.array(data_X, dtype='float32')
    Y = np.array(data_Y, dtype='float32')

    np.save('/home/user/Desktop/datasets/logiqa_chaos_x.npy', X)
    np.save('/home/user/Desktop/datasets/logiqa_chaos_y.npy', Y)

#preprocess_logiqa()



