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

sys.setrecursionlimit(100000)
os.environ['CUDA_VISIBLE_DEVICES'] = "0" 

#data = json.load(open('/home/user/Desktop/proofs_dataset/number_theory.json', 'r'))
data_train = []
with open('/home/user/Desktop/datasets/gsm8k_train.jsonl', 'r') as f:
    for line in f:
        data_train.append(json.loads(line))
all_tokens = {}

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

all_tokens_gsm8k()

alphabet = ''
for token in all_tokens:
    alphabet += token

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
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

    def backward(self, X, A2, dA2, loss, output=0):
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
    
    def load_model(self, filepath):
        self.W1 = np.load(filepath + 'dense' + str(self.index) + 'W1.npy')
        self.b1 = np.load(filepath + 'dense' + str(self.index) + 'b1.npy')
        self.W2 = np.load(filepath + 'dense' + str(self.index) + 'W2.npy')
        self.b2 = np.load(filepath + 'dense' + str(self.index) + 'b2.npy')

class SmartSelectionLayer:
    def __init__(self, input_shape, hidden_size, output_size, index, learning_rate=0.001):
        self.index = index
        self.dense_original = FeedForwardLayer(input_size=input_shape[1] * input_shape[2], hidden_size=hidden_size, output_size=hidden_size, index='ssorig' + str(index), learning_rate=learning_rate)
        self.dense_chaos = FeedForwardLayer(input_size=input_shape[1] * input_shape[2], hidden_size=hidden_size, output_size=hidden_size, index='sschaos' + str(index), learning_rate=learning_rate)
        self.dense_combinator = FeedForwardLayer(input_size=hidden_size * 2, hidden_size=2 * hidden_size, output_size=output_size, index='sscomb' + str(index), learning_rate=learning_rate)
        self.chaos = np.array([[2, 1], [1, 1]], dtype='uint64')
        for i in range(4):
            self.chaos = np.dot(self.chaos, self.chaos)
        self.chaos = tf.convert_to_tensor(self.chaos, dtype='float32')

    def forward(self, X):
        X_chaos = copy.deepcopy(X)
        for i in range(len(X_chaos)):
            X_chaos[i] = tf.transpose(tf.linalg.matmul(self.chaos, X_chaos[i].T)) % 1
        Y_pred_original = self.dense_original.forward(tf.reshape(X, (X.shape[0], X.shape[1] * X.shape[2])))
        self.A2_original = Y_pred_original
        Y_pred_chaos = self.dense_chaos.forward(tf.reshape(X_chaos, (X_chaos.shape[0], X_chaos.shape[1] * X_chaos.shape[2])))
        self.A2_chaos = Y_pred_chaos
        Y_pred_to_combinator = tf.concat((Y_pred_original, Y_pred_chaos), axis=1)
        Y_pred_final = self.dense_combinator.forward(Y_pred_to_combinator)
        return Y_pred_final
    
    def backward(self, X, Y_pred, Y_true, loss):
        X_chaos = copy.deepcopy(X)
        for i in range(len(X_chaos)):
            X_chaos[i] = tf.transpose(tf.linalg.matmul(self.chaos, X_chaos[i].T)) % 1
        X = tf.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]))
        X_chaos = tf.reshape(X_chaos, (X_chaos.shape[0], X_chaos.shape[1] * X_chaos.shape[2]))
        dA2_final = self.dense_combinator.backward(tf.concat((self.dense_original.A1, self.dense_chaos.A1), axis=1), Y_pred, Y_pred - Y_true, loss)
        dA2_original = self.dense_original.backward(X, self.A2_original, dA2_final[:, :self.dense_original.W1.shape[1]], loss)
        dA2_chaos = self.dense_chaos.backward(X_chaos, self.A2_chaos, dA2_final[:, self.dense_chaos.W1.shape[1]:], loss)
        return (dA2_original, dA2_chaos)
    
    def save_model(self, filepath):
        self.dense_original.save_model(filepath)
        self.dense_chaos.save_model(filepath)
        self.dense_combinator.save_model(filepath)
    
    def load_model(self, filepath):
        self.dense_original.load_model(filepath)
        self.dense_chaos.load_model(filepath)
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

def train_model_keras():
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
    input_shape = (X.shape[1], 1, )
    inputs = keras.Input(shape=input_shape)
    lstm1 = layers.LSTM(32, activation='relu', return_sequences=True)(inputs)
    conv1 = layers.Conv1D(32, (3), activation='relu')(lstm1)
    avgpool1 = layers.AveragePooling1D(100)(conv1)
    flatten1 = layers.Flatten()(avgpool1)
    #dense1 = layers.Dense(32, activation='relu')(flatten1)
    dense2 = layers.Dense(32, activation='relu')(flatten1)
    dense3 = layers.Dense(100, activation='relu')(dense2)
    flatten2 = layers.Flatten()(dense3)
    #layers.LSTM(200, activation='relu', return_sequences=False),
    dropout = layers.Dropout(0.1)(flatten2)
    outputs = layers.Dense(2, activation='softmax')(dropout)
    #model = keras.models.load_model('./models_trained/checker_keras.ckpt.keras')
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    batch_size = 64
    epochs = 300
    optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8)
    optimizer = keras.optimizers.SGD(learning_rate=0.1)
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

def use_model_keras():
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
    for sample in data_train:
        X.append([])
        for i in range(len(sample['answer'])):
            X[-1].append([i / len(sample['answer']), alphabet.find(sample['answer'][i]) / (len(alphabet) + 1)])
    mxlen = 0
    for i in X:
        mxlen = max(mxlen, len(i))
    for j in range(len(X)):
        lenx = len(X[j])
        for i in range(mxlen - lenx):
            X[j].append([1, len(alphabet) / (len(alphabet) + 1)])
        for i in range(mxlen):
            X[j][i][0] = i / mxlen
    for i in range(len(data_train)):
        X.append(X[i])
        for j in range(mxlen):
            if random.uniform(0, 1) < 0.3:
                X[-1][j][1] = len(alphabet) / (len(alphabet) + 1)
    X = np.array(X, dtype='float32')
    np.save('/home/user/Desktop/datasets/gsm8k_train_chaos_x.npy', X)
    Y = [[1]] * len(data_train) + [[0]] * len(data_train)
    Y = np.array(Y, dtype='float32')
    np.save('/home/user/Desktop/datasets/gsm8k_train_chaos_y.npy', Y)

#preprocess_gsm8k()

def train_model():
    #X = np.load('/home/user/Desktop/datasets/checker_train_x_chaos.npy')[:200]
    #Y_full = np.load('/home/user/Desktop/datasets/checker_train_y.npy')[:200]
    X = np.concatenate((np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_x.npy')[:100], np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_x.npy')[len(data_train):len(data_train) + 100]), axis=0)
    Y = np.concatenate((np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_y.npy')[:100], np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_y.npy')[len(data_train):len(data_train) + 100]), axis=0)
    print(X.shape)
    learning_rate = 0.001
    smartselection1 = SmartSelectionLayer(X.shape, 100, 100, 1, learning_rate=learning_rate / 100)
    dense_middle1 = FeedForwardLayer(100, 100, 100, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayer(100, 100, 100, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayer(100, 100, 100, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayer(100, 100, 1, 4, learning_rate=learning_rate, dropout=0)
    def trainint_loop(epoch):
        start_time = time.time()
        Y_pred_ss = smartselection1.forward(X)
        #print(Y_pred_ss)
        Y_pred_middle1 = dense_middle1.forward(Y_pred_ss)
        Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
        Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
        Y_pred_final = dense_output.forward(Y_pred_middle3, output=1)
        print(Y_pred_final)
        loss = dense_output.compute_loss(Y_pred_final, Y)
        accuracy = dense_output.compute_accuracy(Y_pred_final, Y)
        dA2 = dense_output.backward(Y_pred_middle3, Y_pred_final, Y_pred_final - Y, loss, output=1)
        dA2 = dense_middle3.backward(Y_pred_middle2, Y_pred_middle3, dA2, loss)
        dA2 = dense_middle2.backward(Y_pred_middle1, Y_pred_middle2, dA2, loss)
        dA2 = dense_middle1.backward(Y_pred_ss, Y_pred_middle1, dA2, loss)
        (_, _) = smartselection1.backward(X, Y_pred_ss, dA2, loss)
        print(f'Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Time: {time.time() - start_time}')
        smartselection1.save_model('./test_model')
        dense_middle1.save_model('./test_model')
        dense_middle2.save_model('./test_model')
        dense_middle3.save_model('./test_model')
        dense_output.save_model('./test_model')

    for epoch in range(1,100):
        trainint_loop(epoch)

def use_model():
    #X = np.load('/home/user/Desktop/datasets/checker_train_x_chaos.npy')
    X = np.load('/home/user/Desktop/datasets/gsm8k_train_chaos_x.npy')[:100]
    Y = np.load('/home/user/Desktop/datasets/checker_train_y.npy')
    input_shape = X.shape
    print(input_shape)
    learning_rate = 0.01
    smartselection1 = SmartSelectionLayer(input_shape, 100, 100, 1, learning_rate=learning_rate / 10)
    dense_middle1 = FeedForwardLayer(100, 100, 100, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayer(100, 100, 100, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayer(100, 100, 100, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayer(100, 100, 2, 4, learning_rate=learning_rate)
    smartselection1.load_model('./test_model')
    dense_middle1.load_model('./test_model')
    dense_middle2.load_model('./test_model')
    dense_middle3.load_model('./test_model')
    dense_output.load_model('./test_model')
    
    content = 'afejoakefjoiaejfioaejfopekaf\'eofjkaporekopt,epokfjopsejfopsjgpojskpofgksd'
    X = [[]]
    for i in range(len(content)):
        X[0].append([1, alphabet.find(content[i]) / (len(alphabet) + 1)])
    for i in range(input_shape[1] - len(content)):
        X[0].append([1, len(alphabet) / (len(alphabet) + 1)])
    for i in range(input_shape[1]):
        X[0][i][0] = i / input_shape[1]
    X = np.array(X, dtype='float32')
    print(X.shape)
    Y_pred_ss = smartselection1.forward(X)
    Y_pred_middle1 = dense_middle1.forward(Y_pred_ss)
    Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
    Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
    Y_pred_final = dense_output.forward(Y_pred_middle3, output=1)
    print(Y_pred_final)

train_model()
#use_model()
#train_model_keras()
#use_model_keras()