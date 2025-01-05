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

all_tokens = {}

data_train = []
mxlen = 0

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
    mxlen = max([mxlen, len(de) + 10, len(en) + 10])

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
            if i == 1000000:
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
        if check == 0:
            continue
        for j in de:
            all_tokens[j] = 1
        for j in en:
            all_tokens[j] = 1
        data_train.append([de, en])
        mxlen = max([mxlen, len(de) + 100, len(en) + 100])
    with open('./all_tokens.txt', 'w') as f:
        for i in all_tokens_list:
            f.write(i + '\n')
all_tokens_list = open('./all_tokens.txt', 'r').readlines()
all_tokens_list = [token[:-1] for token in all_tokens_list]
all_tokens_list = sorted(all_tokens_list)
#print(all_tokens_list)



for i in range(len(all_tokens_list)):
    all_tokens[all_tokens_list[i]] = i

def preprocess_translation_wmt():
    global mxlen

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
        self.Z2 = tf.linalg.matmul(self.A1, self.W2) + self.b2
        #self.learning_rate += random.uniform(-self.learning_rate / 9, self.learning_rate / 10)
        #self.learning_rate = max(self.learning_rate, 0.00001)
        dropout_mask = tf.cast(tf.random.uniform(self.Z2.shape) >= self.dropout, dtype='float32')
        self.mx = tf.math.reduce_max(abs(self.Z2)) + 1e-10
        if output:
            #return dropout_mask * tf.nn.softmax(self.Z2)
            return dropout_mask * self.Z2
        else:
            return dropout_mask * tf.nn.relu(self.Z2)

    def compute_loss(self, Y_pred, Y_true):
        #epsilon = 1e-8
        #loss = -tf.reduce_mean(tf.reduce_sum(Y_true * tf.math.log(Y_pred + epsilon), axis=-1))
        loss = tf.reduce_mean(tf.reduce_sum(tf.math.sqrt(abs(Y_true - Y_pred))))
        return loss

    def compute_accuracy(self, Y_pred, Y_true):
        #y_pred_classes = tf.argmax(Y_pred, axis=-1)
        #y_true_classes = tf.argmax(Y_true, axis=-1)
        y_pred_classes = tf.round(Y_pred * (len(all_tokens) + 1))
        y_true_classes = tf.round(Y_true * (len(all_tokens) + 1))
        correct_predictions = tf.equal(y_true_classes, y_pred_classes)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def backward(self, X, A2, dA2, output=0):
        m = dA2.shape[0] 
        if output:
            #dZ2 = dA2 * tf.nn.softmax(A2)
            dZ2 = dA2 #* tf.cast(A2 > 0, dtype='float32')
        else:
            dZ2 = dA2 * tf.cast(A2 > 0, dtype='float32')
        dW2 = tf.linalg.matmul(tf.transpose(self.A1), dZ2) / m
        db2 = tf.math.reduce_sum(dZ2, axis=0, keepdims=True) / m 

        dA1 = tf.linalg.matmul(dZ2, tf.transpose(self.W2))
        dZ1 = dA1 * tf.cast(self.A1 > 0, dtype='float32')
        dW1 = tf.linalg.matmul(tf.transpose(X), dZ1) / m 
        db1 = tf.math.reduce_sum(dZ1, axis=0, keepdims=True) / m 

        '''mnval = -1
        mxval = 1
        dW1 = tf.clip_by_value(dW1, mnval, mxval)
        db1 = tf.clip_by_value(db1, mnval, mxval)
        dW2 = tf.clip_by_value(dW2, mnval, mxval)
        db2 = tf.clip_by_value(db2, mnval, mxval)'''
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
        if self.X == None or new_X:
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

def train_model_words():
    samples_selected = 10000
    X = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy')[:samples_selected]
    print(len(X))
    '''for i in range(len(X)):
        res = ''
        for j in range(len(X[i])):
            index = round(X[i][j][1] * (len(all_tokens) + 1))
            if index < len(all_tokens):
                res += all_tokens_list[index]
        print(res)'''
    Y = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_y.npy')[:samples_selected]
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
    total_input = 13
    chaostranform1 = ChaosTransformLayer(X.shape, hidden_size, hidden_size * 10, 1, learning_rate=learning_rate / 100, total_input=total_input)
    hidden_size *= 10
    dense_middle1 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayer(hidden_size, hidden_size, hidden_size, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayer(hidden_size, hidden_size, mxlen, 4, learning_rate=learning_rate, dropout=0)
    filepath = './custom_models/test_model'
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
        dA2 = chaostranform1.backward(X, Y_pred_chaos, dA2, double_matrix=0)
        print(f'epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Time: {time.time() - start_time}')
        if epoch % 100 == 0 or loss < 40 or accuracy > 0.94:
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
        if loss < 40 or accuracy > 0.94:
            break
    plt.plot(loss_graph_x, loss_graph_y)
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
    filepath = './custom_models_2/test_model'
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
    samples_selected = 1
    X_full = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy')[:samples_selected]
    
    input_shape = X_full.shape
    print(input_shape)
    learning_rate = 0.01
    chaostranform1 = ChaosTransformLayer(input_shape, 100, 100, 1, learning_rate=learning_rate / 10, total_input=13)
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

content = 'person is great '
if 0:
    train_model_words()
elif 1:
    #use_model_words()
    use_model_seq2seq()