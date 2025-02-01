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
tf.random.set_seed(42)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

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
    mxlen = max([mxlen, len(de) + len(en) + 2])

def data_preparation_common_seq2seq_check():
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
        mxlen = max([mxlen, len(de) + len(en) + 2])
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
            data_train[i][0][j] = [1, all_tokens[data_train[i][0][j]] / (len(all_tokens) + 1)]
    
    X = []
    Y = []
    for i in range(len(data_train)):
        for j in range(len(data_train[i][0])):
            X.append(data_train[i][1] + data_train[i][0][:j + 1])
            Y.append([1])
            for k in range(10):
                token = all_tokens_list[random.randint(0, len(all_tokens_list) - 1)]
                if token != data_train[i][0][j]:
                    X.append(data_train[i][1] + data_train[i][0][:j] + [[1, all_tokens[token] / (len(all_tokens) + 1)]])
                    Y.append([0])
    for i in range(len(X)):
        nowlen = len(X[i])
        for j in range(mxlen - nowlen):
            X[i].append([1, len(all_tokens) / (len(all_tokens) + 1)])
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')

    np.save('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy', X)
    np.save('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_y.npy', Y)

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
        mxlen = max([mxlen, len(de) + len(en) + 2])
    with open('./mxlen_common.txt', 'w') as f:
        f.write(str(mxlen))
    with open('./all_tokens_common.txt', 'w') as f:
        for i in all_tokens:
            f.write(i + '\n')

    all_tokens_list = open('./all_tokens_common.txt', 'r').readlines()
    all_tokens_list = [token[:-1] for token in all_tokens_list]
    all_tokens_list = sorted(all_tokens_list)
    for i in range(len(all_tokens_list)):
        all_tokens[all_tokens_list[i]] = i
    
    for i in range(len(data_train)):
        for j in range(len(data_train[i][1])):
            data_train[i][1][j] = [1, all_tokens[data_train[i][1][j]] / (len(all_tokens) + 1)]
        for j in range(len(data_train[i][0])):
            data_train[i][0][j] = [1, all_tokens[data_train[i][0][j]] / (len(all_tokens) + 1)]
    
    X = []
    Y = []
    for i in range(len(data_train)):
        for j in range(len(data_train[i][0])):
            X.append(data_train[i][1] + data_train[i][0][:j])
            Y.append([k == round(data_train[i][0][j][1] * (len(all_tokens) + 1)) for k in range(len(all_tokens) + 1)])
    for i in range(len(X)):
        nowlen = len(X[i])
        for j in range(mxlen - nowlen):
            X[i].append([1, len(all_tokens) / (len(all_tokens) + 1)])
        for j in range(mxlen):
            X[i][j][0] = j / mxlen
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
#print(all_tokens_list)

for i in range(len(all_tokens_list)):
    all_tokens[all_tokens_list[i]] = i

validation_x = []
validation_y = []
def custom_validation():
    global validation_x
    global validation_y
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
            data_validation[i][0][j] = [1, all_tokens[data_validation[i][0][j]] / (len(all_tokens) + 1)]

    for i in range(len(data_validation)):
        for j in range(len(data_validation[i][0])):
            validation_x.append(data_validation[i][1] + data_validation[i][0][:j])
            validation_y.append([k == round(data_validation[i][0][j][1] * (len(all_tokens) + 1)) for k in range(len(all_tokens) + 1)])
        for i in range(len(validation_x)):
            nowlen = len(validation_x[i])
            for j in range(mxlen - nowlen):
                validation_x[i].append([1, len(all_tokens) / (len(all_tokens) + 1)])
            for j in range(mxlen):
                validation_x[i][j][0] = j / mxlen
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

    def save(self, filepath):
        np.save(filepath + 'm.npy', self.m)
        np.save(filepath + 'v.npy', self.v)
        np.save(filepath + 't.npy', self.t)

    def load(self, filepath):
        self.m = np.load(filepath + 'm.npy')
        self.v = np.load(filepath + 'v.npy')
        self.t = np.load(filepath + 't.npy')
    
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
        self.running_var = Np.load(filepath + 'rv.npy')
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

    def compute_loss(self, Y_pred, Y_true):
        #loss = tf.reduce_sum(tf.math.sqrt(abs(Y_true - Y_pred))) / (Y_pred.shape[0] * Y_pred.shape[1])
        l2_reg_cost = 0*(self.l2_lambda / 2) * (tf.reduce_sum(tf.square(self.W1)) + tf.reduce_sum(tf.square(self.W2)))
        loss = keras.losses.categorical_crossentropy(Y_true, Y_pred)
        return tf.reduce_mean(loss)

    def compute_accuracy(self, Y_pred, Y_true):
        #y_pred_classes = tf.cast(tf.round(Y_pred), dtype='int32')
        #y_true_classes = tf.cast(tf.round(Y_true), dtype='int32')
        
        #y_pred_classes = tf.math.argmax(Y_pred, axis=-1)
        top_k = 3
        _, y_pred_classes = tf.math.top_k(Y_pred, top_k)
        y_true_classes = tf.cast(tf.math.argmax(Y_true, axis=-1), dtype='int32')
        y_true_classes = tf.broadcast_to(tf.reshape(y_true_classes, (y_true_classes.shape[0], 1)),
                                      (y_true_classes.shape[0], top_k))
        correct_predictions = tf.equal(y_true_classes, y_pred_classes)
        #threshold = 5
        #correct_predictions = abs(y_true_classes - y_pred_classes) < threshold
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32)) * top_k

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
        self.W1 = xavier_initialization((input_size, output_size))
        self.b1 = tf.zeros((1, output_size), dtype='float32')
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
        self.optimizer.initialize({
            'W1': self.W1,
            'b1': self.b1,
        })

    def forward(self, X, output=0, middle=0, training=1):
        if self.skipping:
            return X
        self.Z1 = tf.linalg.matmul(X, self.W1) + self.b1
        self.B1 = self.batch_norm.forward(self.Z1, training=training)
        if output:
            return tf.nn.softmax(self.B1)
        elif middle:
            return tf.nn.sigmoid(self.B1)
        else:
            return tf.nn.selu(self.B1)

    def compute_loss(self, Y_pred, Y_true):
        #loss = tf.reduce_sum(tf.math.sqrt(abs(Y_true - Y_pred))) / (Y_pred.shape[0] * Y_pred.shape[1])
        l2_reg_cost = 0*(self.l2_lambda / 2) * (tf.reduce_sum(tf.square(self.W1)) + tf.reduce_sum(tf.square(self.W2)))
        loss = keras.losses.categorical_crossentropy(Y_true, Y_pred)
        return tf.reduce_mean(loss)

    def compute_accuracy(self, Y_pred, Y_true):
        #y_pred_classes = tf.cast(tf.round(Y_pred), dtype='int32')
        #y_true_classes = tf.cast(tf.round(Y_true), dtype='int32')
        
        #y_pred_classes = tf.math.argmax(Y_pred, axis=-1)
        top_k = 3
        _, y_pred_classes = tf.math.top_k(Y_pred, top_k)
        y_true_classes = tf.cast(tf.math.argmax(Y_true, axis=-1), dtype='int32')
        y_true_classes = tf.broadcast_to(tf.reshape(y_true_classes, (y_true_classes.shape[0], 1)),
                                      (y_true_classes.shape[0], top_k))
        correct_predictions = tf.equal(y_true_classes, y_pred_classes)
        #threshold = 5
        #correct_predictions = abs(y_true_classes - y_pred_classes) < threshold
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32)) * top_k

    def backward(self, X, A1, dA1, output=0, middle=0):
        if self.skipping:
            return dA1
        m = dA1.shape[1] 
        if output:
            dZ1 = dA1
        elif middle:
            dZ1 = dA1 * tf.nn.sigmoid(A1) * (1 - tf.nn.sigmoid(A1))
        else:
            dZ1 = dA1 * (tf.cast(A1 > 0, dtype='float32') * self.selu_l + tf.cast(A1 < 0, dtype='float32') * self.selu_l * self.selu_a * tf.math.exp(A1))
        dB1 = self.batch_norm.backward(self.Z1, dZ1)
        dW1 = tf.linalg.matmul(tf.transpose(X), dB1) / m + (self.l2_lambda * self.W1) / m
        db1 = tf.math.reduce_sum(dZ1, axis=0, keepdims=True) / m 
        dA0 = tf.linalg.matmul(dZ1, tf.transpose(self.W1))

        mnval = -1
        mxval = 1
        dW1 = tf.clip_by_value(dW1, mnval, mxval)
        db1 = tf.clip_by_value(db1, mnval, mxval)
        if self.index == 'ssoriginp0_1':
            print(tf.reduce_sum(tf.cast(abs(dW1) > 1e-8, dtype='float32')) / (dW1.shape[0] * dW1.shape[1]))
        #if tf.reduce_max(dW1) > 0.1 or tf.reduce_max(dW1) < 0.00001:
        #    print(self.index, tf.reduce_max(dW1), tf.reduce_min(dW1))

        def optimize():
            return self.optimizer.update({
                'W1': self.W1,
                'b1': self.b1,
            }, {
                'W1': dW1,
                'b1': db1,
            })
        
        new_params = optimize()
        self.W1 = new_params['W1']
        self.b1 = new_params['b1']

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
    def __init__(self, input_shape, hidden_size, output_size, index, learning_rate=0.001, total_input=10, dropout=0):
        self.index = index
        self.total_input = total_input
        self.dense_input = []
        self.dense_middle = []
        for i in range(self.total_input):
            self.dense_input.append(FeedForwardLayer(input_size=input_shape[1] * 2, output_size=hidden_size, index='ssoriginp' + str(index) + '_' + str(i), learning_rate=learning_rate))
            self.dense_middle.append(FeedForwardLayer(input_size=hidden_size, output_size=hidden_size, index='ssorigmid' + str(index) + '_' + str(i), learning_rate=learning_rate, skipping=1))
        self.dense_combinator = FeedForwardLayer(input_size=self.total_input * hidden_size, output_size=output_size, index='sscomb' + str(index), learning_rate=learning_rate, dropout=dropout)
        #self.chaos_0 = np.random.rand(2, 2)
        #self.chaos_large = np.dot(np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype='int32'), 
        #                          np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]], dtype='int32'))
        self.chaos_large = np.array([[5, 2], [7, 3]], dtype='int32')
        #self.chaos_large = np.array([[1, 0], [0, 1]], dtype='float32')
        self.chaos_2_1 = np.array([[2, 1], [1, 1]], dtype='float32')
        #self.mat_1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='int32')
        self.mat_1 = np.array([[1, 0], [0, 1]], dtype='int32')
        self.chaos_1 = []
        self.chaos_1_inv = []
        self.mat_2 = np.array([[1, 0], [0, 1]], dtype='float32')
        self.chaos_2 = []
        for i in range(self.total_input):
            self.chaos_1.append(tf.convert_to_tensor(copy.deepcopy(self.mat_1), dtype='float32'))
            #print(tf.linalg.det(self.chaos_1[-1]))
            self.chaos_1_inv.append(tf.convert_to_tensor(copy.deepcopy(np.linalg.inv(self.mat_1)), dtype='float32'))
            #self.chaos.append(tf.random.uniform((2, 2)))
            self.mat_1 = np.dot(self.mat_1, self.chaos_large)
            self.chaos_2.append(tf.convert_to_tensor(copy.deepcopy(self.mat_2), dtype='float32'))
            self.mat_2 = np.dot(self.mat_2, self.chaos_2_1)
        print(self.mat_1, self.mat_2)
        self.X = None

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
        if not (skip_X is None):
            X_original += skip_X
        if self.X is None:
            self.X = self.forward_chaos(X_original)
        X = self.X
        if new_X:
            X = self.forward_chaos(X_original)
        Y_pred = []
        self.A2inp = []
        self.A2mid = []
        for i in range(self.total_input):
            self.A2inp.append(self.dense_input[i].forward(X[i], training=training))
            #self.A2inp[i] = self.attention(self.A2inp[i], self.A2inp[i], self.A2inp[i])
            #self.A2mid.append(self.backward_chaos(self.dense_middle[i].forward(self.A2inp[i]))[i])
            self.A2mid.append(self.dense_middle[i].forward(self.A2inp[i], training=training))
            Y_pred.append(self.A2mid[i])
        Y_pred_to_combinator = Y_pred[0]
        for i in range(1, self.total_input):
            Y_pred_to_combinator = tf.concat((Y_pred_to_combinator, Y_pred[i]), axis=-1)
        X_skip_to_combinator = tf.concat([X[i] for i in range(self.total_input)], axis=0)
        X_skip_to_combinator = tf.linalg.matmul(X_skip_to_combinator, tf.ones((X_skip_to_combinator.shape[1], Y_pred_to_combinator.shape[1]))) % 1
        Y_pred_final = self.dense_combinator.forward(Y_pred_to_combinator, output=output, middle=middle, training=training)
        return Y_pred_final, X_original
    
    def backward(self, X_original, A2, dA2, double_matrix=0, new_X=0, output=0, middle=0):
        if self.X is None:
            self.X = self.forward_chaos(X_original)
        X = self.X
        if new_X:
            X = self.forward_chaos(X_original)

        #X_with_y = self.forward_chaos(X_original + tf.pad(tf.reshape(Y, (Y.shape[0], Y.shape[1], 1)), [[0, 0], [0, 0], [2, 0]]))
        '''self.A2inp = []
        self.A2mid = []
        for i in range(self.total_input):
            self.A2inp.append(self.dense_input[i].forward(X[i]))
            #self.A2inp[i] = self.attention(self.A2inp[i], self.A2inp[i], self.A2inp[i])
            #self.A2mid.append(self.backward_chaos(self.dense_middle[i].forward(self.A2inp[i]))[i])
            self.A2mid.append(self.dense_middle[i].forward(self.A2inp[i]))
        '''
        self.A2 = self.A2mid[0]
        for i in range(1, self.total_input):
            self.A2 = tf.concat((self.A2, self.A2mid[i]), axis=1)
        A2 = self.dense_combinator.forward(self.A2, output=output, middle=middle)
        dA2_final = self.dense_combinator.backward(self.A2, A2, dA2, output=output, middle=middle)
        dA2 = self.dense_middle[0].backward(self.A2inp[0], self.A2mid[0], dA2_final[:, 0 * self.dense_input[0].W1.shape[1]:(0 + 1) * self.dense_input[0].W1.shape[1]])
        dA2 = self.dense_input[0].backward(X[0], self.A2inp[0], dA2)
        dA0 = (dA2 * X[0])[:, :mxlen] + (dA2 * X[0])[:, mxlen:]
        dX = tf.convert_to_tensor([self.forward_chaos(X[i], flattened=1)[i] for i in range(self.total_input)], dtype='float32')
        for i in range(1, self.total_input):
            dA2 = self.dense_middle[i].backward(self.A2inp[i], self.A2mid[i], dA2_final[:, i * self.dense_input[i].W1.shape[1]:(i + 1) * self.dense_input[i].W1.shape[1]])
            dA2 = self.dense_input[i].backward(X[i], self.A2inp[i], dA2)
            dA0 += (dA2 * dX[i])[:, :mxlen] + (dA2 * dX[i])[:, mxlen:]
        return dA0
    
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

def train_model_words_common():
    samples_selected = 1000
    X = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy')
    print(len(X))
    Y = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_y.npy')
    #print(Y)
    #for i in range(len(Y)):
    #    for j in range(len(Y[i])):
    #        Y[i][j] /= len(all_tokens) + 1
    print(X.shape, Y.shape, mxlen)
    learning_rate = 0.001
    hidden_size = 100
    total_input = 2
    chaostransform1 = ChaosTransformLayer(X.shape, hidden_size, X.shape[1], 1, learning_rate=learning_rate / 100, total_input=total_input)
    chaostransform2 = ChaosTransformLayer(X.shape, hidden_size, len(all_tokens) + 1, 2, learning_rate=learning_rate / 1, total_input=total_input)
    #hidden_size *= total_input
    #dense_middle1 = FeedForwardLayerDouble(hidden_size, hidden_size, hidden_size, 1, learning_rate=learning_rate)
    #dense_middle2 = FeedForwardLayerDouble(hidden_size, hidden_size, hidden_size, 2, learning_rate=learning_rate)
    #dense_middle3 = FeedForwardLayerDouble(hidden_size, hidden_size, hidden_size, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayerDouble(hidden_size, hidden_size, len(all_tokens) + 1, 4, learning_rate=learning_rate)
    filepath = './custom_models_2/test_model'
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
    def trainint_loop(epoch):
        start_time = time.time()
        Y_pred_chaos1 = chaostransform1.forward(X)
        Y_pred_chaos1_with_order = tf.concat((tf.broadcast_to(sequence_order_padding, (Y_pred_chaos1.shape[0], Y_pred_chaos1.shape[1], 1)),
                                              tf.reshape(Y_pred_chaos1, (Y_pred_chaos1.shape[0], Y_pred_chaos1.shape[1], 1))), axis=2)
        Y_pred_final = chaostransform2.forward(Y_pred_chaos1_with_order, new_X=1, output=1)
        #Y_pred_middle1 = dense_middle1.forward(Y_pred_chaos)
        #Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
        #Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
        #Y_pred_final = dense_output.forward(Y_pred_chaos2, output=1)
        loss = dense_output.compute_loss(Y_pred_final, Y)
        avg_loss[epoch % 100] = loss
        accuracy = dense_output.compute_accuracy(Y_pred_final, Y)
        avg_accuracy[epoch % 100] = accuracy
        dA2 = Y_pred_final - Y
        #dA2 = dense_output.backward(Y_pred_chaos2, Y_pred_final, dA2, output=1)
        #dA2 = dense_middle3.backward(Y_pred_middle2, Y_pred_middle3, dA2)
        #dA2 = dense_middle2.backward(Y_pred_middle1, Y_pred_middle2, dA2)
        #dA2 = dense_middle1.backward(Y_pred_chaos, Y_pred_middle1, dA2)
        dA2 = chaostransform2.backward(Y_pred_chaos1_with_order, Y_pred_final, dA2, new_X=1, output=1)
        dA2 = chaostransform1.backward(X, Y_pred_chaos1, dA2)
        val_accuracy = 0
        if not (loss < threshold_loss or accuracy > 0.98):
            print(f'epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Time: {time.time() - start_time}')
            print(f'average loss (per 100): {sum(avg_loss) / 100}, average accuracy: {sum(avg_accuracy) / 100}')
        if (epoch % 100 < 20 or epoch % 100 == 0 or loss < threshold_loss or accuracy > 0.98) and len(validation_x) > 0:
            Y_pred_chaos1 = chaostransform1.forward(validation_x, new_X=1)
            Y_pred_chaos1_with_order = tf.concat((tf.broadcast_to(sequence_order_padding, (Y_pred_chaos1.shape[0], Y_pred_chaos1.shape[1], 1)),
                                              tf.reshape(Y_pred_chaos1, (Y_pred_chaos1.shape[0], Y_pred_chaos1.shape[1], 1))), axis=2)
            Y_pred_final = chaostransform2.forward(Y_pred_chaos1_with_order, new_X=1, output=1)
            #Y_pred_middle1 = dense_middle1.forward(Y_pred_chaos)
            #Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
            #Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
            #Y_pred_final = dense_output.forward(Y_pred_chaos2, output=1)
            val_loss = dense_output.compute_loss(Y_pred_final, validation_y)
            val_accuracy = dense_output.compute_accuracy(Y_pred_final, validation_y)
            print(f'    epoch: {epoch}, validation loss: {val_loss}, validation accuracy: {val_accuracy}')
        if epoch % 100 == 0 or (loss < threshold_loss or accuracy > 0.98) and 0 or val_accuracy > 0.98:
            chaostransform1.save_model(filepath)
            chaostransform2.save_model(filepath)
            #dense_middle1.save_model(filepath)
            #dense_middle2.save_model(filepath)
            #dense_middle3.save_model(filepath)
            dense_output.save_model(filepath)
        loss_graph_x.append(epoch)
        loss_graph_y.append(loss)
        accuracy_graph_x.append(epoch)
        accuracy_graph_y.append(accuracy)
        return (loss, accuracy, val_accuracy)

    for epoch in range(1,30001):
        loss, accuracy, val_accuracy = trainint_loop(epoch)
        if (loss < threshold_loss or accuracy > 0.999) and 0 or val_accuracy > 0.98:
            break
    figure, axis = plt.subplots(1, 2)
    axis[0].plot(loss_graph_x, loss_graph_y)
    axis[0].set_title('loss')
    axis[1].plot(accuracy_graph_x, accuracy_graph_y)
    axis[1].set_title('accuracy')
    plt.show()

'''
samples: 10000
    learning_rate = 0.001
    hidden_size = 100
    total_input = 2
    batch_size = 256
    n = 2
    works 
'''


def train_model_words_wmt():
    validation_size = 500
    samples_selected = 10000 + validation_size
    X = np.load('/home/user/Desktop/datasets/translation_en_de_train_wmt_x.npy', allow_pickle=True)[:samples_selected]
    global mxlen
    for i in X:
        mxlen = max(mxlen, len(i) + 10)
    validation_x = X[-validation_size:]
    X = X[:-validation_size]
    #X = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy')
    print(len(X))
    #Y = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_y.npy')
    Y = np.load('/home/user/Desktop/datasets/translation_en_de_train_wmt_y.npy', allow_pickle=True)[:samples_selected]
    validation_y = Y[-validation_size:]
    Y = Y[:-validation_size]
    print(X.shape, Y.shape, mxlen)
    learning_rate = 0.001
    hidden_size = 100
    total_input = 2
    batch_size = 256
    n = 2
    middle = 1
    layers = []
    for i in range(n):
        layers.append(ChaosTransformLayer((0, mxlen), hidden_size, mxlen, i, learning_rate=learning_rate / 1, total_input=total_input))
    layers.append(ChaosTransformLayer((0, mxlen), hidden_size, len(all_tokens) + 1, n, learning_rate=learning_rate / 1, total_input=total_input))
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
    ''')
    X = X.tolist()
    batches_x = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
    lastlen = len(batches_x[-1])
    for i in range(batch_size - lastlen):
        batches_x[-1].append(batches_x[-1][-1])
    for i in range(len(batches_x)):
        batches_x[i] = np.array(batches_x[i], dtype=object)
    Y = Y.tolist()
    batches_y = [Y[i:i + batch_size] for i in range(0, len(X), batch_size)]
    lastlen = len(batches_y[-1])
    for i in range(batch_size - lastlen):
        batches_y[-1].append(batches_y[-1][-1])
    for i in range(len(batches_y)):
        batches_y[i] = np.array(batches_y[i], dtype=object)
    filepath = './custom_models/test_model'
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
        for i in range(len(batches_x)):
            X = batches_x[i]
            X = pad_arrays(X, sequence_mxlen_padding)
            Y = batches_y[i]
            Y = make_categorical(Y)
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
            dA2 = (Y_pred_final - Y) / batch_size
            dA2 = layers[-1].backward(Y_pred_chaos_with_order[-1], Y_pred_final, dA2, new_X=1, output=1)
            for j in range(len(Y_pred_chaos) - 1, 0, -1):
                dA2 = layers[j].backward(Y_pred_chaos_with_order[j - 1], Y_pred_chaos[j], dA2, new_X=1, middle=1)
            dA2 = layers[0].backward(X, Y_pred_chaos[0], dA2, new_X=1)
        avg_loss[epoch % 100] = loss
        avg_accuracy[epoch % 100] = accuracy
        val_accuracy = 0
        if not (loss < threshold_loss or accuracy > 0.98):
            print(f'epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Time: {time.time() - start_time}')
            print(f'average loss (per 100): {sum(avg_loss) / 100}, average accuracy: {sum(avg_accuracy) / 100}')
        if (epoch % 100 < 20 or epoch % 100 == 0 or loss < threshold_loss or accuracy > 0.98 or 1) and len(validation_x) > 0:
            X = validation_x
            X = pad_arrays(X, sequence_mxlen_padding)
            Y = validation_y
            Y = make_categorical(Y)
            Y_pred_chaos = [0]
            skip_X = [0]
            Y_pred_chaos_with_order = [X]
            Y_pred_chaos[0], skip_X[0] = layers[0].forward(X, new_X=1)
            for j in range(1, len(layers) - 1):
                Y_pred_chaos_with_order.append(tf.concat((tf.broadcast_to(sequence_order_padding, (Y_pred_chaos[-1].shape[0], Y_pred_chaos[-1].shape[1], 1)),
                                                      tf.reshape(Y_pred_chaos[-1], (Y_pred_chaos[-1].shape[0], Y_pred_chaos[-1].shape[1], 1))), axis=2))
                Y_pred_chaos.append(0)
                skip_X.append(0)
                Y_pred_chaos[-1], skip_X[-1] = layers[j].forward(Y_pred_chaos_with_order[-1], new_X=1, middle=1, training=0, skip_X=X)
            Y_pred_final, _ = layers[-1].forward(Y_pred_chaos_with_order[-1], new_X=1, output=1, training=0, skip_X=X)
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

    for epoch in range(1,30001):
        loss, accuracy, val_accuracy = trainint_loop(epoch)
        if (loss < threshold_loss or accuracy > 0.999) and 0 or val_accuracy > 0.98:
            break
    figure, axis = plt.subplots(1, 2)
    axis[0].plot(loss_graph_x, loss_graph_y)
    axis[0].set_title('loss')
    axis[1].plot(accuracy_graph_x, accuracy_graph_y)
    axis[1].set_title('accuracy')
    plt.show()

def train_model_simple_common():
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
    print(X.shape, Y.shape)
    learning_rate = 0.0001
    hidden_size = 100
    total_input = 2
    chaostranform1 = FeedForwardLayerDouble(X.shape[1], hidden_size * total_input, hidden_size * total_input, 0, learning_rate=learning_rate)
    hidden_size *= total_input
    dense_middle1 = FeedForwardLayerDouble(hidden_size, hidden_size, hidden_size, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayerDouble(hidden_size, hidden_size, hidden_size, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayerDouble(hidden_size, hidden_size, hidden_size, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayerDouble(hidden_size, hidden_size, len(all_tokens) + 1, 4, learning_rate=learning_rate)
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
        if not (loss < threshold_loss or accuracy > 0.98):
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
            print(f'   epoch: {epoch}, validation loss: {val_loss}, validation accuracy: {val_accuracy}')
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

def train_model_simple_wmt():
    validation_size = 500
    samples_selected = 10000 + validation_size
    X = np.load('/home/user/Desktop/datasets/translation_en_de_train_wmt_x.npy', allow_pickle=True)[:samples_selected]
    validation_x = X[-validation_size:]
    X = X[:-validation_size]
    #X = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy')
    print(len(X))
    #Y = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_y.npy')
    Y = np.load('/home/user/Desktop/datasets/translation_en_de_train_wmt_y.npy', allow_pickle=True)[:samples_selected]
    validation_y = Y[-validation_size:]
    Y = Y[:-validation_size]
    def convert_to_list(X):
        if isinstance(X, np.ndarray):
            return [convert_to_list(x) for x in X]
        return X
    
    X = convert_to_list(X)
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = X[i][j][1]
    X = np.array(X, dtype=object)
    print(X.shape, Y.shape)
    learning_rate = 0.0001
    hidden_size = 100
    total_input = 2
    batch_size = 1000
    chaostranform1 = FeedForwardLayerDouble(mxlen, hidden_size * total_input, mxlen, 0, learning_rate=learning_rate)
    hidden_size *= total_input
    dense_middle1 = FeedForwardLayerDouble(mxlen, hidden_size, mxlen, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayerDouble(mxlen, hidden_size, mxlen, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayerDouble(mxlen, hidden_size, mxlen, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayerDouble(mxlen, hidden_size, len(all_tokens) + 1, 4, learning_rate=learning_rate)
    X = X.tolist()
    batches_x = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
    lastlen = len(batches_x[-1])
    for i in range(batch_size - lastlen):
        batches_x[-1].append(batches_x[-1][-1])
    batches_x = np.array(batches_x, dtype=object)
    Y = Y.tolist()
    batches_y = [Y[i:i + batch_size] for i in range(0, len(X), batch_size)]
    lastlen = len(batches_y[-1])
    for i in range(batch_size - lastlen):
        batches_y[-1].append(batches_y[-1][-1])
    batches_y = np.array(batches_y, dtype=object)
    print(batches_x.shape, batches_y.shape, mxlen)
    filepath = './custom_models_2/test_model'
    loss_graph_x = []
    loss_graph_y = []
    accuracy_graph_x = []
    accuracy_graph_y = []
    avg_loss = [0 for i in range(100)]
    avg_accuracy = [0 for i in range(100)]
    threshold_loss = 0.03
    sequence_mxlen_padding = []
    for i in range(mxlen):
        sequence_mxlen_padding.append(len(all_tokens) / (len(all_tokens) + 1))
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

    validation_x = convert_to_list(validation_x)
    for i in range(len(validation_x)):
        for j in range(len(validation_x[i])):
            validation_x[i][j] = validation_x[i][j][1]
    validation_x = pad_arrays(validation_x, sequence_mxlen_padding)
    validation_y = make_categorical(validation_y)

    def trainint_loop(epoch):
        start_time = time.time()
        loss = 0
        accuracy = 0
        for i in range(len(batches_x)):
            X = batches_x[i]
            X = pad_arrays(X, sequence_mxlen_padding)
            Y = batches_y[i]
            Y = make_categorical(Y)
            Y_pred_chaos = chaostranform1.forward(X)
            Y_pred_middle1 = dense_middle1.forward(Y_pred_chaos)
            Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
            Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
            Y_pred_final = dense_output.forward(Y_pred_middle3, output=1)
            loss = (loss * i + dense_output.compute_loss(Y_pred_final, Y)) / (i + 1)
            accuracy = (accuracy * i + dense_output.compute_accuracy(Y_pred_final, Y)) / (i + 1)
            dA2 = dense_output.backward(Y_pred_middle3, Y_pred_final, (Y_pred_final - Y) / 1, output=1)
            dA2 = dense_middle3.backward(Y_pred_middle2, Y_pred_middle3, dA2)
            dA2 = dense_middle2.backward(Y_pred_middle1, Y_pred_middle2, dA2)
            dA2 = dense_middle1.backward(Y_pred_chaos, Y_pred_middle1, dA2)
            dA2 = chaostranform1.backward(X, Y_pred_chaos, dA2)
        validation_accuracy= 0
        avg_loss[epoch % 100] = loss
        avg_accuracy[epoch % 100] = accuracy
        print(f'Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Time: {time.time() - start_time}')
        print(f'avg loss: {sum(avg_loss) / 100}, avg accuracy: {sum(avg_accuracy) / 100}')
        if (epoch % 100 == 0 or (loss < 0.01 or accuracy > 0.9) or epoch % 500 < 20) and len(validation_x) > 0:
            Y_pred_chaos = chaostranform1.forward(validation_x, training=0)
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
        if loss < threshold_loss or accuracy > 0.98:
            break
    figure, axis = plt.subplots(1, 2)
    axis[0].plot(loss_graph_x, loss_graph_y)
    axis[0].set_title('loss')
    axis[1].plot(accuracy_graph_x, accuracy_graph_y)
    axis[1].set_title('accuracy')
    plt.show()

#chaotic: 3150
#simple: 19630

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
    dense_middle1 = FeedForwardLayerDouble(100, 100, 100, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayerDouble(100, 100, 100, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayerDouble(100, 100, 100, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayerDouble(100, 100, 2, 4, learning_rate=learning_rate)
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
    dense_middle1 = FeedForwardLayerDouble(100, 100, 100, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayerDouble(100, 100, 100, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayerDouble(100, 100, 100, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayerDouble(100, 100, 2, 4, learning_rate=learning_rate)
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
('be a new man', 'sei eine neuer mann'),
('man is a man', 'mann ist ein mann'),
('be a good day', 'sei ein guter tag'),
('a man', 'eine mann'),
('it is a good day', 'es ist ein guter tag'),
('it is a new day', 'es ist ein neuer tag'),
('you are a man', 'du bist ein mann'),
('you are a good man', 'du bist ein guter mann'),
('you are a great man', 'du bist ein großartiger mann'),
('you make a new man', 'du machst einen neuen mann')
]
evaluation_pairs = validation_sentences
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
    chaostransform1 = ChaosTransformLayer(input_shape, 100, 100, 1, learning_rate=learning_rate / 10, total_input=9)
    chaostransform2 = ChaosTransformLayer(input_shape, 100, 100, 2, learning_rate=learning_rate / 10, total_input=9)
    #chaostranform1 = FeedForwardLayerDouble(input_shape[1], 100, 100, 0, learning_rate=learning_rate)
    dense_middle1 = FeedForwardLayerDouble(100, 100, 100, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayerDouble(100, 100, 100, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayerDouble(100, 100, 100, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayerDouble(100, 100, 2, 4, learning_rate=learning_rate)
    filepath = './custom_models_2/test_model'
    #filepath = './custom_models/test_model'
    chaostransform1.load_model(filepath)
    chaostransform2.load_model(filepath)
    #dense_middle1.load_model(filepath)
    #dense_middle2.load_model(filepath)
    #dense_middle3.load_model(filepath)
    dense_output.load_model(filepath)
    global content
    global mxlen
    value = 0
    totalloss = 0
    sequence_order_padding = []
    for i in range(mxlen):
        sequence_order_padding.append([i / mxlen])
    sequence_order_padding = tf.convert_to_tensor(sequence_order_padding, dtype='float32')
    for n in range(len(evaluation_pairs)):
        content = evaluation_pairs[n][0]
        translation = evaluation_pairs[n][1]
        #print(mxlen, content)
        #mxlen *= 2
        X = []
        Y = []
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
        for j in range(len(de)):
            X.append([])
            for i in range(len(en)):
                X[j].append([1, all_tokens[en[i]] / (len(all_tokens) + 1)])
            for i in range(j):
                X[j].append([1, all_tokens[de[i]] / (len(all_tokens) + 1)])
            nowlen = len(X[j])
            for i in range(mxlen - nowlen):
                X[j].append([1, len(all_tokens) / (len(all_tokens) + 1)])
            Y.append([k == all_tokens[de[j]] for k in range(len(all_tokens) + 1)])
        for j in range(len(X)):
            for i in range(mxlen):
                X[j][i][0] = i / mxlen
        X = np.array(X, dtype='float32')
        Y = np.array(Y, dtype='float32')
        only = 0
        if len(X) == 1:
            only = 1
            X = np.concatenate((X, X), axis=0)
            Y = np.concatenate((Y, Y), axis=0)
        #print(X)
        #X = np.concatenate((X, X), axis=0)
        Y_pred_chaos1 = chaostransform1.forward(X, new_X=1)
        Y_pred_chaos1_with_order = tf.concat((tf.broadcast_to(sequence_order_padding, (Y_pred_chaos1.shape[0], Y_pred_chaos1.shape[1], 1)),
                                              tf.reshape(Y_pred_chaos1, (Y_pred_chaos1.shape[0], Y_pred_chaos1.shape[1], 1))), axis=2)
        Y_pred_final = chaostransform2.forward(Y_pred_chaos1_with_order, new_X=1, output=1)
        #Y_pred_middle1 = dense_middle1.forward(Y_pred_ss)
        #Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
        #Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
        #Y_pred_final = dense_output.forward(Y_pred_chaos2, output=1)
        totalloss += dense_output.compute_loss(Y_pred_final, Y)
        print(f'loss: {dense_output.compute_loss(Y_pred_final, Y)}, accuracy: {dense_output.compute_accuracy(Y_pred_final, Y)}')
        Y_pred_final = Y_pred_final.numpy()
        if only:
            Y_pred_final = Y_pred_final[:len(Y_pred_final) // 2]
        res = ''
        for pred in Y_pred_final:
            if tf.math.argmax(pred) < len(all_tokens):
                print(all_tokens_list[tf.math.argmax(pred)])
                for i in range(len(all_tokens_list)):
                    if pred[i] > 1e-5:
                        print(f'{pred[i]}: {all_tokens_list[i]}')
                res += all_tokens_list[tf.math.argmax(pred)]
        print('final:')
        print(content)
        print(res)
    print(value / len(evaluation_pairs))

def evaluate_model_simple():
    X_full = np.load('/home/user/Desktop/datasets/translation_en_de_anki_x.npy')[:samples_selected]
    Y = np.load('/home/user/Desktop/datasets/translation_en_de_anki_y.npy')[:samples_selected]
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            Y[i][j] /= len(all_tokens) + 1
    
    input_shape = X_full.shape
    #print(X_full)
    print(input_shape)
    learning_rate = 0.01
    #chaostransform1 = ChaosTransformLayer(input_shape, 100, 100, 1, learning_rate=learning_rate / 10, total_input=9)
    #chaostransform2 = ChaosTransformLayer(input_shape, 100, 100, 2, learning_rate=learning_rate / 10, total_input=9)
    chaostranform1 = FeedForwardLayerDouble(input_shape[1], 100, 100, 0, learning_rate=learning_rate)
    dense_middle1 = FeedForwardLayerDouble(100, 100, 100, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayerDouble(100, 100, 100, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayerDouble(100, 100, 100, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayerDouble(100, 100, 2, 4, learning_rate=learning_rate)
    #filepath = './custom_models_2/test_model'
    filepath = './custom_models/test_model'
    #chaostransform1.load_model(filepath)
    #chaostransform2.load_model(filepath)
    chaostranform1.load_model(filepath)
    dense_middle1.load_model(filepath)
    dense_middle2.load_model(filepath)
    dense_middle3.load_model(filepath)
    dense_output.load_model(filepath)
    global content
    global mxlen
    value = 0
    totalloss = 0
    sequence_order_padding = []
    for i in range(mxlen):
        sequence_order_padding.append([i / mxlen])
    sequence_order_padding = tf.convert_to_tensor(sequence_order_padding, dtype='float32')
    for n in range(len(evaluation_pairs)):
        content = evaluation_pairs[n][0]
        translation = evaluation_pairs[n][1]
        #print(mxlen, content)
        #mxlen *= 2
        X = []
        Y = []
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
        for j in range(len(de)):
            X.append([])
            for i in range(len(en)):
                X[j].append([1, all_tokens[en[i]] / (len(all_tokens) + 1)])
            for i in range(j):
                X[j].append([1, all_tokens[de[i]] / (len(all_tokens) + 1)])
            nowlen = len(X[j])
            for i in range(mxlen - nowlen):
                X[j].append([1, len(all_tokens) / (len(all_tokens) + 1)])
            Y.append([k == all_tokens[de[j]] for k in range(len(all_tokens) + 1)])
        for j in range(len(X)):
            for i in range(mxlen):
                X[j][i][0] = i / mxlen
        X = np.array(X, dtype='float32')
        Y = np.array(Y, dtype='float32')
        only = 0
        if len(X) == 1:
            only = 1
            X = np.concatenate((X, X), axis=0)
            Y = np.concatenate((Y, Y), axis=0)
        X = X.tolist()
        for i in range(len(X)):
            for j in range(len(X[i])):
                X[i][j] = X[i][j][1]
        X = np.array(X, dtype='float32')
        #print(X)
        #X = np.concatenate((X, X), axis=0)
        Y_pred_ss = chaostranform1.forward(X)
        Y_pred_middle1 = dense_middle1.forward(Y_pred_ss)
        Y_pred_middle2 = dense_middle2.forward(Y_pred_middle1)
        Y_pred_middle3 = dense_middle3.forward(Y_pred_middle2)
        Y_pred_final = dense_output.forward(Y_pred_middle3, output=1)
        totalloss += dense_output.compute_loss(Y_pred_final, Y)
        print(f'loss: {dense_output.compute_loss(Y_pred_final, Y)}, accuracy: {dense_output.compute_accuracy(Y_pred_final, Y)}')
        Y_pred_final = Y_pred_final.numpy()
        if only:
            Y_pred_final = Y_pred_final[:len(Y_pred_final) // 2]
        res = ''
        for pred in Y_pred_final:
            if tf.math.argmax(pred) < len(all_tokens):
                print(all_tokens_list[tf.math.argmax(pred)])
                for i in range(len(all_tokens_list)):
                    if pred[i] > 1e-5:
                        print(f'{pred[i]}: {all_tokens_list[i]}')
                res += all_tokens_list[tf.math.argmax(pred)]
        print('final:')
        print(content)
        print(res)
    print(value / len(evaluation_pairs))

def check_distribution():
    samples_selected = 1
    X_full = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_x.npy')
    Y = np.load('/home/user/Desktop/datasets/translation_en_de_train_common_seq2seq_y.npy')
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            Y[i][j] /= len(all_tokens) + 1
    
    input_shape = X_full.shape
    #print(X_full)
    print(input_shape)
    learning_rate = 0.01
    chaostranform1 = ChaosTransformLayer(input_shape, 100, 100, 1, learning_rate=learning_rate / 10, total_input=11)
    #chaostranform1 = FeedForwardLayerDouble(input_shape[1], 100, 100, 0, learning_rate=learning_rate)
    dense_middle1 = FeedForwardLayerDouble(100, 100, 100, 1, learning_rate=learning_rate)
    dense_middle2 = FeedForwardLayerDouble(100, 100, 100, 2, learning_rate=learning_rate)
    dense_middle3 = FeedForwardLayerDouble(100, 100, 100, 3, learning_rate=learning_rate)
    dense_output = FeedForwardLayerDouble(100, 100, 2, 4, learning_rate=learning_rate)
    filepath = './custom_models_2/test_model'
    #filepath = './custom_models/test_model'
    chaostranform1.load_model(filepath)
    dense_middle1.load_model(filepath)
    dense_middle2.load_model(filepath)
    dense_middle3.load_model(filepath)
    dense_output.load_model(filepath)
    Ys = []
    for n in range(len(validation_x)):
        X = []
        for i in range(mxlen):
            X.append([i / mxlen, random.randint(0, len(all_tokens)) / (len(all_tokens) + 1), 0])
        X = [X, X]
        #X = [X_full[n], X_full[n]]
        X = [validation_x[n], validation_x[n]]
        X = np.array(X, dtype='float32')
        Y_pred_ss = chaostranform1.forward(X, new_X=1)
        Y_pred_final = dense_output.forward(Y_pred_ss, output=1)
        for i in Y_pred_final[0]:
            y = round(float(i) * (len(all_tokens) + 1))
            if y < len(all_tokens):
                Ys.append(y)
            
    plt.hist(Ys, bins=len(all_tokens) + 1)
    plt.show()
        


if 1:
    #train_model_words_common()
    #train_model_simple_common()
    train_model_words_wmt()
    #train_model_simple_wmt()
elif 1:
    #use_model_words()
    #use_model_seq2seq()
    evaluate_model()
    #check_distribution()
elif 1:
    train_model_keras()
