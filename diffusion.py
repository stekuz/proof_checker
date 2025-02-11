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
import psutil
import gc
import cv2

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

def permute_torus(X_original, n, matrix):
    rows = tf.reshape(tf.range(n, dtype='float32'), (n, 1))
    cols = tf.reshape(tf.range(n, dtype='float32'), (1, n))
    grid = tf.stack((rows * tf.ones_like(cols), cols * tf.ones_like(rows)), axis=-1)
    matrix = tf.expand_dims(tf.expand_dims(matrix, axis=0), axis=0)
    grid = tf.expand_dims(grid, axis=2)
    order = tf.squeeze(tf.cast(tf.round(tf.linalg.matmul(grid, matrix)), dtype='int32') % n, axis=2)
    result = tf.Variable(tf.zeros((n, n), dtype='float32'))
    return tf.tensor_scatter_nd_update(result, order, X_original)

def apply_box_blur(tensor):
    tensor = tf.cast(tensor, dtype=tf.float32)
    tensor = tf.expand_dims(tf.expand_dims(tensor, axis=0), axis=-1)
    kernel = tf.constant([[2, 2, 2], [2, 1, 2], [2, 2, 2]], dtype=tf.float32) / 9.0
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    padded_tensor = tf.pad(tensor, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    result = tf.nn.conv2d(padded_tensor, kernel, strides=[1, 1, 1, 1], padding='VALID')
    result = tf.squeeze(result, axis=[0, -1])
    return result

def save_grayscale_tensors_to_video(tensors, fps, output_path):
    tensors = [np.uint8(tensor * 255) for tensor in tensors]
    height, width = tensors[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    for frame in tensors:
        out.write(frame)
    out.release()
    print(f"Video saved to {output_path}")

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
        
        return dx, dgamma, dbeta
    
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
        self.optimizer = AdamOptimizer(learning_rate=learning_rate, epsilon=1e-8)
        #self.optimizer = SGD(learning_rate=learning_rate)
        self.optimizer.initialize({
            'W1': self.W1,
            'b1': self.b1,
            'gamma': self.batch_norm.gamma,
            'beta': self.batch_norm.beta
        })

    def forward(self, X, output=0, middle=0, training=1):
        if self.skipping:
            return X
        self.Z1 = tf.linalg.matmul(X, self.W1) + self.b1
        self.B1 = self.batch_norm.forward(self.Z1, training=training)
        if output:
            return tf.nn.sigmoid(self.B1)
        elif middle:
            return tf.nn.sigmoid(self.B1)
        else:
            return tf.nn.selu(self.B1)
            #return tf.nn.relu(self.B1)

    def compute_loss(self, Y_pred, Y_true):
        loss = keras.losses.mae(Y_true, Y_pred)
        return tf.reduce_mean(loss)

    def compute_accuracy(self, Y_pred, Y_true):
        y_pred_classes = tf.cast(tf.round(Y_pred), dtype='int32')
        y_true_classes = tf.cast(tf.round(Y_true), dtype='int32')
        return tf.reduce_mean(tf.cast(tf.equal(y_pred_classes, y_true_classes), dtype='float32'))

    def backward(self, X, A1, dA1, output=0, middle=0):
        if self.skipping:
            return dA1
        m = dA1.shape[1] 
        if output:
            dZ1 = dA1 * tf.nn.sigmoid(A1) * (1 - tf.nn.sigmoid(A1))
        elif middle:
            dZ1 = dA1 * tf.nn.sigmoid(A1) * (1 - tf.nn.sigmoid(A1))
        else:
            dZ1 = dA1 * (tf.cast(A1 > 0, dtype='float32') * self.selu_l +
                         tf.cast(A1 < 0, dtype='float32') * self.selu_l * self.selu_a * tf.math.exp(A1 * tf.cast(A1 < 0, dtype='float32')))
            #dZ1 = dA1 * tf.cast(A1 > 0, dtype='float32')
        dB1, dgamma, dbeta = self.batch_norm.backward(self.Z1, dZ1)
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
                'gamma': self.batch_norm.gamma,
                'beta': self.batch_norm.beta
            }, {
                'W1': dW1,
                'b1': db1,
                'gamma': dgamma,
                'beta': dbeta
            })
        
        new_params = optimize()
        self.W1 = new_params['W1']
        self.b1 = new_params['b1']
        self.batch_norm.gamma = new_params['gamma']
        self.batch_norm.beta = new_params['beta']

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

def train_model_mnist():
    data = np.load('/home/user/Desktop/datasets/mnist_raw.npy', allow_pickle=True)
    X = np.concatenate((data['x_train'], data['x_test'])) / 256
    Y = np.concatenate((data['y_train'], data['y_test']))
    def print_memory_usage():
        process = psutil.Process()
        print(f"Memory Usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    print_memory_usage()
    learning_rate = 0.001
    hidden_size = 100
    batch_size = 300
    n = 5
    image_size = 28
    batches_selected = 10
    transform_matrix = tf.convert_to_tensor([[10, 9], [1, 1]], dtype='float32')
    transform_matrix = tf.linalg.matmul(transform_matrix, transform_matrix)
    batches_x = []
    batches_y = []
    batches_x = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
    for i in range(batches_selected):
        batches_x[i] = tf.convert_to_tensor(batches_x[i], dtype='float32')
        X = batches_x[i]
        D = tf.reshape(X - tf.map_fn(lambda x: permute_torus(x, image_size, transform_matrix), X), (batch_size, -1,))
        X = tf.reshape(X, (batch_size, -1,))
        batches_x[i] = X
        batches_y.append(D)
    #batches_y = [Y[i:i + batch_size] for i in range(0, len(X), batch_size)]
    #for i in range(len(batches_y)):
    #    batches_y[i] = tf.convert_to_tensor(batches_y[i], dtype='float32')
    filepath = './custom_models/'
    mxlen = image_size * image_size
    layers = []
    layers.append(FeedForwardLayer(mxlen, hidden_size, 0, learning_rate=learning_rate))
    for i in range(n - 2):
        layers.append(FeedForwardLayer(hidden_size, hidden_size, i + 1, learning_rate=learning_rate))
    layers.append(FeedForwardLayer(hidden_size, mxlen, n, learning_rate=learning_rate))
    print(len(batches_x))
    print(f'''
    learning_rate = {learning_rate}
    hidden_size = {hidden_size}
    batch_size = {batch_size}
    n = {n}
    training_samples = {batches_selected * batch_size}
    ''')
    threshold_loss = 0.1
    def training_loop(epoch):
        start_time = time.time()
        loss = 0
        accuracy = 0
        for i in range(batches_selected):
            X = batches_x[i]
            D = batches_y[i]
            Y_pred = [X]
            for j in range(n):
                Y_pred.append(0)
                Y_pred[-1] = layers[j].forward(Y_pred[-2], output=int(j == n - 1))
            loss = (loss * i + layers[0].compute_loss(Y_pred[-1], D)) / (i + 1)
            accuracy = (accuracy * i + layers[0].compute_accuracy(Y_pred[-1], D)) / (i + 1)
            dA2 = Y_pred[-1] - D
            for j in range(n - 1, -1, -1):
                dA2 = layers[j].backward(Y_pred[j], Y_pred[j + 1], dA2, output=int(j == n - 1))
        if 1:
            print(f'epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Time: {time.time() - start_time}')
        for i in range(len(layers)):
            layers[i].save_model(filepath)
        return loss, accuracy, 0
    for epoch in range(1,30001):
        print_memory_usage()
        loss, accuracy, val_accuracy = training_loop(epoch)
        np.empty((0,))
        gc.collect()
        if loss < threshold_loss:
            break
    
def use_model_mnist():
    learning_rate = 0.001
    hidden_size = 100
    n = 5
    image_size = 28
    filepath = './custom_models/'
    mxlen = image_size * image_size
    layers = []
    layers.append(FeedForwardLayer(mxlen, hidden_size, 0, learning_rate=learning_rate))
    for i in range(n - 2):
        layers.append(FeedForwardLayer(hidden_size, hidden_size, i + 1, learning_rate=learning_rate))
    layers.append(FeedForwardLayer(hidden_size, mxlen, n, learning_rate=learning_rate))
    for i in range(n):
        layers[i].load_model(filepath)
    data = np.load('/home/user/Desktop/datasets/mnist_raw.npy', allow_pickle=True)
    #X = tf.convert_to_tensor(data['x_train'][50000] / 256, dtype='float32')
    test_image = plt.imread('./test.png')
    X = tf.convert_to_tensor(test_image[:, :, 0], dtype='float32')
    transform_matrix = tf.convert_to_tensor([[10, 9], [1, 1]], dtype='float32')
    transform_matrix = tf.linalg.matmul(transform_matrix, transform_matrix)
    transform_matrix = tf.linalg.inv(transform_matrix)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(X, cmap='gray', vmin=0, vmax=255 / 256)
    axes[0].set_title("Original")
    axes[0].axis('off')
    results = []
    for k in range(60):
        Y_pred = [tf.reshape(X, (1, mxlen))]
        for i in range(n):
            Y_pred.append(0)
            Y_pred[-1] = layers[i].forward(Y_pred[-2], output=int(i == n - 1), training=0)
        D = tf.reshape(Y_pred[-1], (image_size, image_size))
        D += tf.random.normal((image_size, image_size), mean=0.0, stddev=0.1)
        result = tf.cast(apply_box_blur(permute_torus(X - D, image_size, transform_matrix)) > 0.8, dtype='float32')
        #result = X - D
        X = result
        results.append(result)
    save_grayscale_tensors_to_video(results, 10, './test.mp4')
    axes[1].imshow(results[0], cmap='gray', vmin=0, vmax=255 / 256)
    axes[1].set_title("New")
    axes[1].axis('off')
    plt.tight_layout()
    #plt.show()

def train_model_mnist_permutation():
    data = np.load('/home/user/Desktop/datasets/mnist_raw.npy', allow_pickle=True)
    X = np.concatenate((data['x_train'], data['x_test'])) / 256
    Y = np.concatenate((data['y_train'], data['y_test']))
    print(Y[0])
    def print_memory_usage():
        process = psutil.Process()
        print(f"Memory Usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    print_memory_usage()
    learning_rate = 0.001
    hidden_size = 100
    batch_size = 300
    n = 5
    image_size = 28
    batches_selected = 10
    validation_size = 3
    transform_matrix = tf.convert_to_tensor([[10, 9], [1, 1]], dtype='float32')
    transform_matrix = tf.linalg.matmul(transform_matrix, transform_matrix)
    batches_x = []
    batches_y = []
    batches_y = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
    batches_x = [Y[i:i + batch_size] for i in range(0, len(Y), batch_size)]
    for i in range(batches_selected + validation_size):
        batches_y[i] = tf.reshape(tf.cast(tf.convert_to_tensor(batches_y[i], dtype='float32') > 0.7, dtype='float32'), (batch_size, -1))
        batches_x[i] = tf.reshape(tf.convert_to_tensor([tf.reshape(tf.cast(batches_x[i], dtype='float32'), (batch_size,)), tf.random.uniform((batch_size,))], dtype='float32'),
                                  (batch_size, 2))
    validation_x = batches_x[batches_selected:batches_selected + validation_size]
    validation_y = batches_y[batches_selected:batches_selected + validation_size]
    filepath = './custom_models/'
    mxlen = image_size * image_size
    layers = []
    layers.append(FeedForwardLayer(2, hidden_size, 0, learning_rate=learning_rate))
    for i in range(n - 2):
        layers.append(FeedForwardLayer(hidden_size, hidden_size, i + 1, learning_rate=learning_rate))
    layers.append(FeedForwardLayer(hidden_size, mxlen, n, learning_rate=learning_rate))
    print(len(batches_x))
    print(f'''
    learning_rate = {learning_rate}
    hidden_size = {hidden_size}
    batch_size = {batch_size}
    n = {n}
    training_samples = {batches_selected * batch_size}
    validation_samples = {validation_size * batch_size}
    ''')
    threshold_loss = 0.1
    def training_loop(epoch):
        start_time = time.time()
        loss = 0
        accuracy = 0
        for i in range(batches_selected):
            X = batches_x[i]
            Y = batches_y[i]
            Y_pred = [X]
            for j in range(n):
                Y_pred.append(0)
                Y_pred[-1] = layers[j].forward(Y_pred[-2], output=int(j == n - 1))
            loss = (loss * i + layers[0].compute_loss(Y_pred[-1], Y)) / (i + 1)
            accuracy = (accuracy * i + layers[0].compute_accuracy(Y_pred[-1], Y)) / (i + 1)
            dA2 = Y_pred[-1] - Y
            for j in range(n - 1, -1, -1):
                dA2 = layers[j].backward(Y_pred[j], Y_pred[j + 1], dA2, output=int(j == n - 1))
        if 1:
            print(f'epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Time: {time.time() - start_time}')
        val_loss = 0
        val_accuracy = 0
        for i in range(len(validation_x)):
            X = validation_x[i]
            Y = validation_y[i]
            Y_pred = [X]
            for j in range(n):
                Y_pred.append(0)
                Y_pred[-1] = layers[j].forward(Y_pred[-2], output=int(j == n - 1))
            val_loss = (val_loss * i + layers[0].compute_loss(Y_pred[-1], Y)) / (i + 1)
            val_accuracy = (val_accuracy * i + layers[0].compute_accuracy(Y_pred[-1], Y)) / (i + 1)
        if 1:
            print(f'    Validation loss: {val_loss}, Validation accuracy: {val_accuracy}')
        for i in range(len(layers)):
            layers[i].save_model(filepath)
        return loss, accuracy, 0
    for epoch in range(1,30001):
        print_memory_usage()
        loss, accuracy, val_accuracy = training_loop(epoch)
        np.empty((0,))
        gc.collect()
        if loss < threshold_loss:
            break

def use_model_mnist_permutation():
    data = np.load('/home/user/Desktop/datasets/mnist_raw.npy', allow_pickle=True)
    X = np.concatenate((data['x_train'], data['x_test'])) / 256
    Y = np.concatenate((data['y_train'], data['y_test']))
    print(Y[0])
    def print_memory_usage():
        process = psutil.Process()
        print(f"Memory Usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    print_memory_usage()
    learning_rate = 0.001
    hidden_size = 100
    batch_size = 300
    n = 5
    image_size = 28
    batches_selected = 10
    validation_size = 3
    filepath = './custom_models/'
    mxlen = image_size * image_size
    layers = []
    layers.append(FeedForwardLayer(2, hidden_size, 0, learning_rate=learning_rate))
    for i in range(n - 2):
        layers.append(FeedForwardLayer(hidden_size, hidden_size, i + 1, learning_rate=learning_rate))
    layers.append(FeedForwardLayer(hidden_size, mxlen, n, learning_rate=learning_rate))
    for i in range(n):
        layers[i].load_model(filepath)
    print('ready')
    while(True):
        label = int(input())
        if label == -1:
            break
        X = tf.reshape(tf.convert_to_tensor([tf.reshape(tf.cast(label, dtype='float32'), (1, 1)), tf.random.uniform((1,1))], dtype='float32'), (1, 2))
        Y_pred = [X]
        for j in range(n):
            Y_pred.append(0)
            Y_pred[-1] = layers[j].forward(Y_pred[-2], output=int(j == n - 1), training=0)
        result = tf.reshape(Y_pred[-1], (image_size, image_size))
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(result, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title("Original")
        axes[0].axis('off')
        plt.tight_layout()
        plt.show()

if 0:
    #train_model_mnist()
    train_model_mnist_permutation()
elif 1:
    #use_model_mnist()
    use_model_mnist_permutation()