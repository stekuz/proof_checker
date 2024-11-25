import random
import json
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import sys

sys.setrecursionlimit(100000)

alphabet = 'abcdefghijklmnopqrstuvwxyz'
alphabet += alphabet.upper()
alphabet += ' ,.\n;:-()'
alphabet += '1234567890'

def save_model(filepath):
    global nn
    np.save(filepath + 'W1.npy', nn.W1)
    np.save(filepath + 'b1.npy', nn.b1)
    np.save(filepath + 'W2.npy', nn.W2)
    np.save(filepath + 'b2.npy', nn.b2)
    np.save(filepath + 'W3.npy', nn.W3)
    np.save(filepath + 'b3.npy', nn.b3)

class AdamOptimizer:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def initialize(self, params):
        self.m = {key: np.zeros_like(value) for key, value in params.items()}
        self.v = {key: np.zeros_like(value) for key, value in params.items()}

    def update(self, params, grads):
        self.t += 1

        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

class FunctionNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.000001, leaky_a=0.1, dropout=0, selu_alpha=1.6733, selu_lambda=1.0507):
        self.W1 = np.random.rand(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.rand(hidden_size, hidden_size)
        self.b2 = np.zeros((1, hidden_size))
        self.W3 = np.random.rand(hidden_size, output_size)
        self.b3 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        self.leaky_a = leaky_a
        self.dropout = dropout
        self.selu_alpha = selu_alpha
        self.selu_lambda = selu_lambda
        self.optimizer = AdamOptimizer()
        self.optimizer.initialize({
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'W3': self.W3,
            'b3': self.b3,
        })

    def selu(self, x):
        if x >= 0:
            return self.selu_lambda * x
        else:
            return self.selu_alpha * self.selu_lambda * (math.e ** x - 1)

    def selu_d(self, x):
        if x >= 0:
            return self.selu_lambda
        else:
            return self.selu_alpha * self.selu_lambda * math.e ** x

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        for i in range(len(self.Z1)):
            sm = np.sum(self.Z1[i])
            if sm == 0:
                continue
            for j in range(len(alphabet)):
                self.Z1[i][j] /= sm
        self.A1 = []
        for i in range(len(self.Z1)):
            self.A1.append([])
            for j in range(len(self.Z1[i])):
                self.A1[i].append(self.selu(self.Z1[i][j]))
        self.A1 = np.array(self.A1)
        #self.A1 = np.tanh(self.Z1)
        #self.A1 = self.sigmoid(self.Z1) - 0.5
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        for i in range(len(self.Z2)):
            sm = np.sum(self.Z2[i])
            if sm == 0:
                continue
            for j in range(len(alphabet)):
                self.Z2[i][j] /= sm
        self.A2 = []
        for i in range(len(self.Z2)):
            self.A2.append([])
            for j in range(len(self.Z2[i])):
                self.A2[i].append(self.selu(self.Z2[i][j]))
        self.A2 = np.array(self.A2)
        #self.A2 = np.tanh(self.Z2)
        #self.A2 = self.sigmoid(self.Z2) - 0.5
        self.Z3 = np.dot(self.A2, self.W3) + self.b3 
        #self.leaky_a = random.uniform(max(-1, self.leaky_a - 0.1), min(0, self.leaky_a + 0.1))
        self.learning_rate += random.uniform(-self.learning_rate / 9, self.learning_rate / 10)
        self.learning_rate = max(self.learning_rate, 0.00001)
        for i in range(len(self.Z3)):
            sm = np.sum(self.Z3[i])
            if sm == 0:
                continue
            for j in range(len(alphabet)):
                self.Z3[i][j] /= sm
        if random.uniform(0,1) < self.dropout:
            return np.zeros(self.Z3.shape)
        return self.Z3

    def compute_loss(self, Y_pred, Y_true):
        return np.mean(abs(Y_pred - Y_true))
        #print(Y_pred)
        #print(Y_pred)
        #print(Y_true)

    def compute_accuracy(self, Y_pred, Y_true):
        sm = 0
        for i in range(len(Y_pred)):
            mx = 0
            mxi = 0
            for j in range(len(alphabet)):
                if mx < Y_pred[i][j]:
                    mxi = j
                    mx = Y_pred[i][j]
            if Y_true[i][mxi] > 0.9:
                sm += 1
        return sm / len(Y_pred)

    def backward(self, X, Y_true, Y_pred):
        m = Y_true.shape[0] 
        dZ3 = Y_pred - Y_true
        dW3 = np.dot(self.A2.T, dZ3) / m  
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = np.dot(dZ3, self.W3.T) 
        dZ2 = []
        for i in range(len(self.Z2)):
            dZ2.append([])
            for j in range(len(self.Z2[i])):
                dZ2[i].append(self.selu_d(self.Z2[i][j]))
        dZ2 = np.array(dZ2)
        dZ2 = dA2 * dZ2
        #dZ2 = dA2 * (1 - np.tanh(self.Z2) ** 2)
        #dZ2 = dA2 * self.sigmoid(self.Z2) * (1 - self.sigmoid(self.Z2))
        dW2 = np.dot(self.A1.T, dZ2) / m  
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m 

        dA1 = np.dot(dZ2, self.W2.T) 
        dZ1 = []
        for i in range(len(self.Z1)):
            dZ1.append([])
            for j in range(len(self.Z1[i])):
                dZ1[i].append(self.selu_d(self.Z1[i][j]))
        dZ1 = np.array(dZ1)
        dZ1 = dA1 * dZ1
        #dZ1 = dA1 * (1 - np.tanh(self.Z1) ** 2)
        #dZ1 = dA1 * self.sigmoid(self.Z1) * (1 - self.sigmoid(self.Z1))
        dW1 = np.dot(X.T, dZ1) / m  
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m 

        clipping = 1

        def clip(A, threshold, newval, less):
            for i in range(len(A)):
                for j in range(len(A[i])):
                    if A[i][j] < threshold and less:
                        A[i][j] = newval
                    if A[i][j] > threshold and less == 0:
                        A[i][j] = newval
        if clipping == 1:
            threshold = 2
            clip(dW1, threshold, threshold,  0)
            clip(dW2, threshold, threshold,  0)
            clip(dW3, threshold, threshold,  0)
            clip(db1, threshold, threshold,  0)
            clip(db2, threshold, threshold,  0)
            clip(db3, threshold, threshold,  0)
            clip(self.W1, threshold, threshold,  0)
            clip(self.W2, threshold, threshold,  0)
            clip(self.W3, threshold, threshold,  0)
            clip(self.b1, threshold, threshold,  0)
            clip(self.b2, threshold, threshold,  0)
            clip(self.b3, threshold, threshold,  0)

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3

        def optimize():
            self.optimizer.update({
                'W1': self.W1,
                'b1': self.b1,
                'W2': self.W2,
                'b2': self.b2,
                'W3': self.W3,
                'b3': self.b3,
            }, {
                'W1': dW1,
                'b1': db1,
                'W2': dW2,
                'b2': db2,
                'W3': dW3,
                'b3': db3,
            })

        optimize()

        #print('d', dW1)
        #print('w', self.W1)

    def train(self, X, Y, epochs):
        for epoch in range(epochs):
            Y_pred = self.forward(X.copy())
            loss = self.compute_loss(Y_pred.copy(), Y.copy())
            accuracy = self.compute_accuracy(Y_pred.copy(), Y.copy())
            self.backward(X, Y_pred, Y)
            if epoch:
                print(f'Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}')
                save_model('./test_model')

class AnosovApproximationNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001, dropout=0):
        self.W1 = np.random.rand(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.rand(hidden_size, hidden_size)
        self.b2 = np.zeros((1, hidden_size))
        self.W3 = np.random.rand(hidden_size, output_size)
        self.b3 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.optimizer = AdamOptimizer()
        self.optimizer.initialize({
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'W3': self.W3,
            'b3': self.b3,
        })

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = np.tanh(self.Z2)
        self.Z3 = np.dot(self.A2, self.W3) + self.b3 
        #self.learning_rate += random.uniform(-self.learning_rate / 9, self.learning_rate / 10)
        #self.learning_rate = max(self.learning_rate, 0.00001)
        if random.uniform(0,1) < self.dropout:
            return np.zeros(self.Z3.shape)
        return self.Z3

    def softmax(self, X):
        return np.exp(X) / np.sum(np.exp(X))

    def compute_loss(self, Y_pred, Y_true):
        Y_pred = self.softmax(Y_pred)
        loss = 0
        for i in range(len(Y_pred)):
            for j in range(len(Y_pred[i])):
                loss += -Y_true[i][j] * np.log(Y_pred[i][j])
        return loss / len(Y_pred)

    def compute_accuracy(self, Y_pred, Y_true):
        accuracy = 0
        for i in range(len(Y_pred)):
            mx = 0
            mxi = 0
            for j in range(len(Y_pred[i])):
                if mx < Y_pred[i][j]:
                    mx = Y_pred[i][j]
                    mxi = j
            accuracy += Y_true[i][mxi] > 0.9
        return accuracy / len(Y_pred)

    def backward(self, X, Y_pred, Y_true):
        m = Y_true.shape[0] 
        dZ3 = Y_pred - Y_true
        dW3 = np.dot(self.A2.T, dZ3) / m  
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = np.dot(dZ3, self.W3.T) 
        dZ2 = dA2 * (1 - np.tanh(self.Z2) ** 2)
        dW2 = np.dot(self.A1.T, dZ2) / m  
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m 

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (1 - np.tanh(self.Z1) ** 2)
        dW1 = np.dot(X.T, dZ1) / m  
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m 

        clipping = 1

        def clip(A, threshold, newval, less):
            for i in range(len(A)):
                for j in range(len(A[i])):
                    if A[i][j] < threshold and less:
                        A[i][j] = newval
                    if A[i][j] > threshold and less == 0:
                        A[i][j] = newval
        if clipping == 1:
            threshold = 2
            clip(dW1, threshold, threshold,  0)
            clip(dW2, threshold, threshold,  0)
            clip(dW3, threshold, threshold,  0)
            clip(db1, threshold, threshold,  0)
            clip(db2, threshold, threshold,  0)
            clip(db3, threshold, threshold,  0)
            clip(self.W1, threshold, threshold,  0)
            clip(self.W2, threshold, threshold,  0)
            clip(self.W3, threshold, threshold,  0)
            clip(self.b1, threshold, threshold,  0)
            clip(self.b2, threshold, threshold,  0)
            clip(self.b3, threshold, threshold,  0)

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3

        def optimize():
            self.optimizer.update({
                'W1': self.W1,
                'b1': self.b1,
                'W2': self.W2,
                'b2': self.b2,
                'W3': self.W3,
                'b3': self.b3,
            }, {
                'W1': dW1,
                'b1': db1,
                'W2': dW2,
                'b2': db2,
                'W3': dW3,
                'b3': db3,
            })

        optimize()
    
    def train(self, X, Y, epochs):
        for epoch in range(epochs):
            Y_pred = self.forward(X.copy())
            loss = self.compute_loss(Y_pred.copy(), Y.copy())
            accuracy = self.compute_accuracy(Y_pred.copy(), Y.copy())
            self.backward(X, Y_pred, Y)
            if epoch:
                print(f'Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}')
                save_model('./test_model')

def text_to_tokens(text):
    tokens=[]
    for i in text:
        tokens.append(alphabet.find(i))
    return tokens

def generate_samples():
    json_output = open('./test_data.json', 'w')
    data = {
        'data': []
    }
    for i in range(100):
        A1 = [[1, random.randint(1,100)], [0, 1]]
        B1 = [[1, 0], [random.randint(1,100), 1]]
        C1 = np.dot(A1, B1)
        A2 = [[1, random.randint(1,100)], [0, 1]]
        B2 = [[1, 0], [random.randint(1,100), 1]]
        C2 = np.dot(A2, B2)
        A3 = [[1, random.randint(1,100)], [0, 1]]
        B3 = [[1, 0], [random.randint(1,100), 1]]
        X = np.dot(A3, B3)
        Y = np.dot(np.dot(C1, X),np.linalg.inv(C2))
        for j in range(len(Y)):
            for k in range(len(Y[j])):
                Y[j][k] = round(Y[j][k])
        data['data'].append([X.tolist(), Y.tolist(), 1])
    json.dump(data, json_output)

def train_model():
    data = json.load(open('./test_data.json', 'r'))['data']
    X = []
    Y = []
    for sample in data:
        X_sample = []
        for k in range(2):
            for i in range(len(sample[k])):
                for j in range(len(sample[k][i])):
                    X_sample.append(sample[k][i][j])
        X.append(X_sample)
        Y.append([0, 1])
    for i in range(100):
        X_sample = []
        for j in range(8):
            X_sample.append(random.randint(1,10000))
        X.append(X_sample)
        Y.append([1, 0])
    X = X * 10
    Y = Y * 10
    X = np.array(X)
    Y = np.array(Y)

    global nn
    nn = MatrixConjugacyNN(input_size=8, hidden_size=300, output_size=2, learning_rate=0.005, dropout=0)
    nn.train(X, Y, epochs=1000)

def predict(predictor):
    predictor = np.array(list(map(int, predictor.split(' '))))
    Y_pred = nn.forward([predictor])
    print(Y_pred)
    if Y_pred[0][0] > Y_pred[0][1]:
        return 0
    else:
        return 1

def use_model():
    global nn
    nn = MatrixConjugacyNN(input_size=8, hidden_size=200, output_size=2, learning_rate=0.005, dropout=0)
    nn.W1 = np.load('./test_modelW1.npy')
    nn.b1 = np.load('./test_modelb1.npy')
    nn.W2 = np.load('./test_modelW2.npy')
    nn.b2 = np.load('./test_modelb2.npy')
    nn.W3 = np.load('./test_modelW3.npy')
    nn.b3 = np.load('./test_modelb3.npy')
    #print(nn.W1,nn.W2)
    print('ready')
    while True:
        predictor=input()
        predicted_class = predict(predictor)
        print(predicted_class)

train_model()
use_model()
#6777 77 88 1 225259 -2925055 220022 -2857051
#10 1 9 1 26 -23 17 -15
#generate_samples()