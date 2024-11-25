import random
import json
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle

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

def g(x):
    if abs(x) < 0.0000001 or abs(x) > 1 or np.isnan(x):
        return 0
    return round(1 / abs(x)) % len(alphabet)

def h(token):
    if token == 0:
        return 0
    return 1 / (token - 0.001)

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
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, leaky_a=0.1, dropout=0, selu_alpha=1.6733, selu_lambda=1.0507):
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

    def activation(self, x):
        if x >= 0:
            return self.selu_lambda * x
        else:
            return self.selu_alpha * self.selu_lambda * (math.e ** x - 1)

    def activation_d(self, x):
        if x >= 0:
            return self.selu_lambda
        else:
            return self.selu_alpha * self.selu_lambda * math.e ** x

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        '''self.A1 = []
        for i in range(len(self.Z1)):
            self.A1.append([])
            for j in range(len(self.Z1[i])):
                self.A1[i].append(self.selu(self.Z1[i][j]))'''
        self.A1 = np.tanh(self.Z1)
        #self.A1 = self.sigmoid(self.Z1) - 0.5
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        '''self.A2 = []
        for i in range(len(self.Z2)):
            self.A2.append([])
            for j in range(len(self.Z2[i])):
                self.A2[i].append(self.selu(self.Z2[i][j]))'''
        self.A2 = np.tanh(self.Z2)
        #self.A2 = self.sigmoid(self.Z2) - 0.5
        self.Z3 = np.dot(self.A2, self.W3) + self.b3 
        #self.leaky_a = random.uniform(max(-1, self.leaky_a - 0.1), min(0, self.leaky_a + 0.1))
        #self.learning_rate += random.uniform(-self.learning_rate / 2, self.learning_rate / 10)
        if random.uniform(0,1) < self.dropout:
            return np.zeros(self.Z3.shape)
        return self.Z3

    def compute_loss(self, Y_pred, Y_true):
        #return np.mean((Y_pred - Y_true) ** 2)
        #print(Y_pred)
        Y_pred = [g(x[0] % 1) for x in Y_pred]
        Y_true = [g(x[0] % 1) for x in Y_true]
        #print(Y_pred)
        #print(Y_true)
        sm = 0
        for i in range(len(Y_pred)):
            sm += abs(Y_pred[i] - Y_true[i])
        return sm / len(Y_pred)

    def compute_accuracy(self, Y_pred, Y_true):
        Y_pred = [g(x[0] % 1) for x in Y_pred]
        Y_true = [g(x[0] % 1) for x in Y_true]
        sm = 0
        for i in range(len(Y_pred)):
            sm += abs(Y_pred[i] - Y_true[i]) < 3
        return sm / len(Y_pred)

    def backward(self, X, Y_true, Y_pred):
        m = Y_true.shape[0] 
        dZ3 = Y_pred - Y_true
        dW3 = np.dot(self.A2.T, dZ3) / m  
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = np.dot(dZ3, self.W3.T) 
        '''dZ2 = []
        for i in range(len(self.Z2)):
            dZ2.append([])
            for j in range(len(self.Z2[i])):
                dZ2[i].append(self.selu_d(self.Z2[i][j]))
        dZ2 = np.array(dZ2)
        dZ2 = dA2 * dZ2'''
        dZ2 = dA2 * (1 - np.tanh(self.Z2) ** 2)
        #dZ2 = dA2 * self.sigmoid(self.Z2) * (1 - self.sigmoid(self.Z2))
        dW2 = np.dot(self.A1.T, dZ2) / m  
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m 

        dA1 = np.dot(dZ2, self.W2.T) 
        '''dZ1 = []
        for i in range(len(self.Z1)):
            dZ1.append([])
            for j in range(len(self.Z1[i])):
                dZ1[i].append(self.selu_d(self.Z1[i][j]))
        dZ1 = np.array(dZ1)
        dZ1 = dA1 * dZ1'''
        dZ1 = dA1 * (1 - np.tanh(self.Z1) ** 2)
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

        #optimize()

        #print('d', dW1)
        #print('w', self.W1)

        


    def train(self, X, Y, epochs):
        for epoch in range(epochs):
            Y_pred = self.forward(X.copy())
            loss = self.compute_loss(Y_pred.copy(), Y.copy())
            accuracy = self.compute_accuracy(Y_pred.copy(), Y.copy())
            self.backward(X, Y, Y_pred)
            if epoch:
                print(f'Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}')
                save_model('./test_model')
                if accuracy > 0.8:
                    break

def text_to_tokens(text):
    tokens=[]
    for i in text:
        tokens.append(alphabet.find(i))
    return tokens

def train_model():
    test_string = ''.join(open('./test_data.txt').readlines())[:3]
    print(test_string)
    
    X = []
    Y = []
    f = 1
    for start_i in range(len(test_string)):
        f = 0
        for i in range(start_i, len(test_string) - 1):
            x = h(alphabet.find(test_string[i]))
            X.append([x, f])
            f = h(alphabet.find(test_string[i+1]) + (i - start_i + 1) * len(alphabet)) * (i - start_i + 1)
            Y.append([f + (i - start_i + 1)])
    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.plot([x[0] for x in X], [x[1] for x in X], [y[0] for y in Y], 'ro', markersize=0.2)
    #plt.show()
    print(X)
    print(Y)
    X = X * 50
    Y = Y * 50
    X = np.array(X)
    Y = np.array(Y)
    np.random.shuffle(X)
    np.random.shuffle(Y)

    global nn
    nn = FunctionNN(input_size=2, hidden_size=100, output_size=1)
    nn.train(X, Y, epochs=200)

    test_input = np.array([[0.5, 0.5]])
    prediction = nn.forward(test_input)
    print(f'Prediction for input test_input: {prediction}')
    
    return

def predict(predictor):
    f = 0
    s = predictor[0]
    for i in range(1,len(predictor)+1):
        f = nn.forward(np.array([[h(predictor[i-1]), f]]).astype(dtype='float64'))[0][0]
        s = g(f % 1)
    return s

def use_model():
    global nn
    nn = FunctionNN(input_size=2, hidden_size=200, output_size=1, dropout=0)
    nn.W1 = np.load('./test_modelW1.npy')
    nn.b1 = np.load('./test_modelb1.npy')
    nn.W2 = np.load('./test_modelW2.npy')
    nn.b2 = np.load('./test_modelb2.npy')
    nn.W3 = np.load('./test_modelW3.npy')
    nn.b3 = np.load('./test_modelb3.npy')
    print('ready')
    while True:
        predictor=text_to_tokens(input())
        res=''
        for i in range(1):
            prediction=predict(predictor)
            res+=alphabet[prediction]
            predictor.append(prediction)
        print(res)

train_model()
use_model()
