import random
import json
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import sys
import PIL

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

class PredictorNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001, dropout=0):
        self.W1 = np.random.rand(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.rand(hidden_size, hidden_size) * 0.01
        self.b2 = np.zeros((1, hidden_size))
        self.W3 = np.random.rand(hidden_size, output_size) * 0.01
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
        self.A1 = np.array([list(map(lambda x: x * (x > 0), Z1)) for Z1 in self.Z1])
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = np.array([list(map(lambda x: x * (x > 0), Z2)) for Z2 in self.Z2])
        self.Z3 = np.dot(self.A2, self.W3) + self.b3 
        #self.learning_rate += random.uniform(-self.learning_rate / 9, self.learning_rate / 10)
        #self.learning_rate = max(self.learning_rate, 0.00001)
        if random.uniform(0,1) < self.dropout:
            return np.zeros(self.Z3.shape)
        return self.Z3

    def compute_loss(self, Y_pred, Y_true):
        return np.mean(abs(Y_pred - Y_true))

    def compute_accuracy(self, Y_pred, Y_true):
        return 1

    def backward(self, X, Y_pred, Y_true):
        m = Y_true.shape[0] 
        dZ3 = Y_pred - Y_true
        dW3 = np.dot(self.A2.T, dZ3) / m  
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = np.dot(dZ3, self.W3.T) 
        dZ2 = dA2 * np.array([list(map(lambda x: x > 0, Z2)) for Z2 in self.Z2])
        dW2 = np.dot(self.A1.T, dZ2) / m  
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m 

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * np.array([list(map(lambda x: x > 0, Z1)) for Z1 in self.Z1])
        dW1 = np.dot(X.T, dZ1) / m  
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m 

        clipping = 0

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
    output = open('./test_data.txt', 'w')
    res = ''
    for i in range(1000):
        X_sample = []
        for j in range(2):
            X_sample.append(random.uniform(0,10))
        Y_sample = X_sample[1]
        for j in range(2):
            res += str(X_sample[j]) + ' '
        res += str(Y_sample) + '\n'
    output.write(res)

def train_model():
    data = open('./test_data.txt').readlines()[:100]

    X = []
    Y = []
    for i in range(len(data)):
        data[i] = list(map(float, data[i].split(' ')))
        X.append(data[i][:2])
        Y.append([data[i][2]])
    X = X * 100
    Y = Y * 100
    X = np.array(X)
    Y = np.array(Y)

    global nn
    nn = PredictorNN(input_size=2, hidden_size=100, output_size=1, learning_rate=0.001, dropout=0)
    nn.train(X, Y, epochs=1000)

def predict(predictor):
    predictor = np.array(list(map(float, predictor.split(' '))))
    Y_pred = nn.forward([predictor])
    return Y_pred[0][0]

def use_model():
    global nn
    nn = PredictorNN(input_size=2, hidden_size=100, output_size=1, learning_rate=0.001, dropout=0)
    nn.W1 = np.load('./test_modelW1.npy')
    nn.b1 = np.load('./test_modelb1.npy')
    nn.W2 = np.load('./test_modelW2.npy')
    nn.b2 = np.load('./test_modelb2.npy')
    nn.W3 = np.load('./test_modelW3.npy')
    nn.b3 = np.load('./test_modelb3.npy')
    #print(nn.W1,nn.W2)
    print('ready')
    while True:
        predictor = input()
        print(predict(predictor))

#train_model()
#use_model()
#generate_samples()
        
from PIL import Image

def show_image_from_pixels(pixels, width, height):
    image = Image.new("RGB", (width, height))
    image.putdata(pixels)
    image.show()

def compare_with_rotated():
    im = Image.open('./test.jpg')
    pixels_original = list(im.getdata())
    width, height = im.size
    pixels_original = [pixels_original[i * width:(i + 1) * width] for i in range(height)]
    im = im.rotate(30)
    pixels_rotated = list(im.getdata())
    pixels_rotated = [pixels_rotated[i * width:(i + 1) * width] for i in range(height)]
    def f(x, y):
        return ((2 * x + y) % height, (x + y) % width)
    for i in range(20):
        new_pixels_original = pixels_original.copy()
        for x in range(height):
            for y in range(width):
                new_x, new_y = f(x, y)
                new_pixels_original[new_x][new_y] = pixels_original[x][y]
        pixels_original = new_pixels_original.copy()
        new_pixels_rotated = pixels_rotated.copy()
        for x in range(height):
            for y in range(width):
                new_x, new_y = f(x, y)
                new_pixels_rotated[new_x][new_y] = pixels_rotated[x][y]
        pixels_rotated = new_pixels_rotated.copy()
    pixels = []
    for i in range(height):
        for j in pixels_original[i]:
            pixels.append(j)
        pixels.append((0, 0, 0))
        for j in pixels_rotated[i]:
            pixels.append(j)
    show_image_from_pixels(pixels, 2 * width + 1, height)

compare_with_rotated()