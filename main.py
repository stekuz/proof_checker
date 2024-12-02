import random
import json
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import sys
import PIL
import time
sys.setrecursionlimit(100000)
from PIL import Image

def show_image_from_pixels(pixels, width, height):
    image = Image.new('RGB', (width, height))
    image.putdata(pixels)
    image.show()

def get_shuffled_image(pixels_original, width, height, flatten=0, flattened=0):
    if flattened:
        pixels_original = [pixels_original[i * width:(i + 1) * width] for i in range(height)]
    def f(x, y):
        return ((10 * x + y) % height, (9 * x + y) % width)
    for i in range(10):
        new_pixels_original = pixels_original.copy()
        for x in range(height):
            for y in range(width):
                new_x, new_y = f(x, y)
                new_pixels_original[new_x][new_y] = pixels_original[x][y]
        pixels_original = new_pixels_original.copy()
    new_pixels_original = pixels_original.copy()
    d = 3
    avg_original = [0, 0, 0]
    for x in range(height):
        for y in range(width):
            for i in range(3):
                avg_original[i] = 0
            for xx in range(x - d, x + d + 1):
                for yy in range(y - d, y + d + 1):
                    for i in range(3):
                        avg_original[i] += pixels_original[xx % height][yy % width][i]
            for i in range(3):
                avg_original[i] //= (d + d + 1) * (d + d + 1)
            new_pixels_original[x][y] = avg_original.copy()
    pixels_original = new_pixels_original.copy()
    if flatten == 0:
        return pixels_original
    else:
        pixels_flattened = []
        for i in range(height):
            for j in pixels_original[i]:
                pixels_flattened.append(j)
        return pixels_flattened

def compare_with_rotated(num):
    im = Image.open('./test.jpg')
    print(np.array(im.getdata()).shape)
    pixels_original = list(im.getdata())
    width, height = im.size
    pixels_original = [pixels_original[i * width:(i + 1) * width] for i in range(height)]
    im = im.rotate(90)
    pixels_rotated = list(im.getdata())
    pixels_rotated = [pixels_rotated[i * width:(i + 1) * width] for i in range(height)]
    def f(x, y):
        return ((10 * x + y) % height, (9 * x + y) % width)
    for i in range(num):
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
    new_pixels_original = pixels_original.copy()
    new_pixels_rotated = pixels_rotated.copy()
    for i in range(height):
        for j in range(width):
            pixels_original[i][j] = list(pixels_original[i][j])
            pixels_rotated[i][j] = list(pixels_rotated[i][j])
    d = 3
    for x in range(height):
        for y in range(width):
            avg_original = [0, 0, 0]
            avg_rotated = [0, 0, 0]
            for xx in range(x - d, x + d + 1):
                for yy in range(y - d, y + d + 1):
                    for i in range(3):
                        avg_original[i] += pixels_original[xx % height][yy % width][i]
                        avg_rotated[i] += pixels_rotated[xx % height][yy % width][i]
            for i in range(3):
                avg_original[i] //= (d + d + 1) * (d + d + 1)
                avg_rotated[i] //= (d + d + 1) * (d + d + 1)
            new_pixels_original[x][y] = tuple(avg_original)
            new_pixels_rotated[x][y] = tuple(avg_rotated)
    pixels_original = new_pixels_original.copy()
    pixels_rotated = new_pixels_rotated.copy()
    for i in range(height):
        for j in pixels_original[i]:
            pixels.append(j)
        pixels.append((0, 0, 0))
        for j in pixels_rotated[i]:
            pixels.append(j)
    show_image_from_pixels(pixels, 2 * width + 1, height)

import tensorflow as tf
import keras
from keras import layers

def train_model_mnist_full():
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
    X_train = np.array(tf.image.grayscale_to_rgb(tf.expand_dims(X_train, axis=3)))
    X_test = np.array(tf.image.grayscale_to_rgb(tf.expand_dims(X_test, axis=3)))
    for i in range(len(X_train)):
        X_train[i] = np.array(get_shuffled_image(list(X_train[i]), 28, 28))
        print(i)
    np.save('./mnist_train.npy', X_train)
    for i in range(len(X_test)):
        X_test[i] = get_shuffled_image(list(X_test[i]), 28, 28)
        print(i, 2)
    np.save('./mnist_test.npy', X_test)
    X_train = np.load('./mnist_train.npy')
    X_test = np.load('./mnist_test.npy')
    X_train = X_train.astype(dtype='float64') / 255
    X_test = X_test.astype(dtype='float64') / 255
    num_classes = 10
    input_shape = (28, 28, 3)
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    Y_test = keras.utils.to_categorical(Y_test, num_classes)
    model = keras.Sequential(
        [   
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(5, 5), activation='relu', name='firstconv'),
            layers.AveragePooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='secondconv'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='sigmoid'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax'),
        ]
    )
    #kernel = (2 / 9) * np.ones((3, 3, 3, 32))
    kernel0 = [[0, 0, 0, 0, 0], [0, -2, -2, -2, 0], [0, -2, 16, -2, 0], [0, -2, -2, -2, 0], [0, 0, 0, 0, 0]]
    kernel = []
    for i in range(32):
        kernel.append([])
        for j in range(3):
            kernel[i].append(kernel0.copy())
    kernel = np.array(kernel)
    kernel = kernel.reshape((5, 5, 3, 32))
    bias = np.zeros((32, ))
    model.get_layer('firstconv').set_weights([kernel, bias])
    model.summary()
    batch_size = 64
    epochs = 20
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    cp_callback = keras.callbacks.ModelCheckpoint(filepath='./models_trained/model_mnist_chaos_4.ckpt.keras')
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[cp_callback])
    
def use_model_mnist_full():
    model = keras.models.load_model('./models_trained/model_mnist_chaos_4.ckpt.keras', compile=True)
    print(model.summary())
    (_, Y_train), (_, Y_test) = tf.keras.datasets.mnist.load_data()
    num_classes = 10
    Y_test = keras.utils.to_categorical(Y_test, num_classes)
    X_test = np.load('./mnist_test.npy')
    X_test = X_test.astype(dtype='float64') / 255
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(score)
    while True:
        print('ready')
        prediction = int(input())
        #X_train = np.load('./mnist_train.npy')
        im = Image.open('./test.jpg')
        pixels_original = list(im.getdata())
        width, height = im.size
        X_train = np.array([get_shuffled_image(pixels_original, width, height, flattened=1)])
        X_train = X_train.astype(dtype='float64') / 255
        #(_, Y_train), (_, Y_test) = tf.keras.datasets.mnist.load_data()
        num_classes = 10
        Y_test = np.array([prediction])
        print(Y_test.shape)
        Y_test = keras.utils.to_categorical(Y_test, num_classes)
        print(model.predict(X_train))
        score = model.evaluate(X_train, Y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

#train_model_mnist_full()
#use_model_mnist_full()
def compare():
    while True:
        print('ready')
        num = int(input())
        compare_with_rotated(num)

def train_model_mnist_trunc():
    (X_train, Y_train_full), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
    X_train_full = np.load('./mnist_train.npy')
    X_test_full = np.load('./mnist_test.npy')
    X_train = []
    Y_train = []
    num_classes = 2
    for i in range(len(Y_train_full)):
        if Y_train_full[i] == 2 or Y_train_full[i] == 3:
            X_train.append(X_train_full[i].copy())
            Y_train.append(Y_train_full[i] - 2)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_train = X_train.astype(dtype='float64') / 255
    X_test = X_test.astype(dtype='float64') / 255
    input_shape = (28, 28, 3)
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    #Y_test = keras.utils.to_categorical(Y_test, num_classes)
    model = keras.Sequential(
        [   
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', name='firstconv'),
            layers.AveragePooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='secondconv'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='sigmoid'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax'),
        ]
    )
    #kernel = (2 / 9) * np.ones((3, 3, 3, 32))
    kernel0 = [[-2, -2, -2], [-2, 16, -2], [-2, -2, -2]]
    kernel = []
    for i in range(32):
        kernel.append([])
        for j in range(3):
            kernel[i].append(kernel0.copy())
    kernel = np.array(kernel)
    kernel = kernel.reshape((3, 3, 3, 32))
    bias = np.zeros((32, ))
    model.get_layer('firstconv').set_weights([kernel, bias])
    model.summary()
    batch_size = 64
    epochs = 10
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    cp_callback = keras.callbacks.ModelCheckpoint(filepath='./models_trained/model_mnist_chaos_3.ckpt.keras')
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[cp_callback])

def use_model_mnist_trunc():
    model = keras.models.load_model('./models_trained/model_mnist_chaos_3.ckpt.keras', compile=True)
    print(model.summary())
    (_, Y_train), (_, Y_test) = tf.keras.datasets.mnist.load_data()
    num_classes = 2
    while True:
        print('ready')
        prediction = int(input()) - 2
        #X_train = np.load('./mnist_train.npy')
        im = Image.open('./test.jpg')
        pixels_original = list(im.getdata())
        width, height = im.size
        X_train = np.array([get_shuffled_image(pixels_original, width, height, flattened=1)])
        X_train = X_train.astype(dtype='float64') / 255
        #(_, Y_train), (_, Y_test) = tf.keras.datasets.mnist.load_data()
        Y_test = np.array([prediction])
        print(Y_test.shape)
        Y_test = keras.utils.to_categorical(Y_test, num_classes)
        print(model.predict(X_train))
        score = model.evaluate(X_train, Y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

#train_model_mnist_trunc()
#use_model_mnist_trunc()

def preprocess_dataset():
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train = np.array(tf.image.grayscale_to_rgb(tf.expand_dims(X_train, axis=3)))
    X_test = np.array(tf.image.grayscale_to_rgb(tf.expand_dims(X_test, axis=3)))
    for i in range(len(X_train)):
        X_train[i] = np.array(get_shuffled_image(list(X_train[i]), 28, 28))
        print(i)
    np.save('./fashion_mnist_train.npy', X_train)
    for i in range(len(X_test)):
        X_test[i] = get_shuffled_image(list(X_test[i]), 28, 28)
        print(i, 2)
    np.save('./fashion_mnist_test.npy', X_test)
    return

def train_model_fashion_mnist():
    (X_train_full, Y_train_full), (X_test, Y_test) = tf.keras.datasets.fashion_mnist.load_data()
    #X_train_full = np.load('./fashion_mnist_train.npy')
    X_test_full = np.load('./fashion_mnist_test.npy')
    X_train = []
    Y_train = []
    for i in range(len(Y_train_full)):
        if Y_train_full[i] < 2 or True:
            X_train.append(X_train_full[i].copy())
            Y_train.append(Y_train_full[i])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_train = X_train.astype(dtype='float64') / 255
    X_test = X_test.astype(dtype='float64') / 255
    input_shape = (28, 28, 3)
    num_classes = 10
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    model = keras.Sequential(
        [   
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', name='firstconv'),
            layers.AveragePooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='secondconv'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='sigmoid'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax'),
        ]
    )
    #kernel = (2 / 9) * np.ones((3, 3, 3, 32))
    kernel0 = [[-2, -2, -2], [-2, 16, -2], [-2, -2, -2]]
    kernel = []
    for i in range(32):
        kernel.append([])
        for j in range(3):
            kernel[i].append(kernel0.copy())
    kernel = np.array(kernel)
    kernel = kernel.reshape((3, 3, 3, 32))
    bias = np.zeros((32, ))
    model.get_layer('firstconv').set_weights([kernel, bias])
    model.summary()
    batch_size = 64
    epochs = 20
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    cp_callback = keras.callbacks.ModelCheckpoint(filepath='./models_trained/model_fashion_mnist_chaos_2.ckpt.keras')
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[cp_callback])

def use_model_fashion_mnist():
    model = keras.models.load_model('./models_trained/model_fashion_mnist_chaos.ckpt.keras', compile=True)
    print(model.summary())
    num_classes = 10
    while True:
        print('ready')
        prediction = int(input())
        #X_train = np.load('./mnist_train.npy')
        im = Image.open('./test.jpg')
        pixels_original = list(im.getdata())
        width, height = im.size
        X_train = np.array([get_shuffled_image(pixels_original, width, height, flattened=1)])
        X_train = X_train.astype(dtype='float64') / 255
        #(_, Y_train), (_, Y_test) = tf.keras.datasets.mnist.load_data()
        Y_test = np.array([prediction])
        print(Y_test.shape)
        Y_test = keras.utils.to_categorical(Y_test, num_classes)
        print(model.predict(X_train))
        score = model.evaluate(X_train, Y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

#train_model_fashion_mnist()
#use_model_fashion_mnist()

def train_model_fashion_mnist_ordinary():
    (X_train_full, Y_train_full), (X_test, Y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train_full = np.array(tf.image.grayscale_to_rgb(tf.expand_dims(X_train_full, axis=3)))
    #X_train_full = np.load('./fashion_mnist_train.npy')
    #X_test_full = np.load('./fashion_mnist_test.npy')
    X_train = []
    Y_train = []
    for i in range(len(Y_train_full)):
        if Y_train_full[i] < 2 or True:
            X_train.append(X_train_full[i].copy())
            Y_train.append(Y_train_full[i])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_train = X_train.astype(dtype='float64') / 255
    X_test = X_test.astype(dtype='float64') / 255
    input_shape = (28, 28, 3)
    num_classes = 10
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    model = keras.Sequential(
        [   
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', name='firstconv'),
            layers.AveragePooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='secondconv'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='sigmoid'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax'),
        ]
    )
    #kernel = (2 / 9) * np.ones((3, 3, 3, 32))
    kernel0 = [[-2, -2, -2], [-2, 16, -2], [-2, -2, -2]]
    kernel = []
    for i in range(32):
        kernel.append([])
        for j in range(3):
            kernel[i].append(kernel0.copy())
    kernel = np.array(kernel)
    kernel = kernel.reshape((3, 3, 3, 32))
    bias = np.zeros((32, ))
    model.get_layer('firstconv').set_weights([kernel, bias])
    model.summary()
    batch_size = 64
    epochs = 20
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    cp_callback = keras.callbacks.ModelCheckpoint(filepath='./models_trained/model_fashion_mnist_ordinary.ckpt.keras')
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[cp_callback])

def use_model_fashion_mnist_ordinary():
    model = keras.models.load_model('./models_trained/model_fashion_mnist_ordinary.ckpt.keras', compile=True)
    print(model.summary())
    num_classes = 10
    while True:
        print('ready')
        prediction = int(input())
        #X_train = np.load('./mnist_train.npy')
        im = Image.open('./test.jpg')
        pixels_original = list(im.getdata())
        width, height = im.size
        X_train = np.array([get_shuffled_image(pixels_original, width, height, flattened=1)])
        X_train = X_train.astype(dtype='float64') / 255
        #(_, Y_train), (_, Y_test) = tf.keras.datasets.mnist.load_data()
        Y_test = np.array([prediction])
        print(Y_test.shape)
        Y_test = keras.utils.to_categorical(Y_test, num_classes)
        print(model.predict(X_train))
        score = model.evaluate(X_train, Y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

#train_model_fashion_mnist_ordinary()
#use_model_fashion_mnist_ordinary()

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

def predict(predictor):
    predictor = np.array(list(map(float, predictor.split(' '))))
    Y_pred = nn.forward([predictor])
    return Y_pred[0][0]

#compare()

def train_model_mnist_attention():
    (X_train_full, Y_train_full), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
    X_train_full = np.load('./shuffled_datasets/mnist_train.npy')
    #X_test_full = np.load('./shuffled_datasets/mnist_test.npy')
    X_train = []
    Y_train = []
    for i in range(len(Y_train_full)):
        if True:
            X_train.append(X_train_full[i].copy())
            Y_train.append(Y_train_full[i])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_train = X_train.astype(dtype='float64') / 255
    X_test = X_test.astype(dtype='float64') / 255
    input_shape = (28, 28, 3)
    num_classes = 10
    batch_size = 64
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    inputs = layers.Input(shape=input_shape)
    conv2d1 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', name='firstconv')(inputs)
    avgpool2d = layers.AveragePooling2D(pool_size=(2, 2))(conv2d1)
    conv2d2 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='secondconv')(avgpool2d)
    maxpool2d = layers.MaxPooling2D(pool_size=(2, 2))(conv2d2)
    flatten1 = layers.Flatten()(maxpool2d)
    dense1 = layers.Dense(128, activation='relu')(flatten1)
    dense1_expanded = layers.Reshape((128, 1))(dense1)
    attention = layers.Attention()([dense1_expanded, dense1_expanded])
    dropout = layers.Dropout(0.5)(attention)
    flatten2 = layers.Flatten()(dropout)
    outputs = layers.Dense(num_classes, activation='softmax')(flatten2)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    #kernel = (2 / 9) * np.ones((3, 3, 3, 32))
    kernel0 = [[-2, -2, -2], [-2, 16, -2], [-2, -2, -2]]
    kernel = []
    for i in range(32):
        kernel.append([])
        for j in range(3):
            kernel[i].append(kernel0.copy())
    kernel = np.array(kernel)
    kernel = kernel.reshape((3, 3, 3, 32))
    bias = np.zeros((32, ))
    model.get_layer('firstconv').set_weights([kernel, bias])
    model.summary()
    epochs = 40
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    cp_callback = keras.callbacks.ModelCheckpoint(filepath='./models_trained/model_mnist_chaos_attention.ckpt.keras')
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[cp_callback])

def use_model_mnist_attention():
    model = keras.models.load_model('./models_trained/model_mnist_chaos_attention.ckpt.keras', compile=True)
    print(model.summary())
    num_classes = 10
    while True:
        print('ready')
        prediction = int(input())
        #X_train = np.load('./mnist_train.npy')
        im = Image.open('./test.jpg')
        pixels_original = list(im.getdata())
        width, height = im.size
        X_train = np.array([get_shuffled_image(pixels_original, width, height, flattened=1)])
        X_train = X_train.astype(dtype='float64') / 255
        #(_, Y_train), (_, Y_test) = tf.keras.datasets.mnist.load_data()
        Y_test = np.array([prediction])
        print(Y_test.shape)
        Y_test = keras.utils.to_categorical(Y_test, num_classes)
        print(model.predict(X_train))
        score = model.evaluate(X_train, Y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

#train_model_mnist_attention()
use_model_mnist_attention()