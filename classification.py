import torch
import torch.nn as nn
#import tensorflow as tf
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

torch.manual_seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def preprocess_glove_for_used():
    raw_glove = open('/home/user/Desktop/datasets/glove.6B.50d.txt', 'r').readlines()
    tokens_glove = {}

    for row in raw_glove:
        row = row[:-1]
        tokens_glove[row.split(' ')[0]] = torch.tensor(list(map(float, row.split(' ')[1:])), dtype=torch.float32)

    all_tokens = {}
    all_tokens_list = open('./all_tokens_imdb.txt', 'r').readlines()
    all_tokens_list = [token[:-1] for token in all_tokens_list]
    for token in all_tokens_list:
        if token in tokens_glove:
            all_tokens[token] = tokens_glove[token]

    np.save('/home/user/Desktop/datasets/glove_used_tokens.npy', all_tokens, allow_pickle=True)

#preprocess_glove_for_used()

special_chars = ['„', '”', "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", 
    ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    " ", "\t", "\n", "\r", "\f", "\v"
]

def tokenize_text(text):
    tokens = []
    current_token = ""
    for char in text:
        if char in special_chars:
            if current_token:
                tokens.append(current_token)
            tokens.append(char)
            current_token = ""
        else:
            current_token += char.lower()
    if current_token:
        tokens.append(current_token)
    return tokens

def preprocess_data_imdb(index):
    glove_words = np.load('/home/user/Desktop/datasets/glove_used_tokens.npy', allow_pickle=True)
    glove_dim = 50
    mxlen = int(open('./mxlen_imdb.txt', 'r').readlines()[0])
    data_raw = []
    data_train = []
    with open('/home/user/Desktop/datasets/IMDB Dataset.csv', "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for i, row in enumerate(reader):
            if len(row) == 2 and i >= 100 * (index - 1 - int(index == 6)) and i < 100 * index:
                data_raw.append(row)
    for row in data_raw:
        tokenized_full = tokenize_text(row[0])
        tokenized = []
        for token in tokenized_full:
            if token in glove_words:
                print(token)
                tokenized.append(glove_words[token])
        if len(tokenized) < 300:
            data_train.append([tokenized, int(row[1] == 'positive')])
    X = []
    Y = []
    for i in range(len(data_train)):
        X.append(data_train[i][0])
        Y.append([data_train[i][1]])
        nowlen = len(X[i])
        for j in range(mxlen - nowlen):
            X[i].append(tf.zeros((glove_dim, )))
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')
    #np.save(f'/home/user/Desktop/datasets/imdb_50k_x_{index - int(index == 6)}.npy', X)
    #np.save(f'/home/user/Desktop/datasets/imdb_50k_y_{index - int(index == 6)}.npy', Y)
    print(index)

def preprocess_training_batches(index):
    X = np.load(f'/home/user/Desktop/datasets/imdb_50k_x_{index}.npy', allow_pickle=True)
    print(len(X))
    Y = np.load(f'/home/user/Desktop/datasets/imdb_50k_y_{index}.npy', allow_pickle=True)
    batch_size = 300
    X = X.tolist()
    batches_x = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
    lastlen = len(batches_x[-1])
    for i in range(batch_size - lastlen):
        batches_x[-1].append(batches_x[-1][-1])
    for i in range(len(batches_x)):
        batches_x[i] = tf.convert_to_tensor(batches_x[i], dtype='float32')
    Y = Y.tolist()
    batches_y = [Y[i:i + batch_size] for i in range(0, len(X), batch_size)]
    lastlen = len(batches_y[-1])
    for i in range(batch_size - lastlen):
        batches_y[-1].append(batches_y[-1][-1])
    for i in range(len(batches_y)):
        batches_y[i] = tf.convert_to_tensor(batches_y[i], dtype='float32')
    for i in range(len(batches_x)):
        np.save(f'/home/user/Desktop/batches/training_batches_x_{136 + i}', batches_x[i])
        np.save(f'/home/user/Desktop/batches/training_batches_y_{136 + i}', batches_y[i])
    return len(batches_x)

def preprocess_training_representation():
    glove_words = np.load('/home/user/Desktop/datasets/glove_used_tokens.npy', allow_pickle=True).item()
    glove_dim = 50
    batch_size = 300
    i = 0
    for word1 in glove_words:
        X = []
        for word2 in glove_words:
            X.append(np.concatenate([glove_words[word1], glove_words[word2]], axis=0))
            X.append(np.concatenate([glove_words[word2], glove_words[word1]], axis=0))
        batches_x = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
        for j in range(len(batches_x)):
            np.save(f'/home/user/Desktop/batches/representation_batches_x_{i}.npy', batches_x[j])
            i += 1

mxlen = int(open('./mxlen_imdb.txt', 'r').readlines()[0])

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n):
        super(Model, self).__init__()
        self.n = n
        
        self.layers_dense = []
        self.layers_batch_norm = []
        self.layers_activation = []

        self.layers_dense = nn.ModuleList()
        self.layers_batch_norm = nn.ModuleList()
        self.layers_activation = nn.ModuleList()

        self.layers_dense.append(nn.Linear(input_size, hidden_size))
        self.layers_batch_norm.append(nn.BatchNorm1d(hidden_size))
        self.layers_activation.append(nn.Sigmoid())
        for i in range(1, n - 1):
            self.layers_dense.append(nn.Linear(hidden_size, hidden_size))
            self.layers_batch_norm.append(nn.BatchNorm1d(hidden_size))
            self.layers_activation.append(nn.ReLU())
        self.layers_dense.append(nn.Linear(hidden_size, output_size))
        self.layers_batch_norm.append(nn.BatchNorm1d(output_size))
        self.layers_activation.append(nn.Tanh())

    def forward(self, X):
        for i in range(self.n):
            X = self.layers_activation[i](self.layers_batch_norm[i](self.layers_dense[i](X)))
        return X.squeeze()
    
    def g(self, a, b):
        return (a + b) % 1

def print_memory_usage():
    process = psutil.Process()
    print(f"Memory Usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")

def train_imdb_representation():
    glove_dim = 50
    batches_selected = 10
    validation_size = 1000
    learning_rate = 0.001
    hidden_size = 100
    n = 4
    batch_size = 300

    eps = 1e-8
    model = Model(input_size=glove_dim, hidden_size=hidden_size, output_size=2, n=n)
    criterion = lambda y_pred, y_target: torch.sum((y_pred - y_target) ** 2) +\
                                         torch.sum((torch.sum(y_pred ** 2, dim=1) - 1) ** 2) +\
                                         torch.sum(1 / ((y_pred - torch.concat([torch.ones([y_pred.shape[0], 1]), torch.zeros([y_pred.shape[0], 1])], dim=1)) ** 2 + eps))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    batches = []
    for i in range(batches_selected):
        batches.append(torch.rand([batch_size, 2, glove_dim]))
    validation_x = torch.rand([validation_size, 2, glove_dim])

    def training_epoch(epoch):
        start_time = time.time()
        total_loss = 0

        for i in range(batches_selected):
            a = batches[i][:, :1, :].squeeze()
            b = batches[i][:, 1:, :].squeeze()
            f_a = model(a)
            f_b = model(b)
            f_ab = model(model.g(a, b))
            real_a = f_a[:, 0]
            real_b = f_b[:, 0]
            im_a = f_a[:, 1]
            im_b = f_b[:, 1]
            prod = torch.stack([real_a * real_b - im_a * im_b, real_a * im_b + real_b * im_a], dim=1)
            loss = criterion(prod, f_ab.squeeze())
            loss = torch.mean(loss)
            total_loss += loss / batches_selected
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if 1:
            print(f'Epoch: {epoch}, Loss: {total_loss}, Time: {time.time() - start_time}')
        if 1:
            a = validation_x[:, :1, :].squeeze()
            b = validation_x[:, 1:, :].squeeze()
            f_a = model(a)
            f_b = model(b)
            f_ab = model(model.g(a, b))
            real_a = f_a[:, 0]
            real_b = f_b[:, 0]
            im_a = f_a[:, 1]
            im_b = f_b[:, 1]
            prod = torch.stack([real_a * real_b - im_a * im_b, real_a * im_b + real_b * im_a], dim=1)
            loss = criterion(prod, f_ab.squeeze())
            loss = torch.mean(loss)
            print(f'    Epoch: {epoch}, Validation loss: {loss}')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss
            }, 'representation_model.pth')
        return loss

    for i in range(1, 10001):
        loss = training_epoch(i)
        print_memory_usage()
        if loss < 0.03:
            break

def use_imdb_representation():
    glove_dim = 50
    batches_selected = 10
    validation_size = 1000
    learning_rate = 0.001
    hidden_size = 100
    batch_size = 300

    model = Model(input_size=glove_dim, hidden_size=hidden_size, output_size=2, n=4)
    criterion = lambda y_pred, y_target: torch.abs(y_pred - y_target)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    checkpoint = torch.load('representation_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    validation_x = torch.rand([5, 2, glove_dim])
    a = validation_x[:, :1, :].squeeze()
    b = validation_x[:, 1:, :].squeeze()

    w = torch.ones([5, glove_dim])
    a = torch.transpose(a, 0, 1)
    b = torch.transpose(b, 0, 1)
    f_a = torch.exp(2j * torch.pi * torch.matmul(w, a))
    f_b = torch.exp(2j * torch.pi * torch.matmul(w, b))
    f_ab = torch.exp(2j * torch.pi * torch.matmul(w, model.g(a, b)))
    print(f_a, f_b, f_a * f_b - f_ab)
    #f_a = model(a)
    #f_b = model(b)
    #f_ab = model(model.g(a, b))
    #print(f_a * f_b - f_ab, f_a, f_b)

def train_glove_imdb():
    glove_words = np.load('/home/user/Desktop/datasets/glove_used_tokens.npy', allow_pickle=True).item()
    glove_dim = 50
    data_raw = []
    data_train = []
    with open('/home/user/Desktop/datasets/IMDB Dataset.csv', "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for i, row in enumerate(reader):
            if len(row) == 2 and i <= 10000:
                data_raw.append(row)
    for row in data_raw:
        tokenized_full = tokenize_text(row[0])
        tokenized = []
        for token in tokenized_full:
            if token in glove_words:
                tokenized.append(glove_words[token])
        data_train.append([tokenized, int(row[1] == 'positive')])
    xs = []
    ys = []
    for i in range(len(data_train)):
        res = torch.zeros([glove_dim])
        for j in range(len(data_train[i][0])):
            res = (res + data_train[i][0][j]) % 1
        w = torch.ones([glove_dim])
        f_a = torch.matmul(w, res) % 1
        xs.append(f_a)
        ys.append(data_train[i][1])
    hist_0 = []
    hist_1 = []
    for i in range(len(xs)):
        if ys[i] == 0:
            hist_0.append(xs[i])
        else:
            hist_1.append(xs[i])
    fig, axes = plt.subplots(1, 2, figsize=(9, 9))
    axes[0].hist(hist_0, bins=100)
    axes[0].set_title('zero class')
    axes[1].hist(hist_1, bins=100)
    axes[1].set_title('one class')
    plt.tight_layout()
    plt.show()

if 1:
    pass
    train_glove_imdb()
    #train_imdb_representation()
elif 1:
    use_imdb_representation()