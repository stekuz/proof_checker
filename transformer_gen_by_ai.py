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
#not learning
sys.setrecursionlimit(100000)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

data_train = []
with open('/home/user/Desktop/datasets/gsm8k_train.jsonl', 'r') as f:
    for line in f:
        data_train.append(json.loads(line))
all_tokens = {}
def all_tokens_gsm8k():
    for sample in data_train:
        for char in sample['question']:
            all_tokens[char] = 1
    for sample in data_train:
        for char in sample['answer']:
            all_tokens[char] = 1
all_tokens_gsm8k()
#all_tokens_nt()

alphabet = ''
for token in all_tokens:
    alphabet += token
x_len = 1100

class PositionalEncoding(layers.Layer):
    def __init__(self, maxlen, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(maxlen, d_model)

    def positional_encoding(self, maxlen, d_model):
        pos = np.arange(maxlen)[:, np.newaxis]  # (maxlen, 1)
        i = np.arange(d_model)[np.newaxis, :]  # (1, d_model)
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
        angle_rads = pos * angle_rates  # (maxlen, d_model)

        # Apply sin to even indices and cos to odd indices
        sines = np.sin(angle_rads[:, 0::2])  # (maxlen, d_model/2)
        cosines = np.cos(angle_rads[:, 1::2])  # (maxlen, d_model/2)
        pos_encoding = np.concatenate([sines, cosines], axis=-1)  # (maxlen, d_model)

        return tf.constant(pos_encoding, dtype=tf.float32)[np.newaxis, ...]  # (1, maxlen, d_model)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

class MultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = layers.Dense(d_model)  # Query
        self.wk = layers.Dense(d_model)  # Key
        self.wv = layers.Dense(d_model)  # Value

        self.dense = layers.Dense(d_model)  # Final linear layer

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))  # (batch_size, seq_len, num_heads, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len, depth)

    def call(self, v, k, q):
        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)

        q = self.split_heads(q)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention_logits = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        
        attention_weights = tf.nn.softmax(scaled_attention_logits)

        output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q , depth)

        output = tf.transpose(output, perm=[0, 2, 1 ,3])  # (batch_size , seq_len_q , num_heads , depth)

        output = tf.reshape(output,(tf.shape(output)[0], -1,self.d_model))  # (batch_size , seq_len_q , d_model)

        return self.dense(output)


class FeedForwardNetwork(layers.Layer):
    def __init__(self,d_ff,d_model):
        super(FeedForwardNetwork,self).__init__()
        
        self.dense1=layers.Dense(d_ff , activation='relu')
        self.dense2=layers.Dense(d_model)
    
    def call(self,x):
        x=self.dense1(x)
        
        return self.dense2(x)
    
class TransformerBlock(layers.Layer):
    def __init__(self,d_model,num_heads,d_ff):
        super(TransformerBlock,self).__init__()
        
        self.attention=MultiHeadAttention(num_heads,d_model)
        
        self.ffn=FeedForwardNetwork(d_ff,d_model)
        
        self.layernorm1=layers.LayerNormalization(epsilon=1e-6)
        
        self.layernorm2=layers.LayerNormalization(epsilon=1e-6)

    def call(self,x):
        
        attn_output=self.attention(x,x,x)  
        x=self.layernorm1(x + attn_output) 
        
        ffn_output=self.ffn(x)
        return self.layernorm2(x + ffn_output)

def create_transformer(input_shape,num_heads,d_ff,num_transformer_blocks):
    inputs=layers.Input(shape=input_shape)  
    x=PositionalEncoding(maxlen=input_shape[0],d_model=input_shape[1])(inputs)
    
    for _ in range(num_transformer_blocks):
      x=TransformerBlock(d_model=input_shape[1],num_heads=num_heads,d_ff=d_ff)(x)

    x=layers.GlobalAveragePooling1D()(x)
    
    outputs=layers.Dense(1 , activation='sigmoid')(x) 
    
    return keras.models.Model(inputs=inputs , outputs=outputs)

# Example Usage
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
input_shape = (X[0].shape[0], 2)   # Sequence length of 10 and feature size of 64
num_heads = 2         # Number of attention heads
d_ff = 1000                # Dimension of feed-forward network
num_transformer_blocks = 3 

model=create_transformer(input_shape,num_heads,d_ff,num_transformer_blocks)

model.compile(optimizer='adam',loss='mean_absolute_error',metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X,Y,batch_size=16 , epochs=100)

# Evaluate the model on some test data
loss , accuracy=model.evaluate(X[:100] , Y[:100])
print(f'Loss: {loss}, Accuracy: {accuracy}')