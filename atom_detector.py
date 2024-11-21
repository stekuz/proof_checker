import json
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import base64

data = json.load(open('./numbers.json','r'))

def encode_x(text):
    res = []
    for i in text:
        res.append(ord(i))
    return res

def encode_y(atom):
    res = []
    for i in atom['name']:
        res.append(ord(i))
    res.append(ord(' '))
    for i in atom['type']:
        res.append(ord(i))
    res.append(ord(' '))
    for i in str(atom['value']):
        res.append(ord(i))
    return res



def train_model():
    mxlen=100
    x_train = []
    y_train = []
    for i in data:
        x = encode_x(i)
        y = encode_y(data[i])
        x_now = x[:]
        x_train.append(x)
        y_train.append(y[0])
        for j in range(len(y)-1):
            x_now.append(y[j])
            mxlen=max(mxlen, len(x_now))
            x_train.append(x_now[:])
            y_train.append(y[j+1])

    for i in range(len(x_train)):
        add = [0]*(mxlen-len(x_train[i]))
        x_train[i] = add[:]+x_train[i][:]

    print(x_train[0],y_train[0],len(y_train),mxlen)
    total_chars = 300

    model = keras.Sequential()
    model.add(layers.Embedding(total_chars, 100))
    model.add(layers.LSTM(150))
    model.add(layers.Dropout(0.1))
    #model.add(layers.LSTM(100))
    model.add(layers.Flatten())
    model.add(layers.Dense(total_chars, activation='softmax'))
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            keras.metrics.CategoricalAccuracy(),
        ],
    )
    x_train = np.asarray(x_train).astype('float64')
    print(len(x_train))
    y_train = keras.utils.to_categorical(y_train, total_chars)
    y_train = np.asarray(y_train).astype('float64')
    cp_callback = keras.callbacks.ModelCheckpoint(filepath='./weights_atom_detector.ckpt.keras')
    model.fit(x_train, y_train, batch_size=16, epochs=20, verbose=2, callbacks=[cp_callback])

def use_model():
    mxlen=100
    model = keras.models.load_model('./weights_atom_detector.ckpt.keras', compile=True)
    print('Ready!')
    while True:
        x = encode_x(input())
        res = ''
        for j in range(20):
            x0 = x[:]
            x = np.asarray([([0]*(mxlen-len(x)))[:] + x[:]]).astype('float64')
            prediction = model.predict(x)
            mx = 0
            mxi = 0
            for i in range(len(prediction[0])):
                if prediction[0][i]>mx:
                    mx = prediction[0][i]
                    mxi = i
            res += chr(mxi)
            x = x0[:] + [mxi]
        print(res)
        

use_model()