# -*- coding: utf-8 -*-
# @Date    : 2019/9/12
# @Author  : mingming.xu
# @File    : train.py
import numpy as np
from tensorflow import keras
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping

from load_data import load_data
from rcnn import RCNN

# config
max_features = 5000
maxlen = 400
batch_size = 128
emb_dim = 50
epochs = 10

print('loading data ...')
try:
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
except:
    print('np bug occur...')
    (x_train, y_train), (x_test, y_test) = load_data(num_words=max_features)
print('train sequence: ', len(x_train))
print('test sequence: ', len(x_test))

print('pad sequence...')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('train data shape: ', x_train.shape)

print('generate context format...')
x_train_current = x_train
x_train_left = np.hstack([np.expand_dims(x_train[:, 0], axis=1), x_train[:, 0:-1]])

x_train_right = np.hstack([x_train[:, 1:], np.expand_dims(x_train[:, -1], axis=1)])

x_test_current = x_test
x_test_left = np.hstack([np.expand_dims(x_test[:, 0], axis=1), x_train[:, 0:-1]])
x_test_right = np.hstack([x_test[:, 1:], np.expand_dims(x_test[:, -1], axis=1)])
print('x train shape: ', x_train_current.shape)
print('x train left shape: ', x_train_left.shape)
print('x train right shape: ', x_train_right.shape)

print('build model...')
model = RCNN(maxlen, max_features, emb_dim).get_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print('training ...')
earlystop = EarlyStopping(monitor='val_acc', patience=3, mode='max')
model.fit([x_train_current, x_train_left, x_train_right], y_train, batch_size=batch_size, epochs=epochs,
          callbacks=[earlystop],
          validation_data=([x_test_current, x_test_left, x_test_right], y_test))
