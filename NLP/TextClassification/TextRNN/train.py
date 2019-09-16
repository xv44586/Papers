# -*- coding: utf-8 -*-
# @Date    : 2019/9/16
# @Author  : mingming.xu
# @File    : train.py
import logging

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.callbacks import EarlyStopping

from load_data import load_data
from textrnn import TextRNN

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
# config
maxlen = 400
max_features = 500
emb_dim = 125
batch_size = 125
epochs = 5

logger.info('loading data...')
try:
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
except:
    logger.info('np bug occur...')
    (x_train, y_train), (x_test, y_test) = load_data(num_words=max_features)
logger.info('train data length: {}'.format(len(x_train)))
logger.info('test data length: {}'.format(len(x_test)))

logger.info('padding data...')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

logger.info('build model...')
model = TextRNN(max_features=max_features, maxlen=maxlen, emb_dim=emb_dim).build_model()

logger.info('training...')
earlystop = EarlyStopping(patience=3, mode='max', monitor='val_acc')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[earlystop],
          validation_data=(x_test, y_test))

logger.info('test...')
test = model.predict(x_test)
logger.info(test[:5])
logger.info(y_test[:5])
