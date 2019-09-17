# -*- coding: utf-8 -*-
# @Date    : 2019/9/17
# @Author  : mingming.xu
# @File    : train.py
import logging

import keras
from keras.preprocessing import sequence
from keras import backend as K
from keras.datasets import imdb
from keras.callbacks import EarlyStopping

from han import HAN
from load_data import load_data


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# config
max_features = 5000
max_words = 25
max_seqs = 16
emb_dim = 125
epochs = 5
batch_size = 128

logger.info('loading data..')
try:
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
except:
    (x_train, y_train), (x_test, y_test) = load_data(num_words=max_features)

logger.info('padding...')
x_train = sequence.pad_sequences(x_train, maxlen=max_seqs*max_words)
x_test = sequence.pad_sequences(x_test, maxlen=max_seqs*max_words)

x_train = x_train.reshape((len(x_train), max_seqs, max_words))
x_test = x_test.reshape((len(x_test), max_seqs, max_words))

logger.info('train data shape is: {}'.format(x_train.shape))
logger.info('test data shape is: {}'.format(x_test.shape))

logger.info('build model...')
model = HAN(max_features=max_features, max_words=max_words, max_seqs=max_seqs, emb_dim=emb_dim).build_model()
model.compile('adam', 'binary_crossentropy', ['acc'])

logger.info('training...')
earlystop = EarlyStopping(patience=3, monitor='val_acc', mode='max')
model.fit(x_train, y_train,
          callbacks=[earlystop],
          batch_size=batch_size,
          epochs=epochs,
          validation_data=[x_test, y_test])

logger.info('test...')
pred = model.predict(x_test)
logger.info(pred[:10])
logger.info(y_test[:10])


