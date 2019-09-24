# -*- coding: utf-8 -*-
# @Date    : 2019/9/23
# @Author  : mingming.xu
# @File    : train.py


import logging

from keras.datasets import imdb
from keras.preprocessing import sequence

from load_data import load_data
from transformer import Transformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# config
maxlen = 80
max_features = 20000
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
model = Transformer(max_features, maxlen, emb_dim, layers_num=2).build()
model.summary()

logger.info('training...')
model.compile('adam', 'binary_crossentropy', ['acc'])
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=[x_test, y_test])

logger.info('test...')
pred = model.predict(x_test)
logger.info(pred[:10])
logger.info(y_test[:10])
