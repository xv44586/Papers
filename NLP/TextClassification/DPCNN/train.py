# -*- coding: utf-8 -*-
# @Date    : 2019/9/18
# @Author  : mingming.xu
# @File    : train.py

import logging

from keras.datasets import imdb
from keras.preprocessing import sequence

from dpcnn import DPCNN
from load_data import load_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# config
max_features = 5000
maxlen = 400
emb_dim = 128

# dp layer num
repeat_num = 3

batch_size = 128
epochs = 5

logger.info('loading data...')
try:
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
except:
    (x_train, y_train), (x_test, y_test) = load_data(num_words=max_features)
logger.info('train data length is : {}'.format(len(x_train)))
logger.info('test data length is : {}'.format(len(x_test)))

logger.info('padding...')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

logger.info('build model...')
model = DPCNN(maxlen=maxlen, max_features=max_features, emb_dim=emb_dim, repeat_num=repeat_num).build_model()
model.compile('adam', 'binary_crossentropy', ['acc'])
logger.info('training...')
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=[x_test, y_test])

logger.info('test...')
pred = model.predict(x_test)
logger.info(pred[:10])
logger.info(y_test[:10])
