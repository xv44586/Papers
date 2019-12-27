# -*- coding: utf-8 -*-
# @Date    : 2019/12/11
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : train.py

import logging
from itertools import chain

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.callbacks import EarlyStopping
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

from load_data import load
from model import BilstmCRF

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
# config
maxlen = 40
char_emb_dim = 100
seg_label_size = 4
seg_emb_dim = 20
tag_schema = 'iobes'
batch_size = 125
epochs = 10

logger.info('loading data...')
train_data, dev_data, test_data, char_to_id, tag_to_id, id_to_char, id_to_tag = load()
logger.info('train data length: {}'.format(len(train_data)))
logger.info('dev data length: {}'.format(len(dev_data)))
logger.info('test data length: {}'.format(len(test_data)))
logger.info(train_data[0])
class_num = len(tag_to_id)
max_features = len(char_to_id)
lstm_unit_dim = 100

logger.info('class num is: {}'.format(class_num))
logger.info('char num: {}'.format(max_features))


def padding(data, maxlen, char_padding_value=0, seg_padding_value=0, target_padding_value=0, with_string=False):
    data = np.array(data)
    char_data, seg_data, target_data = data[:,1], data[:,2], data[:,3]
    char_padding = sequence.pad_sequences(char_data, maxlen, padding='post', value=char_padding_value)
    seg_padding = sequence.pad_sequences(seg_data, maxlen, padding='post', value=seg_padding_value)
    target_padding = sequence.pad_sequences(target_data, maxlen, padding='post', value=target_padding_value)
    if not with_string:
        return [char_padding, seg_padding, target_padding]

    return [data[:,0], char_padding, seg_padding, target_padding]

train_data = padding(train_data, maxlen)
dev_data = padding(dev_data, maxlen)
test_data = padding(test_data, maxlen, with_string=True)

'''crf layer expects the labels in a shape of (num_examples, max_length, 1)
 https://github.com/keras-team/keras-contrib/issues/244
'''

train_data_x = train_data[:2]
train_data_y = np.expand_dims(train_data[-1], 2)
dev_data_x = dev_data[:2]
dev_data_y = np.expand_dims(dev_data[-1], 2)
test_data_string = test_data[0]
test_data_x = test_data[1:3]
test_data_y = np.expand_dims(test_data[-1], 2)

logger.info('build model...')
model = BilstmCRF(max_features=max_features, maxlen=maxlen, char_emb_size=char_emb_dim,seg_emb_size=seg_emb_dim,
                  seg_label_size=seg_label_size, class_num=class_num, unit_size=lstm_unit_dim).build_model()

logger.info('training...')
earlystop = EarlyStopping(patience=3, mode='max', monitor='val_crf_viterbi_accuracy')
model.compile(optimizer='adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
model.fit(train_data_x, train_data_y, batch_size=batch_size, epochs=epochs, callbacks=[earlystop],
          validation_data=(dev_data_x, dev_data_y))

model.save('project_new.h5')
logger.info('test...')

pred = model.predict(test_data_x).argmax(-1)
test_char = [[id_to_char[c] for c in cc] for cc in test_data_x[0]]

pred_label = [[id_to_tag[p] for p in pp] for pp in pred]

logger.info(list(zip(pred_label, test_char))[:10])
