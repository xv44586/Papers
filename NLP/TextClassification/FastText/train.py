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
from fasttext import FastText

# config
maxlen = 400
max_features = 1000
emb_size = 50
batch_size = 32
epochs = 10
ngram_range = 1  # start from 2, end to ngram_range


def build_ngram(input_list, ngram=2):
    '''
    build ngram feature, for example:
    input->[1,2,3,4,5], ngram->2, ngram_info->[(1,2),(2,3),(3,4),(4,5)]
    '''
    return set(zip(*[input_list[i:] for i in ngram]))


def add_ngram(seq, token2id, ngram_range):
    new_seq = []
    for input_ in seq:
        new_list = input_[:]
        for ngram in range(2, ngram_range + 1):
            for i in range(len(input_) - ngram + 1):
                gram = input_[i: i + ngram]
                if gram in token2id:
                    new_list.append(token2id[gram])
        new_seq.append(new_list)
    return new_seq


print('load data...')
try:
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
except:
    print('np.load bug occur...')
    (x_train, y_train), (x_test, y_test) = load_data(num_words=max_features)

# if use ngram feature
if ngram_range > 1:
    ngram_features = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            ngram_features.update(build_ngram(input_list, ngram=i))

    start_ind = max_features + 1
    token2id = {v: k + start_ind for k, v in enumerate(ngram_features)}
    id2token = {k + start_ind: v for k, v in enumerate(ngram_features)}
    max_features += len(token2id)

    x_train = add_ngram(x_train, token2id, ngram_range)
    x_test = add_ngram(x_test, token2id, ngram_range)
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

x_train = sequence.pad_sequences(x_train, maxlen)
x_test = sequence.pad_sequences(x_test, maxlen)

print('build model...')
model = FastText(maxlen, max_features, emb_size).build_model()
model.compile('adam', loss='binary_crossentropy', metrics=['acc'])

print('training...')
earlystop = EarlyStopping(monitor='val_acc', patience=3, mode='max')
model.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[earlystop], validation_data=[x_test, y_test])
