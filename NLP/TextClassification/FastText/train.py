# -*- coding: utf-8 -*-
# @Date    : 2019/9/12
# @Author  : mingming.xu
# @File    : train.py
import numpy as np
from tensorflow import keras
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping

from load_data import load_data, load_data_ngram
from fasttext import FastText

# config
maxlen = 40
max_features = 50
emb_size = 15
batch_size = 1
epochs = 10
ngram_range = 1  # start from 2, end to ngram_range


def build_ngram(input_list, ngram=2):
    '''
    build ngram feature, for example:
    input->[1,2,3,4,5], ngram->2, ngram_info->[(1,2),(2,3),(3,4),(4,5)]
    '''
    ret = set()
    for seq in input_list:
        for word in seq:
            ret.update(set(zip(*[word[i:] for i in range(2, ngram+1)])))
    return set([''.join(r) for r in ret if len(r) > 1])


def build_vocab(input_list):
    vocab_set = set()
    [vocab_set.update(inp) for inp in input_list]
    vocab_ = dict()
    for ind, word in enumerate(vocab_set):
        vocab_[word] = ind + 1
    return vocab_


def word2ind(input_list, vocab=None):
    if not vocab:
        vocab = build_vocab(input_list)

    ret_list = []
    for lst in input_list:
        ret = list(map(lambda x: vocab.get(x, 0), lst))
        ret_list.append(ret)

    return ret_list


def add_ngram(seq, token2id, ngram_range):
    new_seq = []
    for input_ in seq:
        new_list = []
        for word in input_:
            for ngram in range(2, ngram_range + 1):
                for i in range(len(word) - ngram + 1):
                    gram = word[i: i + ngram]
                    if gram in token2id:
                        new_list.append(token2id[gram])
        new_seq.append(new_list)
    return new_seq


print('load data...')
if ngram_range > 1:  # using ngram
    (x_train, y_train), (x_test, y_test) = load_data_ngram()
else:
    try:
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    except:
        print('np.load bug occur...')
        (x_train, y_train), (x_test, y_test) = load_data(num_words=max_features)

# if use ngram feature
if ngram_range > 1:
    ngram_features = set()
    x_train = [x.split(' ') for x in x_train]
    x_test = [x.split(' ') for x in x_test]
    vocab = build_vocab(x_train)
    x_train_ind = word2ind(x_train, vocab)
    x_test_ind = word2ind(x_test, vocab)
    print(len(vocab))
    max_features = len(vocab)
    # for input_list in x_train:
    for i in range(2, ngram_range + 1):
         ngram_features.update(build_ngram(x_train, ngram=i))

    start_ind = max_features + 1
    token2id = {v: k + start_ind for k, v in enumerate(ngram_features)}
    id2token = {k + start_ind: v for k, v in enumerate(ngram_features)}
    max_features += len(token2id)
    max_features += 1
    x_train_ngram = add_ngram(x_train, token2id, ngram_range)
    x_test_ngram = add_ngram(x_test, token2id, ngram_range)
    x_train = list(map(lambda x: x[0] + x[1], zip(x_train_ind, x_train_ngram)))
    x_train = np.array(x_train)
    x_test = list(map(lambda x: x[0] + x[1], zip(x_test_ind, x_test_ngram)))
    x_test = np.array(x_test)
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))
    print('x_train shape: ', x_train.shape)

x_train = sequence.pad_sequences(x_train, maxlen)
x_test = sequence.pad_sequences(x_test, maxlen)
y_train = np.array(y_train)
y_test = np.array(y_test)
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
