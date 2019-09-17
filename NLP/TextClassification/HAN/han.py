# -*- coding: utf-8 -*-
# @Date    : 2019/9/17
# @Author  : mingming.xu
# @File    : han.py
'''
https://www.aclweb.org/anthology/N16-1174
'''

import tensorflow as tf
from tensorflow import keras
from keras import Input, Model
from keras.layers import Embedding, Dense, LSTM, TimeDistributed, Bidirectional

from attention import Attention


class HAN(object):
    def __init__(self, max_words, max_seqs, max_features, emb_dim, class_num=1, last_activation='sigmoid'):
        self.max_words = max_words
        self.max_seqs = max_seqs
        self.emb_dim = emb_dim
        self.max_features = max_features
        self.class_num = class_num
        self.last_activation = last_activation

    def build_model(self):
        # word enc
        input_ = Input((self.max_words,))
        emb = Embedding(input_dim=self.max_features, output_dim=self.emb_dim, input_length=self.max_words)(input_)
        enc_word = Bidirectional(LSTM(128, activation='tanh', return_sequences=True))(emb)
        enc_word = Attention(self.max_words)(enc_word)
        model_word = Model(input_, enc_word)

        # sentence enc
        input_ = Input((self.max_seqs, self.max_words))
        enc_sen = TimeDistributed(model_word)(input_)
        enc_sen = Bidirectional(LSTM(128, activation='tanh', return_sequences=True))(enc_sen)
        enc_sen = Attention(self.max_seqs)(enc_sen)

        output = Dense(self.class_num, activation=self.last_activation)(enc_sen)
        model = Model(input_, output)
        return model
