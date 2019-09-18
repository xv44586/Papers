# -*- coding: utf-8 -*-
# @Date    : 2019/9/17
# @Author  : mingming.xu
# @File    : text_birnn_att.py

'''
https://www.aclweb.org/anthology/P16-2034
'''


import tensorflow as tf
from tensorflow import keras
from keras import Input, Model
from keras.layers import Embedding, Dense, LSTM, TimeDistributed, Bidirectional

from attention import Attention


class TextBiRNNAtt(object):
    def __init__(self, maxlen, max_features, emb_dim, class_num=1, last_activation='sigmoid'):
        self.maxlen = maxlen
        self.emb_dim = emb_dim
        self.max_features = max_features
        self.class_num = class_num
        self.last_activation = last_activation

    def build_model(self):
        input_ = Input((self.maxlen,))
        emb = Embedding(input_dim=self.max_features, output_dim=self.emb_dim, input_length=self.maxlen)(input_)
        enc = Bidirectional(LSTM(128, activation='tanh', return_sequences=True))(emb)
        enc = Attention(self.maxlen)(enc)

        output = Dense(self.class_num, activation=self.last_activation)(enc)
        model = Model(input_, output)
        return model
