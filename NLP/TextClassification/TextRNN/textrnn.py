# -*- coding: utf-8 -*-
# @Date    : 2019/9/16
# @Author  : mingming.xu
# @File    : textrnn.py
import tensorflow as tf
from tensorflow import keras
from keras.models import Input, Model
from keras.layers import Embedding, Dense, LSTM


class TextRNN(object):
    def __init__(self,maxlen, max_features,emb_dim, clas_num=1, last_activation='sigmoid'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.emb_dim = emb_dim
        self.class_num =clas_num
        self.last_activation = last_activation

    def build_model(self):
        input_ = Input((self.maxlen, ))
        embedding = Embedding(input_dim=self.max_features, output_dim=self.emb_dim, input_length=self.maxlen)(input_)
        x = LSTM(128, activation='tanh')(embedding)  # rnn or gru
        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=[input_], outputs=output)
        return model