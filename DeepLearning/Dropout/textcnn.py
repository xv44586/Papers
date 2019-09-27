# -*- coding: utf-8 -*-
# @Date    : 2019/9/27
# @Author  : mingming.xu
# @File    : TextCNN.py

'''
https://www.aclweb.org/anthology/D14-1181
'''

import tensorflow as tf
from tensorflow import keras
from keras.models import Input, Model, Layer
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Concatenate, Dropout
from keras import backend as K

from dropout import *


class TextCNN(object):
    def __init__(self, maxlen, max_features, emb_dim, class_num=1, scaled_inputs=False, last_activation='sigmoid'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.emb_dim = emb_dim
        self.class_num = class_num
        self.last_activation = last_activation
        self.scaled_inputs = scaled_inputs

    def build_model(self):
        input_ = Input((self.maxlen,))
        embedding = Embedding(input_dim=self.max_features, output_dim=self.emb_dim, input_length=self.maxlen)(input_)
        # embedding = Dropout(0.5)(embedding)
        conv = []
        for size_ in [2, 3, 4]:
            con_ = Conv1D(filters=25, kernel_size=size_, activation='tanh')(embedding)
            pool = GlobalMaxPooling1D()(con_)
            conv.append(pool)
        x = Concatenate()(conv)
        # scaled inputs
        if self.scaled_inputs:
            x = MyDropout(0.5, scaled=True)(x)
            x = Dense(5, activation='tanh')(x)
        else:
            x = MyDropout(0.5, scaled=False)(x)
            x = Dense(5, activation='tanh')(x)
            x = AfterDropout(0.5)(x)
        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=[input_], outputs=output)
        return model
