# -*- coding: utf-8 -*-
# @Date    : 2019/9/16
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


class AfterDropout(Layer):
    def __init__(self, rate, **kwargs):
        super(AfterDropout, self).__init__(**kwargs)
        self.rate = rate

    def build(self, input_shape):
        super(AfterDropout, self).build(input_shape)

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:

            def scaled_inputs():
                return inputs * self.rate

            return K.in_train_phase(inputs, scaled_inputs,
                                    training=training)
        return inputs


class TextCNN(object):
    def __init__(self, maxlen, max_features, emb_dim, class_num=1, last_activation='sigmoid'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.emb_dim = emb_dim
        self.class_num = class_num
        self.last_activation = last_activation

    def build_model(self):
        input_ = Input((self.maxlen, ))
        embedding = Embedding(input_dim=self.max_features, output_dim=self.emb_dim, input_length=self.maxlen)(input_)
        # embedding = Dropout(0.5)(embedding)
        conv = []
        for size_ in [3, 4, 5]:
            con_ = Conv1D(filters=25, kernel_size=size_, activation='tanh')(embedding)
            pool = GlobalMaxPooling1D()(con_)
            conv.append(pool)
        x = Concatenate()(conv)
        x = Dropout(0.5)(x)
        x = Dense(10, activation='relu')(x)
        # x = AfterDropout(0.5)(x)
        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=[input_], outputs=output)
        return model
