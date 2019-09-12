# -*- coding: utf-8 -*-
# @Date    : 2019/9/12
# @Author  : mingming.xu
# @File    : rcnn.py
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, Embedding, LSTM, concatenate, Lambda, Conv1D, GlobalMaxPool1D
from keras.models import Model


class RCNN(object):
    def __init__(self, maxlen, max_feats, emb_dim, class_num=1, last_activation='sigmoid'):
        self.maxlen = maxlen
        self.max_feats = max_feats
        self.emb_dim = emb_dim
        self.class_num = class_num
        self.last_activation = last_activation

    def get_model(self):
        input_ = Input((self.maxlen, ))
        input_left = Input((self.maxlen, ))
        input_right = Input((self.maxlen, ))

        embedding = Embedding(input_dim=self.max_feats, output_dim=self.emb_dim, input_length=self.maxlen)
        emb_ = embedding(input_)
        emb_left = embedding(input_left)
        emb_right = embedding(input_right)

        enc_left = LSTM(128, activation='tanh', return_sequences=True)(emb_left)
        enc_right = LSTM(128, activation='tanh', go_backwards=True, return_sequences=True)(emb_right)
        enc_right = Lambda(lambda x: tf.reverse(x, axis=[1]))(enc_right)

        x = concatenate([enc_left, emb_, enc_right], axis=-1)

        x = Conv1D(32, kernel_size=1, activation='tanh')(x)
        x = GlobalMaxPool1D()(x)

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=[input_, input_left, input_right], outputs=output)
        return model
