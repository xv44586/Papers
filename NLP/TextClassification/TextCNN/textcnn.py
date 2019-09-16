# -*- coding: utf-8 -*-
# @Date    : 2019/9/16
# @Author  : mingming.xu
# @File    : TextCNN.py
import tensorflow as tf
from tensorflow import keras
from keras.models import Input, Model
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Concatenate


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
        conv = []
        for size_ in [3, 4, 5]:
            con_ = Conv1D(filters=25, kernel_size=size_, activation='relu')(embedding)
            pool = GlobalMaxPooling1D()(con_)
            conv.append(pool)
        x = Concatenate()(conv)
        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=[input_], outputs=output)
        return model
