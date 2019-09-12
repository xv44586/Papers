# -*- coding: utf-8 -*-
# @Date    : 2019/9/12
# @Author  : mingming.xu
# @File    : fasttext.py
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense
from keras.models import Model


class FastText(object):
    def __init__(self, maxlen, max_features, emb_dim, class_num=1, last_activation='sigmoid'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.emb_dim = emb_dim
        self.class_num = class_num
        self.last_activation = last_activation

    def build_model(self):
        input_ = Input((self.maxlen, ))
        embedding = Embedding(input_dim=self.max_features, output_dim=self.emb_dim, input_length=self.maxlen)(input_)
        output = GlobalAveragePooling1D()(embedding)
        output = Dense(self.class_num, activation=self.last_activation)(output)
        model = Model(inputs=[input_], outputs=output)
        return model
