# -*- coding: utf-8 -*-
# @Date    : 2019/9/23
# @Author  : mingming.xu
# @File    : transformer.py


from keras import Input, Model
from keras.layers import *

from SelfAttention import *


class Transformer(object):
    def __init__(self, max_features, maxlen, emb_dim, class_num=1, last_activation='sigmoid', layers_num=1, headers=6,
                 value_size=20, key_size=None, dropout=0.3):
        self.max_features = max_features
        self.maxlen = maxlen
        self.emb_dim = emb_dim
        self.class_num = class_num
        self.last_activation = last_activation
        self.layers_num = layers_num
        self.headers = headers
        self.value_size = value_size
        self.key_size = key_size
        self.dropout = dropout

    def build(self):
        input_ = Input((self.maxlen,))
        embedding = Embedding(self.max_features, self.emb_dim, input_length=self.maxlen)(input_)
        att = SelfAttention(headers=self.headers, value_size=self.value_size, key_size=self.key_size)(
            embedding)
        # attention loop
        for _ in range(self.layers_num - 1):
            att = SelfAttention(headers=self.headers, value_size=self.value_size, key_size=self.key_size)(att)

        output = GlobalAveragePooling1D()(att)
        output = Dropout(self.dropout)(output)
        output = Dense(self.class_num, activation=self.last_activation)(output)
        model = Model(input_, output)
        return model
