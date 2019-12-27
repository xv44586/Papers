# -*- coding: utf-8 -*-
# @Date    : 2019/12/11
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : model.py
import keras
from keras import Input, Model
from keras.layers import Embedding, Dense, LSTM, Dropout, Bidirectional, concatenate
from keras_contrib.layers import CRF


class BilstmCRF(object):
    def __init__(self, char_emb_size, maxlen, max_features, class_num, unit_size, seg_emb_size, seg_label_size=4):
        self.maxlen = maxlen
        self.char_emb_size = char_emb_size
        self.class_num = class_num
        self.unit_size = unit_size
        self.max_features = max_features
        self.seg_emb_size = seg_emb_size
        self.seg_label_size = seg_label_size  # bies

    def build_model(self):
        char_input = Input((self.maxlen,), name='char_input')
        char_emb = Embedding(input_length=self.maxlen, input_dim=self.max_features, output_dim=self.char_emb_size,
                             name='char_emb', mask_zero=False)(char_input)

        seg_input = Input((self.maxlen,), name='seg_input')
        seg_emb = Embedding(input_length=self.maxlen, input_dim=self.max_features, output_dim=self.seg_emb_size,
                            name='seg_emb', mask_zero=False)(seg_input)

        emb = concatenate([char_emb, seg_emb], axis=-1)
        lstm = Bidirectional(LSTM(self.unit_size, activation='tanh', return_sequences=True))(emb)
        dense = Dense(self.class_num, activation='relu')(lstm)
        dense = Dropout(0.7)(dense)
        crf = CRF(units=self.class_num, sparse_target=True)
        output = crf(dense)

        model = Model([char_input, seg_input], output)
        model.summary()
        return model
