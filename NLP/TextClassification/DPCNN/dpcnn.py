# -*- coding: utf-8 -*-
# @Date    : 2019/9/18
# @Author  : mingming.xu
# @File    : dpcnn.py

'''
https://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf
'''


from tensorflow import keras
from keras import Input, Model
from keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, BatchNormalization, Dropout, PReLU
from keras.layers import Add, SpatialDropout1D, LeakyReLU


class DPCNN(object):
    def __init__(self, maxlen, max_features, emb_dim, repeat_num=None, class_num=1, last_activation='sigmoid'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.emb_dim = emb_dim
        self.repeat_num = repeat_num
        self.class_num = class_num
        self.last_activation = last_activation

        self.conv_activation = 'linear'  # convolution layer: W * a(x) + b
        self.conv_fiters = 250
        self.conv_kernel_size = 3
        self.conv_stride = 1

        self.pool_size = 3
        self.pool_stride = 2

        self.sp_dropout = 0.2

        self.dense_units = 100
        self.dropout = 0.2

    def build_model(self):
        input_ = Input((self.maxlen,))
        emb = Embedding(input_dim=self.max_features, output_dim=self.emb_dim, input_length=self.maxlen)(input_)
        # region embedding
        emb = SpatialDropout1D(self.sp_dropout)(emb)
        region_emb = Conv1D(filters=self.conv_fiters, kernel_size=1, padding='same')(emb)
        pre_activation = PReLU()(region_emb)

        block = None

        repeat_num = self.maxlen // 2
        if self.repeat_num:
            repeat_num = min(repeat_num, self.repeat_num)

        for i in range(repeat_num):
            # first layer
            if i == 0:
                block = self.ResCon(emb)
                res_block = Add()([pre_activation, block])
                block = MaxPooling1D(pool_size=self.pool_size, strides=self.pool_stride)(res_block)
            # last layer
            elif i == repeat_num - 1:
                block_last = self.ResCon(block)
                res_block = Add()([block, block_last])
                block = GlobalMaxPooling1D()(res_block)
                break

            # middle layer
            else:
                block_mid = self.ResCon(block)
                res_block = Add()([block_mid, block])
                block = MaxPooling1D(pool_size=self.pool_size, strides=self.pool_stride)(res_block)

        # dense layer
        x = Dense(units=self.dense_units, activation='linear', use_bias=False)(block)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout)(x)
        output = Dense(units=self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input_, outputs=output)
        model.summary()
        return model

    def ResCon(self, input_):
        '''
        two views
        '''
        con_1 = Conv1D(filters=self.conv_fiters,
                       kernel_size=self.conv_kernel_size,
                       strides=self.conv_stride,
                       padding='same',
                       activation=self.conv_activation)(input_)
        x = BatchNormalization()(con_1)

        con_2 = Conv1D(filters=self.conv_fiters,
                       kernel_size=self.conv_kernel_size,
                       strides=self.conv_stride,
                       padding='same',
                       activation=self.conv_activation)(x)
        x = BatchNormalization()(x)
        # per-activation, activation in paper is relu
        x = PReLU()(x)
        return x
