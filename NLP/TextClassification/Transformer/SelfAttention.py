# -*- coding: utf-8 -*-
# @Date    : 2019/9/24
# @Author  : mingming.xu
# @File    : SelfAttention.py
from keras.layers import *
from keras import backend as K

MAX = 1e10


def mask_matrix(x, mask_, mode='mul'):
    if not mask_:
        return x
    for _ in range(K.ndim(x) - K.ndim(mask_)):
        mask_ = K.expand_dims(mask_, -1)
    if mode == 'mul':
        return x * mask_
    else:
        return x - (1 - mask_) * MAX


class Attention(Layer):
    def __init__(self, headers, value_size, key_size=None, mask_right=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.headers = headers  # header count
        self.value_size = value_size  # value vec dim
        self.out_dim = headers * value_size  # output vec dim
        self.key_size = key_size or value_size  # key/query vec dim
        self.mask_right = mask_right  # if mask right part

    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        self.q_dense = Dense(self.headers * self.key_size, use_bias=False)
        self.k_dense = Dense(self.headers * self.key_size, use_bias=False)
        self.v_dense = Dense(self.out_dim, use_bias=False)

    def call(self, inputs, **kwargs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None  # mask_size-> [batch_size, seq_len] / [batch_size, seq_len, 1]
        if len(inputs) > 3:
            v_mask = inputs[3]
        if len(inputs) > 4:
            q_mask = inputs[4]

        # multi-head project
        qh = self.q_dense(q)
        kh = self.k_dense(k)
        vh = self.v_dense(v)

        # transform shape
        qh = K.reshape(qh, (-1, K.shape(qh)[1], self.headers, self.key_size))
        kh = K.reshape(kh, (-1, K.shape(kh)[1], self.headers, self.key_size))
        vh = K.reshape(vh, (-1, K.shape(vh)[1], self.headers, self.value_size))

        # axis change
        qh = K.permute_dimensions(qh, (0, 2, 1, 3))
        kh = K.permute_dimensions(kh, (0, 2, 1, 3))
        vh = K.permute_dimensions(vh, (0, 2, 1, 3))

        # attention
        a = K.batch_dot(qh, kh, [3, 3]) / (self.key_size ** 0.5)  # a's shape -> [batch, headers, seq_len, seq_len]

        # mask by v-mask
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = mask_matrix(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))

        # mask right
        if self.mask_right is True:
            ones = K.ones_like(a[:1, :1])
            mask_ = (ones - K.tf.matrix_band_part(ones, -1, 0)) * MAX
            a = a - mask_
        elif self.mask_right:
            # 0/1 matrix, shape -> [q_len, k_len]
            mask_ = (1 - K.constant(self.mask_right)) * MAX
            mask_ = K.expand_dim(K.expand_dims(mask_, 0), 0)
            a = a - mask_

        # softmax
        a = K.softmax(a)
        output = K.batch_dot(a, vh, [3, 2])
        output = K.permute_dimensions(output, (0, 2, 1, 3))
        output = K.reshape(output, (-1, K.shape(output)[1], self.out_dim))
        output = mask_matrix(output, q_mask, 'mul')
        return output

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0][0], input_shape[0][1], self.out_dim
        else:
            return input_shape[0], input_shape[1], self.out_dim


class SelfAttention(Layer):
    def __init__(self, headers, key_size, value_size=None, mask_right=False, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.headers = headers  # header count
        self.value_size = value_size  # value vec dim
        self.out_dim = headers * value_size  # output vec dim
        self.key_size = key_size or value_size  # key/query vec dim
        self.mask_right = mask_right  # if mask right part

    def build(self, input_shape):
        super(SelfAttention, self).build(input_shape)
        self.attention = Attention(
            self.headers, self.value_size, self.key_size, self.mask_right
        )

    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            x, x_mask = inputs
            output = self.attention([x, x, x, x_mask, x_mask])
        else:
            x = inputs
            output = self.attention([x, x, x])

        return output

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0][0], input_shape[0][1], self.out_dim
        else:
            return input_shape[0], input_shape[1], self.out_dim


class TrainablePositionEmbedding(Layer):
    def __init__(self, maxlen, p_dim, mode='add', **kwargs):
        super(TrainablePositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen  # seq length
        self.p_dim = p_dim  # embedding dimension
        self.mode = mode  # add if 'add' else concatenate

    def build(self, input_shape):
        super(TrainablePositionEmbedding, self).build(input_shape)
        self.p_embeddings = self.add_weight(
            shape=(self.maxlen, self.p_dim),
            initializer='zero',
            name='position_embedding'
        )

    def call(self, inputs, **kwargs):
        # allow input current position
        if isinstance(inputs, list):
            x, cp = inputs
        else:
            x = inputs
            cp = 0

        pind = K.arange(K.shape(x)[1])  # position indices
        pind = K.expand_dims(pind, 0)  # add col axis
        pind = K.tile(pind, (K.shape(x)[0], 1))  # expand to batch size
        pind = K.abs(K.cast(pind - cp, 'int32'))  # start from current position
        pemb = K.gather(self.p_embeddings, pind)

        if self.mode == 'add':
            return x + pemb

        return K.concatenate([x, pemb], -1)

    def compute_output_shape(self, input_shape):
        if self.mode == 'add':
            return input_shape

        return input_shape[0], input_shape[1], input_shape[2] + self.p_dim


