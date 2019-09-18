# -*- coding: utf-8 -*-
# @Date    : 2019/9/17
# @Author  : mingming.xu
# @File    : attention.py
import tensorflow as tf
from tensorflow import keras
from keras.engine.topology import Layer
from keras import regularizers, constraints, initializers
from keras import backend as K


class Attention(Layer):
    '''
    Uit = tanh(Ww * hit + bw)
    a = softmax(Uit * Uw)
    Si = sum(a * hit)
    '''
    def __init__(self, step_dim,
                 W_constraint=None, b_constraint=None,
                 W_regularizer=None, b_regularizer=None, bias=False, **kwargs):
        self.step_dim = step_dim
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        wh = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            wh += self.b

        Uit = K.tanh(wh)
        alpha = K.exp(Uit)
        att = alpha / K.cast(K.sum(alpha, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        att = K.expand_dims(att)

        Si = K.sum(att * x, axis=1)
        return Si

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim
