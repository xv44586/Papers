# -*- coding: utf-8 -*-
# @Date    : 2019/9/27
# @Author  : mingming.xu
# @File    : dropout.py


import tensorflow as tf
from tensorflow import keras
from keras.models import Input, Model, Layer
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Concatenate, Dropout
from keras import backend as K
from keras.legacy import interfaces


class MyDropout(Layer):
    """Applies Dropout to the input.

    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.

    # Arguments
        rate: float between 0 and 1. Fraction of the input units to drop.
        noise_shape: 1D integer tensor representing the shape of the
            binary dropout mask that will be multiplied with the input.
            For instance, if your inputs have shape
            `(batch_size, timesteps, features)` and
            you want the dropout mask to be the same for all timesteps,
            you can use `noise_shape=(batch_size, 1, features)`.
        seed: A Python integer to use as random seed.
        scaled: if scaled-up inputs

    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    """

    @interfaces.legacy_dropout_support
    def __init__(self, rate, noise_shape=None, seed=None, scaled=False, **kwargs):
        super(MyDropout, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True
        self.scaled = scaled

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

    @staticmethod
    def drop(inputs, rate,scaled, noise_shape=None):
        keep_prob = 1.0 - rate
        if keep_prob >= 1.:
            return inputs

        random_tensor = keep_prob
        noise_shape = noise_shape if noise_shape is not None else K.shape(inputs)
        random_tensor += K.random_uniform(noise_shape, dtype=K.dtype(inputs))
        binary_tensor = K.tf.floor(random_tensor)
        if scaled:
            inputs = K.tf.div(inputs, keep_prob) * binary_tensor
        else:
            inputs = inputs * binary_tensor
        return inputs

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return MyDropout.drop(inputs,self.rate, self.scaled, noise_shape)

            return K.in_train_phase(dropped_inputs, inputs,
                                    training=training)
        return inputs

    def get_config(self):
        config = {'rate': self.rate,
                  'noise_shape': self.noise_shape,
                  'seed': self.seed}
        base_config = super(Dropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
