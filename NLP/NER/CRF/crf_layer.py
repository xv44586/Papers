# -*- coding: utf-8 -*-
# @Date    : 2019/12/24
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : crf.py
import keras
from keras.layers import Layer
from keras import backend as K


class CRF(Layer):
    def __init__(self, mask_label=False, **kwargs):
        self.mask_label = 1 if mask_label else 0
        super(CRF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_label = input_shape[-1] - self.mask_label
        self.trans = self.add_weight(name='crf_trans',
                                     shape=(self.num_label, self.num_label),
                                     trainable=True,
                                     initializer='glorot_uniform')

    def path_score(self, inputs, labels):
        '''
        :param inputs: (batch_size, timesteps, num_label), obtained from rnn(lstm, bilstm. etc.)
        :param labels: one-hot, (batch_size, timesteps, num_label) , real target series
        :return:  path score
        '''
        point_score = K.sum(K.sum(inputs * labels, 2), 1, keepdims=True)
        label_pre = K.expand_dims(labels[:, :-1], 3)
        label_next = K.expand_dims(labels[:, 1:], 2)
        label_trans = label_pre * label_next
        trans = K.expand_dims(K.expand_dims(self.trans, 0), 0)
        trans_score = K.sum(K.sum(label_trans * trans, [2, 3]), 1, keepdims=True)
        return point_score + trans_score

    def log_norm_pre(self, inputs, states):
        '''
        expand previous states and inputs, sum with trans
        :param inputs: (batch_size, num_label), current word emission scores
        :param states: (batch_size, num_label), all paths  score of previous word
        :return:
        '''
        states = K.expand_dims(states[0], 2)
        inputs = K.expand_dims(inputs, 1)
        trans = K.expand_dims(self.trans, 0)
        scores = states + trans + inputs
        output = K.logsumexp(scores, 1)
        return output, [output]
        # states = K.expand_dims(states[0], 2)  # (batch_size, output_dim, 1)
        # trans = K.expand_dims(self.trans, 0)  # (1, output_dim, output_dim)
        # output = K.logsumexp(states + trans, 1)  # (batch_size, output_dim)
        # return output + inputs, [output + inputs]

    def loss(self, y_true, y_pre):
        '''
        :param inputs: (batch_size, timesteps, num_label)
        :return:
        '''
        # mask = 1 - y_true[:, 1: -1] if self.mask_label else None
        # # y_true, y_pred = y_true[:, :, :self.num_label], y_pre[:, :, :self.num_label]
        # real_path_score = self.path_score(y_pre, y_true)
        # init_states = [y_pre[:, 0]]
        # log_norm, _ = K.rnn(self.log_norm_pre, initial_states=init_states, inputs=y_pre[:, 1:], mask=mask)  # log(Z)
        # log_norm_score = K.logsumexp(log_norm, 1, keepdims=True)
        # return log_norm_score - real_path_score
        mask = 1 - y_true[:, 1:, -1] if self.mask_label else None
        y_true, y_pre = y_true[:, :, :self.num_label], y_pre[:, :, :self.num_label]
        init_states = [y_pre[:, 0]]  # 初始状态
        log_norm, _, _ = K.rnn(self.log_norm_pre, y_pre[:, 1:], init_states, mask=mask)  # 计算Z向量（对数）
        log_norm = K.logsumexp(log_norm, 1, keepdims=True)  # 计算Z（对数）
        path_score = self.path_score(y_pre, y_true)  # 计算分子（对数）
        return log_norm - path_score  # 即log(分子/分母)

    def call(self, inputs):  # crf 只是loss，不改变inputs
        return inputs

    def accuracy(self, y_true, y_pred):
        mask = 1 - y_true[:, :, -1] if self.mask_label else None
        y_true, y_pred = y_true[:, :, :self.num_label], y_pred[:, :, :self.num_label]
        isequal = K.equal(K.argmax(y_true, 2), K.argmax(y_pred, 2))
        isequal = K.cast(isequal, 'float32')
        if mask == None:
            return K.mean(isequal)
        else:
            return K.sum(isequal * mask) / K.sum(mask)