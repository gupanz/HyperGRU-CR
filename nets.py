#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Xusong Chen

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import tensorflow as tf


def var_init(name, shape, initializer=tf.contrib.layers.xavier_initializer(),
             trainable=True):
    with tf.device('/gpu:0'):
        var = tf.get_variable(
            name=name,
            shape=shape,
            initializer=initializer,
            trainable=trainable
        )
        if not tf.get_variable_scope().reuse:
            tf.add_to_collection("parameters", var)
        return var


def dense(x,
          units,
          name,
          keep_prob=1.0,
          activation=None,
          kernel_initializer=tf.contrib.layers.xavier_initializer(),
          bias_initializer=tf.zeros_initializer(),
          reuse=None):
    """
    Functional interface for the densely-connected layer.
    :param x: Tensor input.
    :param units: Integer or Long, dimensionality of the output space.
    :param name: String, the name of the parameter.
    :param keep_prob: A scalar Tensor with the same type as x. The probability that each element is kept.
    :param activation: Activation function (callable). Set it to None to maintain a linear activation.
    :param kernel_initializer:
    :param bias_initializer:
    :param reuse:
    :return:
    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        if not isinstance(name, (list, tuple)):
            raise ValueError('name should be list or tuple')

        prev_units = x.get_shape().as_list()[-1]
        w = var_init(name[0], (prev_units, units), kernel_initializer)
        out = tf.tensordot(x, w, axes=[[-1], [0]])
        if len(name) > 1:
            b = var_init(name[1], units, bias_initializer)
            out += b

        if activation is not None:
            out = activation(out)

        out = tf.nn.dropout(out, keep_prob)
        return out



def scaled_tanh(x, scale=5.):
    return scale * tf.nn.tanh(1. / scale * x)



def softmax_mask(x, mask):
    """
    :param x: [n,m,d] or [n,b,m,d]
    :param mask: [n,m] or [n,b,m]
    :return:
    """
    x_shape = x.get_shape().as_list()
    pad_num = len(x_shape) - 1
    mask = tf.tile(tf.expand_dims(mask, axis=-1), pad_num * [1] + [x_shape[-1]])
    paddings = tf.ones_like(mask, tf.float32) * (-2 ** 32 + 1)
    softmax_mask = tf.where(mask, x, paddings)
    return softmax_mask


def mask01(x, mask):
    """
    :param x: [n,m,d] or [n,b,m,d]
    :param mask: [n,m] or [n,b,m]
    :return:
    """
    x_shape = x.get_shape().as_list()
    pad_num = len(x_shape) - 1
    mask = tf.tile(tf.expand_dims(mask, axis=-1), pad_num * [1] + [x_shape[-1]])
    mask_x = tf.where(mask, x, tf.cast(mask, tf.float32))
    return mask_x


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable('beta', shape=params_shape)
        gamma = tf.get_variable('gamma', shape=params_shape)
        normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))
        outputs = gamma * normalized + beta

    return outputs




def dnn(x,
        fusion_layers,
        keep_prob):
    """
    Feedforward network.
    :param x: Tensor input.
    :param fusion_layers: List, the layers of feedforward network, like [256, 128].
    :param keep_prob:
    :return: The micro-video click-through probabilities.
    """
    _, dim = x.get_shape().as_list()
    n_layers = len(fusion_layers)
    for i in range(n_layers):
        x = dense(x, fusion_layers[i], ['w{}'.format(i + 1), 'b{}'.format(i + 1)], keep_prob, tf.nn.relu)

    logit = dense(x, 1, ['w{}'.format(n_layers + 1), 'b{}'.format(n_layers + 1)], keep_prob)
    return tf.squeeze(logit)
