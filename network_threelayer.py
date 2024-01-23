# -*- coding:utf-8 -*-
import logging
import sys

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell

# from tensorflow.contrib.rnn import GRUCell, LSTMCell
# from tensorflow.nn import dynamic_rnn


import warnings
import nets

warnings.filterwarnings("ignore")
# tf.compat.v1.enable_eager_execution()
# tf.enable_eager_execution()

epsilon = 1e-9


class Model(object):

    def __init__(self, settings):
        self.max_length = settings.max_length
        self.n_items = settings.n_items
        self.n_users = settings.n_users
        self.item_id_dim = settings.item_dim       # item_id的维度
        self.item_dim = settings.item_dim
        self.dnn_size = settings.dnn_size
        self.hidden_size = self.item_dim
        self.lr_reg = settings.lr_reg
        self.cons_w = settings.cons_w
        self.tau = settings.tau
        self.contrastive_length_threshold = settings.contrastive_length_threshold
        self.pos_w = settings.pos_w
        self.predict_source = settings.predict_source

        self.batch_size = settings.batch_size

        self.global_step = tf.Variable(0, trainable=False, name='Global_Step')

        self.lr = tf.maximum(1e-5, tf.train.exponential_decay(settings.learning_rate,
                                                              self.global_step,
                                                              settings.decay_steps,
                                                              settings.decay_rate,
                                                              staircase=True))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        self.item_embedding = tf.get_variable('item_embedding', [self.n_items, self.item_id_dim], initializer=tf.glorot_normal_initializer(), trainable=True)

        # placeholders
        self.tst = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.name_scope('Inputs'):
            self.seq_inputs = tf.placeholder(tf.int32, [self.batch_size, self.max_length], name='seq_inputs')  # all history  seq_input
            self.seq_mask_inputs = tf.placeholder(tf.int32, [self.batch_size], name='seq_mask_input')
            self.seq_clickmask_input = tf.placeholder(tf.bool, [self.batch_size, self.max_length], name='seq_clickmask_input')  # click mask
            self.seq_likemask_input = tf.placeholder(tf.bool, [self.batch_size, self.max_length], name='seq_likemask_input')  # like mask
            self.seq_pos_inputs = tf.placeholder(tf.int32, [self.batch_size, self.max_length], name='seq_pos_inputs')
            self.seq_pos_mask_inputs = tf.placeholder(tf.int32, [self.batch_size], name='seq_pos_mask_inputs')
            self.seq_neg_inputs = tf.placeholder(tf.int32, [self.batch_size, self.max_length], name='seq_neg_inputs')
            self.seq_neg_mask_inputs = tf.placeholder(tf.int32, [self.batch_size], name='seq_neg_mask_inputs')
            self.y_inputs = tf.placeholder(tf.float32, [self.batch_size], name='y_input')  # label
            self.item_inputs = tf.placeholder(tf.int32, [self.batch_size], name='item_inputs')  # item_input
            self.item = tf.nn.embedding_lookup(self.item_embedding, self.item_inputs)
            self.seq_features = tf.nn.embedding_lookup(self.item_embedding, self.seq_inputs)
            self.seq_pos_features = tf.nn.embedding_lookup(self.item_embedding, self.seq_pos_inputs)
            self.seq_neg_features = tf.nn.embedding_lookup(self.item_embedding, self.seq_neg_inputs)

            key_masks = tf.sequence_mask(self.seq_mask_inputs, self.max_length)  # [B, T]
            self.unclick_mask = tf.logical_and(tf.logical_not(self.seq_clickmask_input), key_masks)
            # key_masks = tf.sequence_mask(self.seq_mask_inputs, self.max_length)  # [B, T]
            seq_msk = tf.cast(key_masks, tf.float32)
            with tf.name_scope('het_gru'), tf.variable_scope("het_gru", reuse=tf.AUTO_REUSE):
                rnn_outputs, unclick_rnn_outputs \
                    = self.het_gru(self.seq_features,seq_msk, tf.cast(self.seq_clickmask_input, tf.float32), tf.cast(self.seq_likemask_input, tf.float32))

            with tf.name_scope('predict'):
                if self.predict_source == "pos_neg":
                    self.user_emb = self.pos_w * rnn_outputs + (1-self.pos_w) * unclick_rnn_outputs
                elif self.predict_source == "pos_neg_adap":
                    output = tf.layers.dense(tf.concat([rnn_outputs, unclick_rnn_outputs, self.item], axis=1), activation=tf.nn.relu, units=self.dnn_size, use_bias=True, name='fuse1')
                    output = tf.nn.dropout(output, self.keep_prob)
                    output = tf.layers.dense(output, activation=tf.sigmoid, units=1, use_bias=True, name='fuse2')
                    self.user_emb = output * rnn_outputs+(1-output)*unclick_rnn_outputs

                self.joint_output = self.predict_score_new(self.user_emb)

        self.joint_evaoutput = tf.nn.sigmoid(self.joint_output)

        l2_norm = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'embedding' not in v.name])
        self.joint_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.joint_output, labels=self.y_inputs)) + l2_norm * self.lr_reg

        with tf.name_scope('contrastive_loss'), tf.variable_scope("contrastive_loss"):
            def get_prox(hist_input, keys_length):   #[B,L,D] [B]
                key_masks = tf.sequence_mask(keys_length, self.max_length)   #[B,L]
                key_masks = tf.cast(key_masks, tf.float32)
                hist_mean = tf.reduce_sum(hist_input * tf.expand_dims(key_masks, -1), 1) / (tf.reduce_sum(key_masks, 1, keepdims=True)+epsilon)   #[B,D]/[B,1]
                return hist_mean

            click_prox = get_prox(self.seq_pos_features, self.seq_pos_mask_inputs)
            unclick_prox = get_prox(self.seq_neg_features, self.seq_neg_mask_inputs)

            rnn_outputs = self.contras_proj(rnn_outputs, reuse=False)
            unclick_rnn_outputs = self.contras_proj(unclick_rnn_outputs, reuse=True)
            click_prox = self.contras_proj(click_prox, reuse=True)
            unclick_prox = self.contras_proj(unclick_prox, reuse=True)

            user_contras_loss = self.contras_loss_3(click_prox,rnn_outputs, unclick_rnn_outputs, unclick_prox)

            self.joint_loss = self.joint_loss + self.cons_w * user_contras_loss

        for v in tf.trainable_variables():
                logging.info(v.name)

        grads_and_vars = self.optimizer.compute_gradients(self.joint_loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)


    def contras_loss_3(self, click_prox, rnn_outputs, unclick_rnn_outputs, unclick_prox):
        contrastive_length_threshold = self.contrastive_length_threshold
        neg_contrastive_mask = tf.where(
            tf.greater(self.seq_neg_mask_inputs, contrastive_length_threshold),  # [B]
            tf.ones_like(self.seq_neg_mask_inputs, dtype=tf.float32),
            tf.zeros_like(self.seq_neg_mask_inputs, dtype=tf.float32)
        )
        pos_contrastive_mask = tf.where(
            tf.greater(self.seq_pos_mask_inputs, contrastive_length_threshold),  # [B]
            tf.ones_like(self.seq_pos_mask_inputs, dtype=tf.float32),
            tf.zeros_like(self.seq_pos_mask_inputs, dtype=tf.float32)
        )
        contrastive_mask = neg_contrastive_mask * pos_contrastive_mask

        click_prox = tf.nn.l2_normalize(click_prox, axis=-1)  # tf.nn.l2_normalize(, axis=1)
        rnn_outputs = tf.nn.l2_normalize(rnn_outputs, axis=-1)  # tf.nn.l2_normalize(, axis=1)
        unclick_rnn_outputs = tf.nn.l2_normalize(unclick_rnn_outputs, axis=-1)  # tf.nn.l2_normalize(, axis=1)
        unclick_prox = tf.nn.l2_normalize(unclick_prox, axis=-1)  # tf.nn.l2_normalize(, axis=1)
        def get_infonce_score(anchor, pos, neg):
            pos_scores = tf.reduce_sum(tf.multiply(anchor, pos), axis=-1)  # [B]
            neg_scores = tf.reduce_sum(tf.multiply(anchor, neg), axis=-1)  # [B]
            posScore = tf.exp(pos_scores / self.tau)
            negScore = tf.exp(neg_scores / self.tau)
            rate = -tf.log(posScore / (posScore + negScore + epsilon) + epsilon)
            return tf.reduce_sum(contrastive_mask * rate) / (tf.reduce_sum(contrastive_mask) + epsilon)  # 1
        contras_loss1 = get_infonce_score(click_prox, rnn_outputs, unclick_rnn_outputs)
        contras_loss2 = get_infonce_score(rnn_outputs, click_prox, unclick_prox)
        contras_loss3 = get_infonce_score(unclick_rnn_outputs, unclick_prox, click_prox)
        contras_loss4 = get_infonce_score(unclick_prox, unclick_rnn_outputs, rnn_outputs)

        contrastive_loss = (contras_loss1 + contras_loss2 + contras_loss3 + contras_loss4)/4
        return contrastive_loss

    def contras_proj(self, X, reuse):
        output = tf.layers.dense(X, activation=tf.nn.relu, units=self.item_dim, use_bias=True, name='contras_proj1', reuse=reuse)
        output = tf.layers.dense(output, activation=None, units=self.item_dim, use_bias=True, name='contras_proj2', reuse=reuse)
        return output


    def cell_hetgru_forward(self, xt, H_c, H_v, gc_pre,neg_gc_pre, gc, gv, gs):
        n_hidden = self.hidden_size
        r_c = tf.sigmoid(tf.layers.dense(xt, units=n_hidden, use_bias=True, name='xfc') + \
                         tf.multiply(gc_pre, tf.layers.dense(H_v, units=n_hidden, use_bias=False, name='hfc2')) + \
                         tf.multiply(neg_gc_pre, tf.layers.dense(H_c, units=n_hidden, use_bias=False, name='hfc3')))
        z_c = tf.sigmoid(tf.layers.dense(xt, units=n_hidden, use_bias=True, name='xic') + \
                         tf.multiply(gc_pre , tf.layers.dense(H_v, units=n_hidden, use_bias=False, name='hic2')) + \
                         tf.multiply(neg_gc_pre, tf.layers.dense(H_c, units=n_hidden, use_bias=False, name='hic3')))
        g_c = tf.tanh(tf.layers.dense(xt, units=n_hidden, use_bias=True, name='xgc') + \
                      tf.multiply(gc_pre, tf.layers.dense(tf.multiply(r_c, H_v), units=n_hidden, use_bias=False, name='hgc2'))+ \
                      tf.multiply(neg_gc_pre, tf.layers.dense(tf.multiply(r_c, H_c), units=n_hidden, use_bias=False, name='hgc3')))

        H_c = tf.multiply((1 - gs)+gs*gc, H_c) + tf.multiply(gs*(1-gc), tf.multiply(1-z_c, H_c) + tf.multiply(z_c, g_c))

        # click
        r_v = tf.sigmoid(tf.layers.dense(xt, units=n_hidden, use_bias=True, name='xfv') + \
                         tf.multiply(gc_pre, tf.layers.dense(H_v, units=n_hidden, use_bias=False, name='hfv2')) + \
                         tf.multiply(neg_gc_pre, tf.layers.dense(H_c, units=n_hidden, use_bias=False, name='hfv3')))

        z_v = tf.sigmoid(tf.layers.dense(xt, units=n_hidden, use_bias=True, name='xiv') + \
                         tf.multiply(gc_pre, tf.layers.dense(H_v, units=n_hidden, use_bias=False, name='hiv2')) + \
                         tf.multiply(neg_gc_pre, tf.layers.dense(H_c, units=n_hidden, use_bias=False, name='hiv3')))

        g_v = tf.tanh(tf.layers.dense(xt, units=n_hidden, use_bias=True, name='xgv') + \
                      tf.multiply(gc_pre, tf.layers.dense(tf.multiply(r_v, H_v), units=n_hidden, use_bias=False, name='hgv2')) + \
                      tf.multiply(neg_gc_pre, tf.layers.dense(tf.multiply(r_v, H_c), units=n_hidden, use_bias=False, name='hgv3')))

        H_v = tf.multiply(1 - gc, H_v) + tf.multiply(gc, tf.multiply(1-z_v, H_v) + tf.multiply(z_v, g_v))

        # like

        hwz_v = tf.layers.dense(H_v, units=n_hidden, activation=tf.sigmoid, use_bias=True, name='hwz_v')  # 无输入时 transform_gate
        hwh_v = tf.layers.dense(H_v, units=n_hidden, activation=tf.nn.relu, use_bias=True, name='hwh_v')  # 无输入时 transformed_output  activation=tf.nn.relu

        H_v = tf.multiply(1 - gv, H_v) + tf.multiply(gv, tf.multiply(hwz_v, H_v) + tf.multiply(1 - hwz_v, hwh_v))

        return H_c, H_v


    def het_gru(self, inputs, seq_mask, click_mask,like_mask):

        inputs = tf.unstack(inputs, axis=1)  # [max_len,bs,64]
        gc = tf.unstack(click_mask, axis=1)  # [max_len,bs]
        gc = tf.expand_dims(gc, axis=-1)

        gs = tf.unstack(seq_mask, axis=1)  # [max_len,bs]
        gs = tf.expand_dims(gs, axis=-1)  # [max_len,bs,1]

        gv = tf.unstack(like_mask, axis=1)
        gv = tf.expand_dims(gv, axis=-1)

        n_hidden = self.hidden_size

        H_c = tf.zeros(shape=(tf.shape(inputs)[1], n_hidden))  # (bs,hidden)
        H_v = tf.zeros(shape=(tf.shape(inputs)[1], n_hidden))  # (bs,hidden)

        H_c, H_v = self.cell_hetgru_forward(inputs[0], H_c, H_v, 0.0,0.0, gc[0],gv[0],gs[0])
        for i in range(1, self.max_length):
            H_c, H_v = self.cell_hetgru_forward(inputs[i], H_c, H_v, gc[i - 1],1-gc[i - 1], gc[i], gv[i],gs[i])

        return H_v, H_c

    def predict_score_new(self, interest_emb):
        output = tf.layers.dense(tf.concat([interest_emb, self.item], axis=1), activation=tf.nn.relu, units=self.dnn_size, use_bias=True, name='dnn1')
        output = tf.nn.dropout(output, self.keep_prob)
        output = tf.layers.dense(output, activation=None, units=1, use_bias=True, name='dnn2')
        return tf.reshape(output, [-1])
