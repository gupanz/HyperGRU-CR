from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Settings(object):
    def __init__(self):
        self.phase = 'train'
        self.model_name = "het_gru"
        self.predict_source = "pos_neg_adap"      # pos_neg_adap pos_neg
        self.pos_w = 0.5
        self.contrastive_length_threshold = 1
        self.cons_w = 0.1
        self.tau = 0.9

        self.n_items = 9239 + 1
        self.n_users = 7176 + 1

        self.batch_size = 1024
        self.max_epoch = 2
        self.display = 200  # print info during training,test

        self.max_length = 200

        self.item_dim = 64
        self.dnn_size = 128
        self.keep_prob = 1.0    # keep_prob  0.8
        self.learning_rate = 0.001
        self.decay_steps = 1000
        self.decay_rate = 0.98
        self.lr_reg = 0
        self.restore = False
        self.out_dir = './data/output'
        self.data_dir = "/data/gp/KuaiRec/newinputs2"
        self.restore_model_dir = './data/output'


