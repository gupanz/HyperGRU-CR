from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random, logging, os
import pickle
from tqdm import tqdm


class DataLoader(object):
    def __init__(self, params):
        self.data_dir = params.data_dir
        self.batch_size = params.batch_size
        self.max_length = params.max_length
        self.item_num = params.n_items
        self.preload_feat_into_memory()

    def preload_feat_into_memory(self):
        # load dataset
        logging.info('load data.....')

        file_name = "kuairec_dataset.pkl"
        pkl_dir = os.path.join(self.data_dir, file_name)

        with open(pkl_dir, 'rb') as f:
            self.train_interaction_data = pickle.load(f)
            self.val_interaction_data = pickle.load(f)
            self.history_data = pickle.load(f)  #

            self.user_item_cate_seq = pickle.load(f)
            self.user_item_tag_seq = pickle.load(f)

            self.click_mask = pickle.load(f)
            self.like_mask = pickle.load(f)
            self.complete_mask = pickle.load(f)

            random.shuffle(self.train_interaction_data)

            self.epoch_train_length = len(self.train_interaction_data)
            self.epoch_test_length = len(self.val_interaction_data)
            self.user_num = len(self.history_data)

            logging.info('kuairec_dataset.pkl')
            logging.info('{} train samples'.format(self.epoch_train_length))
            logging.info('{} test samples'.format(self.epoch_test_length))
            logging.info('{} user_num'.format(self.user_num))

    def getBatch(self, num, train_test_flag):
        # ["user_id","video_id","validclick","like","watch_ratio","time_span","entire_end_idx","date","timestamp"]
        if train_test_flag == 'train':
            self.epoch_data = self.train_interaction_data
        elif train_test_flag == 'test':
            self.epoch_data = self.val_interaction_data

        user_ids = np.zeros([self.batch_size], dtype=np.int)
        label = np.zeros([self.batch_size], dtype=np.int)
        item_input = np.zeros([self.batch_size], dtype=np.int)

        seq_input = np.zeros([self.batch_size, self.max_length], dtype=np.int)
        seq_mask_input = np.zeros([self.batch_size], dtype=np.int)   # 整个序列的mask，值小于等于max_length
        seq_clickmask_input = np.zeros([self.batch_size, self.max_length], dtype=np.bool)
        seq_likemask_input = np.zeros([self.batch_size, self.max_length], dtype=np.bool)

        # click
        seq_pos_input = np.zeros([self.batch_size, self.max_length], dtype=np.int)    #这个是item的ID
        seq_pos_mask_input = np.zeros([self.batch_size], dtype=np.int)  # 整个序列的mask，值小于等于max_length

        # unclick
        seq_neg_input = np.zeros([self.batch_size, self.max_length], dtype=np.int)
        seq_neg_mask_input = np.zeros([self.batch_size], dtype=np.int)  # 整个序列的mask，值小于等于max_length

        for i in range(num * self.batch_size, (num + 1) * self.batch_size):
            # ["user_id","video_id","validclick","like","watch_ratio","time_span","entire_end_idx","date","timestamp"]
            user_id = self.epoch_data[i][0]
            item_id = self.epoch_data[i][1]
            is_click = 1 if self.epoch_data[i][2] is True else 0
            # is_like = 1 if self.epoch_data[i][3] is True else 0
            # watch_ratio = self.epoch_data[i][4]
            time_span = self.epoch_data[i][5] - 1  # 从1开始的
            end_idx = self.epoch_data[i][6]  # 的end_idx
            # is_complete = 1 if self.epoch_data[i][7] is True else 0

            user_ids[i % self.batch_size] = user_id
            item_input[i % self.batch_size] = item_id
            label[i % self.batch_size] = is_click

            user_seq = self.history_data[user_id][time_span]
            user_click_msk = self.click_mask[user_id][time_span]
            user_like_msk = self.like_mask[user_id][time_span]

            start_idx = max(end_idx - self.max_length, 0)
            seq_pad = self.max_length - (end_idx - start_idx)
            seq_input[i % self.batch_size] = user_seq[start_idx:end_idx].tolist() + [0] * seq_pad
            seq_mask_input[i % self.batch_size] = end_idx - start_idx

            click_msk = user_click_msk[start_idx:end_idx]
            seq_clickmask_input[i % self.batch_size] = click_msk.tolist() + [False] * seq_pad
            seq_likemask_input[i % self.batch_size] = user_like_msk[start_idx:end_idx].tolist() + [False] * seq_pad

            user_click_seq = user_seq[start_idx:end_idx][user_click_msk[start_idx:end_idx]]
            click_seq_len = len(user_click_seq)
            seq_pos_input[i % self.batch_size] = user_click_seq.tolist() + [0] * (self.max_length - click_seq_len)
            seq_pos_mask_input[i % self.batch_size] = click_seq_len
            user_unclick_seq = user_seq[start_idx:end_idx][user_click_msk[start_idx:end_idx] == False]
            unclick_seq_len = len(user_unclick_seq)
            seq_neg_input[i % self.batch_size] = user_unclick_seq.tolist() + [0] * (self.max_length - unclick_seq_len)
            seq_neg_mask_input[i % self.batch_size] = unclick_seq_len

        return user_ids,item_input, label, seq_input, seq_mask_input, seq_clickmask_input, seq_likemask_input, seq_pos_input, seq_pos_mask_input, seq_neg_input, seq_neg_mask_input
