  # coding=UTF-8
# 配置参数
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import logging
import argparse
import warnings

# from network import Model
# from network_fourlayer import Model
from network_threelayer import Model


from solver import Solver
from data_loader import DataLoader
import config

warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"     #指令的时候删除

def process_args(args, defaults):
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', dest='batch_size', type=int, default=defaults.batch_size)
    parser.add_argument('--max-epoch', dest='max_epoch', type=int, default=defaults.max_epoch)
    parser.add_argument('--learning-rate', dest="learning_rate", type=float, default=defaults.learning_rate)
    parser.add_argument('--max-length', dest='max_length', type=int, default=defaults.max_length)
    parser.add_argument('--item-dim', dest='item_dim', type=int, default=defaults.item_dim)
    parser.add_argument('--dnn-size', dest='dnn_size', type=int, default=defaults.dnn_size)
    parser.add_argument('--lr_reg', dest='lr_reg', type=float, default=defaults.lr_reg)
    parser.add_argument('--model_name', dest='model_name', type=str, default=defaults.model_name)
    parser.add_argument('--n_items', dest='n_items', type=int, default=defaults.n_items)
    parser.add_argument('--n_users', dest='n_users', type=int, default=defaults.n_users)
    parser.add_argument('--contrastive_length_threshold', dest='contrastive_length_threshold', type=int, default=defaults.contrastive_length_threshold)
    parser.add_argument('--pos_w', dest='pos_w', type=float, default=defaults.pos_w)
    parser.add_argument('--cons_w', dest='cons_w', type=float, default=defaults.cons_w)
    parser.add_argument('--tau', dest='tau', type=float, default=defaults.tau)
    parser.add_argument('--restore', dest='restore', type=bool, default=defaults.restore)
    parser.add_argument('--keep_prob', dest='keep_prob', type=float, default=defaults.keep_prob, help='The probability that each element is kept.')
    parser.add_argument('--display', dest='display', type=int, default=defaults.display)
    parser.add_argument('--decay-steps', dest='decay_steps', type=int, default=defaults.decay_steps)
    parser.add_argument('--decay-rate', dest='decay_rate', type=float, default=defaults.decay_rate)
    # parser.add_argument('--interest-dim', dest='interest_dim', type=int, default=defaults.interest_dim)
    parser.add_argument('--predict_source', dest='predict_source', type=str, default=defaults.predict_source)
    parser.add_argument('--out-dir', dest='out_dir', type=str, default=defaults.out_dir)
    parser.add_argument('--data-dir', dest='data_dir', type=str, default=defaults.data_dir)
    parser.add_argument('--restore-model-dir', dest='restore_model_dir', type=str, default=defaults.restore_model_dir)

    parameters = parser.parse_args(args)
    return parameters


def init_logging(params):
    log_name = os.path.join(params.out_dir, '{}.log'.format(time.strftime('%Y%m%d-%H%M')))

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=log_name,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info('batch_size: {}'.format(params.batch_size))
    logging.info('max_epoch: {}'.format(params.max_epoch))
    logging.info('learning_rate: {}'.format(params.learning_rate))
    logging.info('out_dir: {}'.format(params.out_dir))
    logging.info('max_length: {}'.format(params.max_length))
    logging.info('item_dim: {}'.format(params.item_dim))
    logging.info('keep_prob: {}'.format(params.keep_prob))
    logging.info('display: {}'.format(params.display))
    logging.info('decay_steps: {}'.format(params.decay_steps))
    logging.info('decay_rate: {}'.format(params.decay_rate))
    # logging.info('interest_dim: {}'.format(params.interest_dim))
    logging.info('dnn_size: {}'.format(params.dnn_size))
    logging.info('restore: {}'.format(params.restore))

    logging.info('--------------------')


def main(args, param):
    param = process_args(args, param)
    directory = '{}-{}'.format("GRAPHCAPS", time.strftime('%Y%m%d-%H%M%S'))

    param.out_dir = os.path.join(param.out_dir, directory)
    if not os.path.exists(param.out_dir):
        os.makedirs(param.out_dir)

    init_logging(param)
    data = DataLoader(param)
    model = Model(param)
    solver = Solver(model, data, param)

    solver.train()

if __name__ == '__main__':
    main(sys.argv[1:], config.Settings())
