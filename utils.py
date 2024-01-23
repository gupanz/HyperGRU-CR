import os
import scipy
import numpy as np
import tensorflow as tf
import time

# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)


def get_shape(inputs, name=None):
    name = "shape" if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return (shape)


def evaluation(pred_dict,top_k=0):  # pred_dict[user] = y_pred, y_label
    precisions, recalls, ndcgs, auc_lst = [], [], [], []

    pos_cnt = 0

    for key in pred_dict:
        preds = pred_dict[key]
        preds.sort(key=lambda x: x[0], reverse=True)
        preds = np.array(preds)
        pos_num = sum(preds[:, 1])
        neg_num = len(preds) - pos_num

        if pos_num == 0 or neg_num == 0:
            pos_cnt += 1
            continue

        # precision and recall

        precisions.append([sum(preds[:top_k, 1]) / min(top_k, len(preds)), len(preds)])
        # precisions.append([sum(preds[:top_k, 1]) / top_k, len(preds)])
        recalls.append([sum(preds[:top_k, 1]) / sum(preds[:, 1]), len(preds)])

        # ndcg
        pos_idx = np.where(preds[:top_k, 1] == 1)[0]
        dcg = np.sum(np.log(2) / np.log(2 + pos_idx))
        idcg = np.sum(np.log(2) / np.log(2 + np.arange(len(pos_idx))))
        ndcg = dcg / (idcg + 1e-8)
        ndcgs.append([ndcg, len(preds[:top_k])])

        # auc
        pos_count, neg_count = 0, 0
        for i in range(len(preds)):
            if preds[i, 1] == 0:
                neg_count += (pos_num - pos_count)
            else:
                pos_count += 1
            if pos_count == pos_num:
                auc = 1 - (neg_count / (pos_num * neg_num))
                auc_lst.append([auc, len(preds), key])
                break

    print("过滤掉的用户数量：pos_cnt: {}".format(pos_cnt))
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    auc_lst = np.array(auc_lst)
    ndcgs = np.array(ndcgs)


    p_unweight = np.mean(precisions[:, 0])
    r_unweight = np.mean(recalls[:, 0])
    f_unweight = 2 * p_unweight * r_unweight / (p_unweight + r_unweight)
    auc_unweight = np.mean(auc_lst[:, 0])
    ndcg_unweight = np.mean(ndcgs[:, 0])

    return p_unweight, r_unweight, f_unweight,auc_unweight,ndcg_unweight


