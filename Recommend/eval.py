import pickle
from timeit import default_timer
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import Recommend.config as cfg
import os


def topk(y_true, pred, k):
    y_true_ = y_true[:k]
    pred_ = pred[:k]
    if sum(y_true_) == 0:
        return 0
    else:
        return roc_auc_score(y_true_, pred_)


def rank_by_batch(pickle_file, iter_, bs, pickle_file_length, model_mlp, purpose):
    """
    user_output, item_p_output, i_neg2_output, preference_list = list(), list(), list(), list()
    """
    left, right = iter_ * bs, min(pickle_file_length, (iter_ + 1) * bs)
    if purpose == 'test':
        left, right = 20 * bs + iter_ * bs, min(pickle_file_length, 20 * bs + (iter_ + 1) * bs)

    I = pickle_file[0][left:right]
    II = pickle_file[1][left:right]
    III = pickle_file[2][left:right]
    IV = pickle_file[3][left:right]

    i = 0
    index_none = list()

    for user_output, item_p_output, i_neg2_output, preference_list in zip(I, II, III, IV):
        if i_neg2_output is None or len(i_neg2_output) == 0:
            index_none.append(i)
        i += 1

    i = 0
    result_list = list()
    for user_output, item_p_output, i_neg2_output, preference_list in zip(I, II, III, IV):
        if i in index_none:
            i += 1
            continue

        item_list = list(i_neg2_output)[: 1000] + [item_p_output]
        item_list = torch.tensor(list(np.array(item_list) + cfg.USER_COUNT))
        user_list = torch.tensor([user_output] * len(item_list))
        attr_list = list(np.array(list(preference_list)) + cfg.USER_COUNT + cfg.ITEM_COUNT)
        attr_list = torch.tensor([attr_list] * len(item_list))

        predictions = model_mlp.propagation_eval(user_list, item_list, attr_list).detach()
        predictions = predictions.cpu().numpy()

        mini_gtitems = [item_p_output]
        num_gt = len(mini_gtitems)
        num_neg = len(item_list) - num_gt
        predictions = predictions.reshape((num_neg + 1, 1)[0])
        y_true = [0] * len(predictions)
        y_true[-1] = 1
        tmp = list(zip(y_true, predictions))
        tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
        y_true, predictions = zip(*tmp)
        auc = roc_auc_score(y_true, predictions)

        result_list.append((auc, topk(y_true, predictions, 10), topk(y_true, predictions, 50)
                            , topk(y_true, predictions, 100), topk(y_true, predictions, 200),
                            topk(y_true, predictions, 500), len(predictions)))
        a = topk(y_true, predictions, 10)
        i += 1
    return result_list


def evaluate(model_mlp, epoch, filename, rd, purpose):
    model_mlp.eval_embed()

    s_t = default_timer()
    pickle_file_path = './Recommend/data/KG_data/valid_amazon/v1-speed-valid-1.pickle'
    with open(pickle_file_path, 'rb') as f:
        pickle_file = pickle.load(f)
    print('Open evaluation pickle file: {} takes {} seconds, evaluation length: {}'.format(pickle_file_path,
                                                                                           default_timer() - s_t,
                                                                                           len(pickle_file[0])))
    pickle_file_length = len(pickle_file[0])

    s_t = default_timer()
    print('Evaluating epoch {}'.format(epoch))
    bs = 64
    max_iter = 20

    result = list()
    for iter_ in range(max_iter):
        if iter_ > 1 and iter_ % 50 == 0:
            print('--')
            print('Takes {} seconds to finish {}% of this task'.format(str(default_timer() - s_t),
                                                                       float(iter_) * 100 / max_iter))
        result += rank_by_batch(pickle_file, iter_, bs, pickle_file_length, model_mlp, purpose)

    auc_mean = np.mean(np.array([item[0] for item in result]))
    auc_median = np.median(np.array([item[0] for item in result]))
    print('auc mean: {}'.format(auc_mean), 'auc median: {}'.format(auc_median),
          'over num {}'.format(len(result)))
    auc = np.array([item[0] for item in result])

    PATH = './Recommend/_log/' + filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('validating {} epoch on item prediction\n'.format(epoch))
        auc_mean = np.mean(np.array([item[0] for item in result]))
        auc_median = np.median(np.array([item[0] for item in result]))
        f.write('auc mean: {}\n'.format(auc_mean))
        f.write('auc median: {}\n'.format(auc_median))

    return auc_mean
