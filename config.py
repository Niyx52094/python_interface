import torch
import pickle
import os
# def load_pickle(path):
#     with open(path,"rb") as f:
#         return pickle.load(f)
#
#
# USER_COUNT = load_pickle('./pickle/USER_COUNT.pkl')
# ITEM_COUNT = load_pickle('./pickle/ITEM_COUNT.pkl')
# ATTR_COUNT = load_pickle('./pickle/ATTR_COUNT.pkl')
#
# KG_RELA_COUNT = load_pickle('./pickle/RELATION_COUNT.pkl')
KG_EMBED_SIZE = 64
KG_USER_ITEM_RELATION_ID = 0

UI_RELATION_ID = 0
IA_RELATION_ID = 1
UA_RELATION_ID = 2


ProP_Layer_1 = 64
ProP_Layer_2 = 64
ProP_Layer_3 = 64
ProP_n_Layer = 4

RC_dr = 0.5
RC_lr = 0.001
seed = 1114

MAP = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
