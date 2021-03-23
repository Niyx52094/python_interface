import torch
import os


USER_COUNT = 2248
ITEM_COUNT = 2475
ATTR_COUNT = 115

KG_RELA_COUNT = 3
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
