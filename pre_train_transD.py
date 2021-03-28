import torch
import random
import numpy as np
from util import *
from openke.module.model import TransD
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from timeit import default_timer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pre_train(train_set,USER_COUNT,ITEM_COUNT,ATTR_COUNT,KG_RELA_COUNT,KG_EMBED_SIZE):
    transd = TransD(
        ent_tot=USER_COUNT + ITEM_COUNT + ATTR_COUNT + 1,
        rel_tot=KG_RELA_COUNT,
        dim_e=KG_EMBED_SIZE,
        dim_r=KG_EMBED_SIZE,
        p_norm=1,
        norm_flag=True)

    cuda_(transd)
    # transd.train()

    optimizer = optim.SGD(transd.parameters(), lr=1.0, weight_decay=0)
    nepoch = 50
    nbatch = 100

    #dataset = load_pikle('./train_set/train_dataset_0.pkl')
    dataset = train_set[0]

    batch_size = len(dataset) // nbatch

    margin = nn.Parameter(torch.Tensor([4.0])).cuda()

    for epoch in range(nepoch):
        s_t = default_timer()

        #dataset = load_pickle('../data/KG_data/yelp/train/train_dataset_{}.pkl'.format(epoch))
        dataset=train_set[epoch]
        random.shuffle(dataset)

        res = 0.0
        transd.train()
        for batch in range(nbatch + 1):
            optimizer.zero_grad()

            lef = batch * batch_size
            rig = min((batch + 1) * batch_size, len(dataset))
            batch_data = list(zip(*dataset[lef: rig]))

            score = transd(
                {'batch_h': Variable(torch.from_numpy(np.array(list(batch_data[0]) + list(batch_data[0]))).cuda()),
                 'batch_r': Variable(torch.from_numpy(np.array(list(batch_data[3]) + list(batch_data[3]))).cuda()),
                 'batch_t': Variable(torch.from_numpy(np.array(list(batch_data[1]) + list(batch_data[2]))).cuda()),
                 'mode': 'normal'})

            loss = (torch.max(score[:rig - lef] - score[rig - lef:], -margin)).mean() + margin

            loss.backward()
            optimizer.step()

            res += loss.item()

        print('Epoch: {}  loss: {} time: {} ...................................................'.format(epoch,res / (nbatch + 1),default_timer() - s_t))
    print("Finish pre_trained by transD,saving model and embedding vector")

    transd.save_checkpoint('../model/transd_embed_ant_trainset.ckpt')