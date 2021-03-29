import torch
import random
import numpy as np
from util import *
from openke.module.model import TransD
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from timeit import default_timer
import config as cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pre_train(KG_EMBED_SIZE):
    USER_COUNT=load_pickle("./pickle/USER_COUNT.pkl")
    ITEM_COUNT=load_pickle("./pickle/ITEM_COUNT.pkl")
    ATTR_COUNT=load_pickle("./pickle/ATTR_COUNT.pkl")
    KG_RELA_COUNT=load_pickle("./pickle/RELATION_COUNT.pkl")

    transd = TransD(
        ent_tot=USER_COUNT + ITEM_COUNT + ATTR_COUNT + 1,
        rel_tot=KG_RELA_COUNT+1,
        dim_e=KG_EMBED_SIZE,
        dim_r=KG_EMBED_SIZE,
        p_norm=1,
        norm_flag=True)

    cuda_(transd)
    # transd.train()

    optimizer = optim.SGD(transd.parameters(), lr=1.0, weight_decay=0)
    nepoch = 50
    # nbatch=8 #测试
    nbatch = 100 #实际使用这个

    dataset = load_pickle('./Recommend/data/trans_data/h_pt_nt_r_ant_train_dataset_0.pkl')
    batch_size = len(dataset) // nbatch
    #adjust round of batch
    if(batch_size*nbatch!=len(dataset)):
        nbatch=nbatch+1
    margin = nn.Parameter(torch.Tensor([4.0],device=cfg.device))
    print("pre-trainning TRANSD model.............")

    for epoch in range(nepoch):
        s_t = default_timer()

        dataset = load_pickle('./Recommend/data/trans_data/h_pt_nt_r_ant_train_dataset_{}.pkl'.format(str(epoch)))
        random.shuffle(dataset)

        res = 0.0
        transd.train()
        for batch in range(nbatch):
            optimizer.zero_grad()

            lef = batch * batch_size
            rig = min((batch + 1) * batch_size, len(dataset))
            batch_data = list(zip(*dataset[lef: rig]))
            print(batch_data)
            H=torch.from_numpy(np.array(list(batch_data[0]) + list(batch_data[0])))
            H=H.type(torch.LongTensor)
            # H=H.to(cfg.device)
            H=Variable(H).to(cfg.device)

            R=torch.from_numpy(np.array(list(batch_data[3]) + list(batch_data[3])))
            R=R.type(torch.LongTensor)
            # R=R.to(cfg.device)
            R=Variable(R).to(cfg.device)

            T=torch.from_numpy(np.array(list(batch_data[1]) + list(batch_data[2])))
            T=T.type(torch.LongTensor)
            # T=T.to(cfg.device)
            T=Variable(T).to(cfg.device)

            score = transd(
                {'batch_h':H,
                 'batch_r':R,
                 'batch_t':T,
                 'mode': 'normal'})

            loss = (torch.max(score[:rig - lef] - score[rig - lef:], -margin)).mean() + margin

            loss.backward()
            optimizer.step()

            res += loss.item()

        print('Epoch: {}  loss: {} time: {} ...................................................'.format(epoch,res / (nbatch + 1),default_timer() - s_t))
    transd.save_checkpoint('./model/transd_embed_ant_trainset.ckpt')
    print("Finish pre_trained by transD,saving model and embedding vector")

