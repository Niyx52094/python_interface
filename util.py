import pickle
import torch
import os
import numpy as np

#I/O about files
def load_file(path):
    with open(path,"r",encoding='utf-8') as f:
        file=f.read()
    return file.split("\n")
def save_file(List_file,path):
    if isinstance(List_file,list):
        with open(path,"w") as f:
            for i in range(len(List_file)):
                f.write(str(i)+" "+str(List_file[i])+'\n')
    if isinstance(List_file,dict):
        with open(path,"w") as f:
            for (key,value) in List_file.items():
                f.write(str(key)+' '+str(value)+'\n')

#我这里假设人和item在entity.test里面是混乱排序的，所以把这些id再transfer成从0开始，比如把人的entity_id{1,3,4}转为{0，1，2}
#item同理，entity_id{2,5,6}转为{0，1，2}，之后再用len(人的ids)+item的id来表示最终的item_id。
#之后我会输出这个dict来做对应
def transfor_originial_id_to_new_id(original_id):

    COUNT=len(original_id)
    trans_id=dict()
    index=0
    for i in original_id:
        trans_id[i]=index
        index+=1
    return trans_id
    #I.O about objects
def load_pickle(path):
    with open(path,"rb") as f:
        return pickle.load(f)

def save_pickle(name,path):
    with open(path,"wb") as f:
        pickle.dump(name,f)


def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var

def get_sim(uia_embed):
    size=uia_embed.size(1)
    ui_embed = uia_embed[:, :2, :]
    ia_embed = uia_embed[:, 2:, :]
    i_embed = uia_embed[:,2:(size-2)//2+2,:]
    a_embed = uia_embed[:, (size-2)//2+2:, :]

    # ui_embed[:, :, 64:] = 0
    # ia_embed[:, :, 64:] = 0
    # a_embed[:, :, 64:] = 0

    summed_features_embedding_squared = ui_embed.sum(dim=1, keepdim=True) ** 2
    squared_sum_features_embedding = (ui_embed * ui_embed).sum(dim=1, keepdim=True)
    out = 0.5 * (summed_features_embedding_squared - squared_sum_features_embedding)

    summed_features_embedding_squared_1 = ia_embed.sum(dim=1, keepdim=True) ** 2
    squared_sum_features_embedding_1 = (ia_embed * ia_embed).sum(dim=1, keepdim=True)
    out = out + 0.5 * (summed_features_embedding_squared_1 - squared_sum_features_embedding_1)

    #attr*attr
    summed_features_embedding_squared_2 = a_embed.sum(dim=1, keepdim=True) ** 2
    squared_sum_features_embedding_2 = (a_embed * a_embed).sum(dim=1, keepdim=True)
    out = out - 0.5 * (summed_features_embedding_squared_2 - squared_sum_features_embedding_2)
    #item*item
    summed_features_embedding_squared_3 = i_embed.sum(dim=1, keepdim=True) ** 2
    squared_sum_features_embedding_3 = (i_embed * i_embed).sum(dim=1, keepdim=True)
    out = out - 0.5 * (summed_features_embedding_squared_3 - squared_sum_features_embedding_3)

    out = out.sum(dim=2, keepdim=False)
    return out


def get_sim_w_u(uia_embed):
    ui_embed = uia_embed[:, :2, :]
    # ia_embed = uia_embed[:, 2:, :]
    # a_embed = uia_embed[:, 3:, :]

    # ui_embed[:, :, 64:] = 0

    summed_features_embedding_squared = ui_embed.sum(dim=1, keepdim=True) ** 2
    squared_sum_features_embedding = (ui_embed * ui_embed).sum(dim=1, keepdim=True)
    out = 0.5 * (summed_features_embedding_squared - squared_sum_features_embedding)

    # summed_features_embedding_squared_1 = ia_embed.sum(dim=1, keepdim=True) ** 2
    # squared_sum_features_embedding_1 = (ia_embed * ia_embed).sum(dim=1, keepdim=True)
    # out = out + 0.5 * (summed_features_embedding_squared_1 - squared_sum_features_embedding_1)
    #
    # summed_features_embedding_squared_2 = a_embed.sum(dim=1, keepdim=True) ** 2
    # squared_sum_features_embedding_2 = (a_embed * a_embed).sum(dim=1, keepdim=True)
    # out = out - 0.5 * (summed_features_embedding_squared_2 - squared_sum_features_embedding_2)

    out = out.sum(dim=2, keepdim=False)
    return out


def get_sim_wo_u(uia_embed):
    # ui_embed = uia_embed[:, :2, :]
    ia_embed = uia_embed[:, 2:, :]
    a_embed = uia_embed[:, 3:, :]

    # ia_embed[:, :, 64:] = 0
    # a_embed[:, :, 64:] = 0

    # summed_features_embedding_squared = ui_embed.sum(dim=1, keepdim=True) ** 2
    # squared_sum_features_embedding = (ui_embed * ui_embed).sum(dim=1, keepdim=True)
    # out = 0.5 * (summed_features_embedding_squared - squared_sum_features_embedding)

    summed_features_embedding_squared_1 = ia_embed.sum(dim=1, keepdim=True) ** 2
    squared_sum_features_embedding_1 = (ia_embed * ia_embed).sum(dim=1, keepdim=True)
    out = 0.5 * (summed_features_embedding_squared_1 - squared_sum_features_embedding_1)

    summed_features_embedding_squared_2 = a_embed.sum(dim=1, keepdim=True) ** 2
    squared_sum_features_embedding_2 = (a_embed * a_embed).sum(dim=1, keepdim=True)
    out = out - 0.5 * (summed_features_embedding_squared_2 - squared_sum_features_embedding_2)

    out = out.sum(dim=2, keepdim=False)
    return out
