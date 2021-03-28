import argparse
from util import *
import torch.optim as optim
from train import train
# from Recommend.eval import evaluate
import pprint as pp
import os
from pre_train_transD import *
from generate_train_set import *
import time
from Model import *
import config as cfg

def main():
    # ---- get options ----
    parser = argparse.ArgumentParser(description="Run Recommend Module for Conversational Recommender System.")
    # adjust following options according to your needs
    parser.add_argument('--use_gpu', action='store_true',
                        help='whether use gpu during training / testing, = False if not claimed')
    # parser.add_argument('--relation_data', action='store_true',
    #                     help='raw data -> simplified relation data, default save in ./Recommend/data/relation/')
    # parser.add_argument('--trans_data', action='store_true',
    #                     help='relation data -> train set for TransD, default save in ./Recommend/data/trans_data/')
    # parser.add_argument('--prop_data', action='store_true',
    #                     help='relation data -> train/valid/test set for Prop Model, default save in ./Recommend/data/')
    parser.add_argument('--gpu_id', type=str, default="0", help='use which gpu')
    parser.add_argument('--purpose', type=str, default="train", help='train / test')
    # suggest to keep following options as default unless the performance is not good
    parser.add_argument('--prop_n_layer', type=int, default=4, help='propagate how many times for final embedding')
    parser.add_argument('--layer_size', type=int, nargs='*', default=[64, 64, 64],
                        help='output dimension of each layer (layer size of 1st layer = kg embed size)')
    parser.add_argument('--kg_embed_size', type=int, default=64, help='dimension of knowledge graph embedding')
    parser.add_argument('--map', type=int, default=1,
                        help='whether do the projection when calculate item score, 1 = projection, 0 = no projection')
    #import file path
    # parser.add_argument("--entity",type=str,help="import the aboslute path of entity file")
    # parser.add_argument("--relation",type=str,help="import the aboslute path of relation file")
    # parser.add_argument("--pool",type=str,help="import the aboslute path of fund-pool file")
    # parser.add_argument("--history",type=str,help="import the aboslute path of user-history file")
    # parser.add_argument("--attribute",type=str,help="import the aboslute path of attribute file")
    opts = parser.parse_args()
    # process data if needed
    # if opts.relation_data:
    #     pass
    # if opts.trans_data:
    #     pass
    # if opts.prop_data:
    #     pass
    # load following options according to the data / environment / above options
    opts.device = "cuda" if (torch.cuda.is_available() and opts.use_gpu) else "cpu"
    opts.filename = "ant_graph_based_prop_model"


    opts.entity=r'D:\CS\NTU\MSC-Project\pythoninterface\1_entity.txt'
    opts.relation =r'D:\CS\NTU\MSC-Project\pythoninterface\1_relation.txt'
    opts.pool = r'D:\CS\NTU\MSC-Project\pythoninterface\1_pool.txt'
    opts.history = r'D:\CS\NTU\MSC-Project\pythoninterface\1_hist.txt'
    opts.attribute = r'D:\CS\NTU\MSC-Project\pythoninterface\1_attr.txt'
    # print all options
    pp.pprint(vars(opts))
    print(' ')

    if len(opts.layer_size) + 1 != opts.prop_n_layer:
        print('Layer_size & Prop_n_layer Setting Error...')
        exit()
    if opts.device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id

    # ---- create basic kg model ----
    # ---- create propagation model ---
    ### 生成h_t_r 和相应的dict，保存pickle
    generate_htr(opts.entity,opts.relation,opts.pool,opts.history,opts.attribute)
    ### 创建transD预训练

    pre_train(cfg.KG_EMBED_SIZE)

    ### 调用模型获得embedding set
    mlp = Graph_Rec(cfg.KG_EMBED_SIZE, cfg.ProP_Layer_1, cfg.ProP_Layer_2, cfg.ProP_Layer_3, cfg.ProP_n_Layer,
                    cfg.KG_EMBED_SIZE, cfg.MAP)
    if opts.device == 'cuda':
        mlp = cuda_(mlp)
    print(mlp)
    print('')
    # ---- train / test model ----
    if opts.purpose == 'train':
        optimizer = optim.SGD(mlp.parameters(), lr=0.001, weight_decay=0)
        train(mlp, 64, 50, optimizer, 1, 8, opts.filename, opts.purpose)
    # elif opts.purpose == 'test':
    #     PATH = './model/KG_model/' + opts.filename + '.pt'
    #     mlp.load_state_dict(torch.load(PATH, map_location=torch.device(opts.device)))
    #     evaluate(mlp, 50, opts.filename, 0, opts.purpose)
    else:
        print('Purpose Setting Error...')


if __name__ == '__main__':
    main()
