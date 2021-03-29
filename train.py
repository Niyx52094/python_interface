import time
import torch
# from eval import evaluate
import pickle
import random
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import config as cfg
from util import *
import os



def train(model_mlp, bs, max_epoch, optimizer1, observe, command, filename, purpose):
    model_mlp.train()
    lsigmoid = torch.nn.LogSigmoid()

    USER_COUNT=load_pickle("./pickle/USER_COUNT.pkl")
    ITEM_COUNT=load_pickle("./pickle/ITEM_COUNT.pkl")
    ATTR_COUNT=load_pickle("./pickle/ATTR_COUNT.pkl")
    KG_RELA_COUNT=load_pickle("./pickle/RELATION_COUNT.pkl")

    best_acc = 0

    margin = torch.nn.Parameter(torch.Tensor([4.0],device=cfg.device))

    for epoch in range(max_epoch):
        # _______ Do the evaluation _______
        # if epoch == 0:
        #     print('Evaluating on item similarity')
        #     cur_acc = evaluate(model_mlp, epoch, filename, 0, purpose)

        tt = time.time()
        pickle_file_path = './Recommend/data/train_rec_data/v1-speed-train-{}.pickle'.format(epoch)
        with open(pickle_file_path, 'rb') as f:
            pickle_file = pickle.load(f)
        pickle_file_length = len(pickle_file[0])
        print('Open pickle file: {} takes {} seconds'.format(pickle_file_path, time.time() - tt), '  Length:',
              pickle_file_length)
        model_mlp.train()
        #user,positive_item,neg_item1,neg_item2,attr, 6:relation_between_user_and item,7:relation between item and attr
        mix = list(zip(pickle_file[0], pickle_file[1], pickle_file[2], pickle_file[3], pickle_file[4]
                       , pickle_file[5], pickle_file[6]))
        random.shuffle(mix)
        I, II, III, IV, V, VI, VII= zip(*mix)
        start = time.time()
        print('Starting {} epoch'.format(epoch))
        epoch_loss = 0
        epoch_loss2 = 0
        max_iter = int(pickle_file_length / float(bs))

        for iter_ in range(max_iter):
            if iter_ > 1 and iter_ % 200 == 0:
                print('--')
                print('Takes {} seconds to finish {}% of this epoch'.format(str(time.time() - start),
                                                                            float(iter_) * 100 / max_iter))
                print('loss is: {}'.format(float(epoch_loss) / (bs * iter_)))

            # Optimize Recommendation begin...
            optimizer1.zero_grad()
            left, right = iter_ * bs, min(pickle_file_length, (iter_ + 1) * bs)

            user_list = torch.tensor(I[left:right],dtype=torch.long)
            p_item_list = torch.tensor(np.array(II[left:right]) + USER_COUNT,dtype=torch.long)
            u_i_r_list=torch.tensor(VI[left:right],dtype=torch.long)
            n_item_list = torch.tensor(np.array(III[left:right]) + USER_COUNT,dtype=torch.long)
            attr_list = []
            item_attr_rel_list = []
            n_item_list_2 = []
            index_ = []
            for attr_iter in range(left, right):
                attr_list.append(torch.tensor(np.array(V[attr_iter]) +USER_COUNT + ITEM_COUNT))
                item_attr_rel_list.append(torch.tensor(np.array(VII[attr_iter])))
                if IV[attr_iter] is not None:
                    index_.append(attr_iter - left)
                    n_item_list_2.append(IV[attr_iter])
            attr_list = pad_sequence(attr_list, batch_first=True,
                                     padding_value=USER_COUNT + ITEM_COUNT + ATTR_COUNT)
            attr_list=attr_list.type(torch.LongTensor)
            item_attr_rel_list=pad_sequence(item_attr_rel_list,batch_first=True,
                                        padding_value=KG_RELA_COUNT)
            item_attr_rel_list = item_attr_rel_list.type(torch.LongTensor)
            result_pos = model_mlp.propagation_train(user_list, p_item_list, attr_list,u_i_r_list,item_attr_rel_list)[:, 0]
            result_neg = model_mlp.propagation_train(user_list, n_item_list, attr_list,u_i_r_list,item_attr_rel_list)[:, 0]

            diff1 = (result_pos - result_neg)

            attr_list_2 = attr_list[index_]
            user_list_2 = user_list[index_]
            u_i_r_list_2=u_i_r_list[index_]
            item_attr_rel_list_2=item_attr_rel_list[index_]

            n_item_list_2 = torch.tensor(np.array(n_item_list_2) + USER_COUNT,dtype=torch.long)
            result_pos_2 = result_pos[index_]
            result_neg_2 = model_mlp.propagation_train(user_list_2, n_item_list_2, attr_list_2,u_i_r_list_2,item_attr_rel_list_2)[:, 0]

            diff2 = (result_pos_2 - result_neg_2)

            loss = - lsigmoid(diff1).sum(dim=0) - lsigmoid(diff2).sum(dim=0)

            epoch_loss += loss.data
            loss.backward()
            optimizer1.step()
            # Optimize Recommendation end...

            # Optimize KG begin...
            optimizer1.zero_grad()

            # entity list & relation list start
            user_list = torch.tensor(I[left:right])
            user_list = torch.reshape(user_list, (right - left, 1))
            p_item_list = list(II[left:right])
            ui_relation=torch.tensor(VI[left:right]) #2D dim(B，1）
            # ui_relation = torch.tensor([cfg.KG_USER_ITEM_RELATION_ID] * len(p_item_list))
            ui_relation = torch.reshape(ui_relation, (right - left, 1))
            p_item_list = torch.tensor(np.array(p_item_list) + USER_COUNT)
            p_item_list = torch.reshape(p_item_list, (right - left, 1))

            attr_list = []
            face_list = []#relation between item and attr
            # n_attr_list = []
            # ua_relation = []
            for attr_iter in range(left, right):
                attr_list.append(torch.tensor(np.array(V[attr_iter]) + USER_COUNT + ITEM_COUNT))
                face_list.append(torch.tensor(np.array(VII[attr_iter]))) #2D(B,X)
                # face_list.append(torch.tensor([1] * len(list(V[attr_iter]))))
                # n_attr_list.append(torch.tensor(np.array(VI[attr_iter]) + USER_COUNT + ITEM_COUNT))
                # ua_relation.append(torch.tensor([2] * len(list(VI[attr_iter]))))
            attr_list = pad_sequence(attr_list, batch_first=True, padding_value=USER_COUNT + ITEM_COUNT + ATTR_COUNT)#2D(B,max_len)
            face_list = pad_sequence(face_list, batch_first=True, padding_value=KG_RELA_COUNT)#2D(B,max_len)
            # n_attr_list = pad_sequence(n_attr_list, batch_first=True, padding_value=USER_COUNT+ITEM_COUNT+ATTR_COUNT)
            # ua_relation = pad_sequence(ua_relation, batch_first=True, padding_value=2)

            attr_length = attr_list.size(1)

            p_item_list_1 = p_item_list[:, 0]
            p_item_list_a = list(p_item_list_1.numpy())
            # 2d (B, max_len)
            p_item_list_a = torch.tensor([[item_id for iii in range(attr_list.size(1))] for item_id in p_item_list_a])

            n_item_list = np.array(III[left:right])
            n_item_list = torch.tensor(n_item_list + USER_COUNT)
            n_item_list = torch.reshape(n_item_list, (right - left, 1))

            n_item_list_1 = n_item_list[:, 0]
            n_item_list_a = list(n_item_list_1.numpy())
            #2d (B, max_len)
            n_item_list_a = torch.tensor([[item_id for iii in range(attr_list.size(1))] for item_id in n_item_list_a])

            user_list_a = user_list[:, 0]
            user_list_a = list(user_list_a.numpy())
            user_list_a = torch.tensor([[user_id for iii in range(attr_list.size(1))] for user_id in user_list_a])
            user_list=user_list.long()
            p_item_list=p_item_list.long()
            attr_list=attr_list.long()
            p_item_list_a=p_item_list_a.long()
            n_item_list=n_item_list.long()
            n_item_list_a=n_item_list_a.long()
            ui_relation=ui_relation.long()
            face_list=face_list.long()
            positive_e = torch.cat((user_list, p_item_list, attr_list, p_item_list_a), dim=1)
            negative_e = torch.cat((user_list, n_item_list, attr_list, n_item_list_a), dim=1)
            positive_r = torch.cat((ui_relation, ui_relation, face_list, face_list), dim=1)
            negative_r = torch.cat((ui_relation, ui_relation, face_list, face_list), dim=1)
            # entity list & relation list end

            entity = cuda_(torch.cat((positive_e, negative_e), dim=0))
            relation = cuda_(torch.cat((positive_r, negative_r), dim=0))

            e = model_mlp.kg_model.ent_embeddings(entity)
            r = model_mlp.kg_model.rel_embeddings(relation)

            e_transfer = model_mlp.kg_model.ent_transfer(entity)
            r_transfer = model_mlp.kg_model.rel_transfer(relation)

            e_embed = model_mlp.kg_model._transfer(e, e_transfer, r_transfer)

            h_embed = e_embed[:right - left, 0, :]
            t_embed = e_embed[:right - left, 1, :]
            r_embed = r[:right - left, 0, :]
            for ii in range(attr_length):
                h_embed = torch.cat((h_embed, e_embed[:right - left, 2 + attr_length + ii, :]), dim=0)
                t_embed = torch.cat((t_embed, e_embed[:right - left, 2 + ii, :]), dim=0)
                r_embed = torch.cat((r_embed, r[:right - left, 2 + ii, :]), dim=0)
            # for ii in range(attr_length):
            #     h_embed = torch.cat((h_embed, e_embed[:right-left, 2+3*attr_length+ii, :]), dim=0)
            #     t_embed = torch.cat((t_embed, e_embed[:right-left, 2+2*attr_length+ii, :]), dim=0)
            #     r_embed = torch.cat((r_embed, r[:right-left, 2+2*attr_length+ii, :]), dim=0)
            h_embed = torch.cat((h_embed, e_embed[right - left:, 0, :]), dim=0)
            t_embed = torch.cat((t_embed, e_embed[right - left:, 1, :]), dim=0)
            r_embed = torch.cat((r_embed, r[right - left:, 0, :]), dim=0)
            for ii in range(attr_length):
                h_embed = torch.cat((h_embed, e_embed[right - left:, 2 + attr_length + ii, :]), dim=0)
                t_embed = torch.cat((t_embed, e_embed[right - left:, 2 + ii, :]), dim=0)
                r_embed = torch.cat((r_embed, r[right - left:, 2 + ii, :]), dim=0)
            # for ii in range(attr_length):
            #     h_embed = torch.cat((h_embed, e_embed[right-left:, 2+3*attr_length+ii, :]), dim=0)
            #     t_embed = torch.cat((t_embed, e_embed[right-left:, 2+2*attr_length+ii, :]), dim=0)
            #     r_embed = torch.cat((r_embed, r[right-left:, 2+2*attr_length+ii, :]), dim=0)

            score = model_mlp.kg_model._calc(h_embed, t_embed, r_embed, 'normal')
            loss = (torch.max(score[:int(score.size(0) / 2)] - score[int(score.size(0) / 2):], -margin)).mean() + margin
            epoch_loss2 += loss.data
            loss.backward()
            optimizer1.step()
            # Optimize KG end...

        model_mlp.update_weight()
        print('epoch loss1: {}'.format(epoch_loss / pickle_file_length))
        print('epoch loss2: {}'.format(epoch_loss2 / pickle_file_length))

        PATH = './Recommend/_log/' + filename + '.txt'
        with open(PATH, 'a') as f:
            f.write('Starting {} epoch\n'.format(epoch))
            f.write('training loss 1: {}\n'.format(epoch_loss / pickle_file_length))

        # if epoch % observe == 0 and epoch > -1:
        #     print('Evaluating on item similarity')
        #     cur_acc = evaluate(model_mlp, epoch, filename, 0, purpose)
        #
        #
        ##embedding//
        # if epoch % 1 == 0 and cur_acc > best_acc:
        #     best_acc = cur_acc
    PATH = './model/' + filename + '.pt'
    torch.save(model_mlp.state_dict(), PATH) ##
    print('Model saved at {}'.format(PATH))

    ent_embed=model_mlp.kg_model.ent_embeddings.weight
    rel_embed=model_mlp.kg_model.rel_embeddings.weight
    ent_trans=model_mlp.kg_model.ent_transfer.weight
    rel_trans=model_mlp.kg_model.rel_transfer.weight

    h_embedding=ent_embed
    all_embedding=[h_embedding]
    #加入邻居节点
    for k in range(1, cfg.ProP_n_Layer):
        length = 1000
        current_id = 0
        n_fold = 0
        while current_id < model_mlp.n_user+model_mlp.n_item+model_mlp.n_attr+1:
            if current_id == 0:
                n_embedding = torch.sparse.mm(model_mlp.sp_weight[n_fold].to(cfg.device), h_embedding)
            else:
                n_embedding = torch.cat((n_embedding, torch.sparse.mm(model_mlp.sp_weight[n_fold].to(cfg.device), h_embedding)),
                                        0)
            current_id = current_id + length
            n_fold = n_fold + 1

        add_embedding = h_embedding + n_embedding
        if k == 1:
            h_embedding = torch.nn.functional.relu(model_mlp.fc1(add_embedding))
        elif k == 2:
            h_embedding = torch.nn.functional.relu(model_mlp.fc2(add_embedding))
        else:
            h_embedding = torch.nn.functional.relu(model_mlp.fc3(add_embedding))
        norm_embedding = torch.nn.functional.normalize(h_embedding, 2, -1)
        all_embedding += [norm_embedding]

    user_trans_ids_dict=load_pickle('./pickle/new_user_id_with_original_id.pkl')
    item_trans_ids=load_pickle('./pickle/new_item_id_with_original_item_id.pkl')

    new_ori_user_ids=dict()
    new_ori_item_ids=dict()
    output_embedding_file1=dict()
    output_embedding_file_add_neb=dict()

    for key,value in user_trans_ids_dict.items():
        new_ori_user_ids[value]=key
    for key,value in item_trans_ids.items():
        new_ori_item_ids[value]=key
        # print("new_ori_item_ids: key={} and value={}".format(value,key))

    #get the embeding for entity original id
    for i in range(model_mlp.n_user+model_mlp.n_item+model_mlp.n_attr):
        if i<model_mlp.n_user:
            output_embedding_file1[new_ori_user_ids[i]]=tuple(np.array(ent_embed[i].detach()))
            output_embedding_file_add_neb[new_ori_user_ids[i]] = tuple(list(np.array(all_embedding[0][i].detach()))
                                                                       +list(np.array(all_embedding[1][i].detach()))
                                                                       +list(np.array(all_embedding[2][i].detach()))
                                                                       +list(np.array(all_embedding[3][i].detach())))

        elif i<model_mlp.n_user+model_mlp.n_item:
            output_embedding_file1[new_ori_item_ids[i]] = tuple(np.array(ent_embed[i].detach()))
            output_embedding_file_add_neb[new_ori_item_ids[i]] = tuple(list(np.array(all_embedding[0][i].detach()))
                                                                       +list(np.array(all_embedding[1][i].detach()))
                                                                       +list(np.array(all_embedding[2][i].detach()))
                                                                       +list(np.array(all_embedding[3][i].detach())))
        # else:
        #     print("attr start")
        #     output_embedding_file1[i+1]=tuple(np.array(ent_embed[i].detach()))
        #     output_embedding_file_add_neb[i+1] = tuple(list(np.array(all_embedding[0][i].detach()))
        #                                                                + list(np.array(all_embedding[1][i].detach()))
        #                                                                + list(np.array(all_embedding[2][i].detach()))
        #                                                                + list(np.array(all_embedding[3][i].detach())))
    save_file(output_embedding_file1,'./Recommend/data/output_data/embedding_1.txt')
    print("Successfully save the entity embedding to 'embedding_1.txt'>>>>>>>>>>>>>>>>>>>>>")
    save_file(output_embedding_file_add_neb,'./Recommend/data/output_data/embedding_2.txt')
    print("Successfully save the entity embedding with the neighbour embeding  to 'embedding_2.txt'>>>>>>>>>>>>>>>>>>>>>")

