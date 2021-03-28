import config as cfg
import numpy as np
from openke.module.model import TransD
from timeit import default_timer
from util import *
import torch
import os


class Graph_Rec(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2,
                 output_size, n_layer, embed_size, mapping):
        super(Graph_Rec, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = torch.nn.Linear(hidden_size2, output_size)

        self.tanh = torch.nn.Tanh()

        n_user = load_pickle("./pickle/USER_COUNT.pkl")
        n_item = load_pickle("./pickle/ITEM_COUNT.pkl")
        n_attr = load_pickle("./pickle/ATTR_COUNT.pkl")
        n_relation = load_pickle("./pickle/RELATION_COUNT.pkl")

        # 训练一下TransDmoxing
        self.kg_model = TransD(ent_tot=n_user + n_item + n_attr + 1, rel_tot=n_relation+1,
                               dim_e=embed_size, dim_r=embed_size, p_norm=1, norm_flag=True)
        self.kg_model.load_checkpoint('./model/transd_embed_ant_trainset.ckpt')
        self.kg_model = cuda_(self.kg_model)
        print("Createing Model Parameters Done...\n")

        self.n_layer = n_layer

        self.n_user = n_user
        self.n_item = n_item
        self.n_attr = n_attr
        self.n_relation = n_relation
        self.embed_size = embed_size
        self.map = mapping
        print("Saving Data Parameters Done...\n")
        #
        #     #已改
        # self.user_user_dict=load_pickle('./pickle/user_user_dict.pkl')
        # self.reverse_user_user_dict=load_pickle('./pickle/reversed_user_user_dict.pkl')
        #
        # self.user_item_dict = load_pickle('./pickle/user_item_dict.pkl')
        # self.item_user_dict = load_pickle('./pickle/item_user_dict.pkl')
        #
        # self.item_attr_dict = load_pickle('./pickle/item_attr_dict.pkl')
        # self.attr_item_dict = load_pickle('./pickle/attr_item_dict.pkl')

        # import entity and relation,format:dict[i]=tuple(x1,x2)
        self.user_user_rel = load_pickle("./pickle/user_(user,rel)_rel.pkl")
        self.user_item_rel = load_pickle("./pickle/user_(item,rel)_rel.pkl")
        self.item_attr_rel = load_pickle("./pickle/item_(attr,rel)_rel.pkl")

        # import reverse entity and relation format:dict[i]=tuple(x1,-x2)
        self.reverse_user_user_rel = load_pickle('./pickle/reversed_user_(user,rel)_dict.pkl')
        self.item_user_rel = load_pickle('./pickle/item_(user,rel)_dict.pkl')
        self.attr_item_rel = load_pickle('./pickle/attr_(item,rel)_dict.pkl')

        # 修改 变为user-user,user_item,item-attr
        self.kg_pair = []
        self.kg_relation = []
        self.kg_direction=[]
        for i in range(n_user + n_item + n_attr):
            if i < n_user:
                # self.kg_pair.append(list(n_user + np.array(self.user_item_dict[i])) + list(
                #     n_user + n_item + np.array(self.user_attr_dict[i])))
                # 改成user—user 和user-item
                user1, user2, item, relation1, relation2, relation3 = [], [], [], [], [], []
                if len(self.user_user_rel[i]) != 0:
                    user1, relation1 = zip(*self.user_user_rel[i])
                if len(self.reverse_user_user_rel[i]) != 0:
                    user2, relation2 = zip(*self.reverse_user_user_rel[i])
                if len(self.user_item_rel[i]) != 0:
                    item, relation3 = zip(*self.user_item_rel[i])
                # generate kg
                self.kg_pair.append(list(user1) + list(user2) + list(np.array(item) + n_user))
                self.kg_relation.append(list(relation1) + list(relation2) + list(relation3))
                self.kg_direction.append([1]*len(list(relation1))+[-1]*len(list(relation2))+[1]*len(list(relation3)))
            elif i < n_user + n_item:
                # item-user and iterm-attr
                attr1, user1, relation1, relation2 = [], [], [], []
                if len(self.item_attr_rel[i - n_user]) != 0:
                    attr1, relation1 = zip(*self.item_attr_rel[i - n_user])
                if len(self.item_user_rel[i - n_user]) != 0:
                    user1, relation2 = zip(*self.item_user_rel[i - n_user])

                # generate kg
                self.kg_pair.append(list(np.array(attr1) + n_user + n_item) + list(user1))
                self.kg_relation.append(list(relation1) + list(relation2))
                self.kg_direction.append(
                    [1] * len(list(relation1)) + [-1] * len(list(relation2)) )
            else:
                #
                # self.kg_pair.append(
                #     list(n_user + np.array(self.attr_item_dict[i - n_user - n_item])) + self.attr_user_dict[
                #         i - n_user - n_item])

                item1, relation1 = [], []
                if len(self.attr_item_rel[i - n_user - n_item]) != 0:
                    item1, relation1 = zip(*self.attr_item_rel[i - n_user - n_item])

                # generate kg
                self.kg_pair.append(list(np.array(item1) + n_user))
                self.kg_relation.append(list(relation1))
                self.kg_direction.append([-1] * len(list(relation1)))
                # self.kg_pair.append(
                #     list(n_user + np.array(self.attr_item_dict[i - n_user - n_item])))
        self.kg_pair.append([])
        self.kg_relation.append([])
        self.kg_direction.append([])
        print("Loading (h, r, t) Done...\n")

        self.sp_weight = dict()
        self.update_weight()
        self.all_embedding_eval = []
        torch.cuda.empty_cache()
        print("Initialize Weight Done...\n")

    def update_weight(self):
        all_id = torch.tensor([ii for ii in range(self.n_user + self.n_item + self.n_attr + 1)],
                              device=cfg.device)
        all_rel = torch.tensor([iii for iii in range(self.n_relation + 1)], device=cfg.device)

        all_id_embedding = self.kg_model.ent_embeddings(all_id).detach()
        all_id_transfer = self.kg_model.ent_embeddings(all_id).detach()
        all_rel_embedding = self.kg_model.rel_embeddings(all_rel).detach()
        all_rel_transfer = self.kg_model.rel_embeddings(all_rel).detach()
        self.sp_weight = dict()
        length = 1000
        current_id = 0
        n_fold = 0

        while current_id < self.n_user + self.n_item + self.n_attr + 1:
            row = []
            col = []
            weight = []
            right_id = min(current_id + length, self.n_user + self.n_item + self.n_attr + 1)

            for i in range(current_id, right_id):

                h_embedding = all_id_embedding[i]  # dim(1,Embedding_size)
                h_embedding=h_embedding.repeat(len(self.kg_pair[i]),1)# dim (len,Emebed_size)
                h_transfer=all_id_transfer[i]
                h_transfer=h_transfer.repeat(len(self.kg_pair[i]),1)# dim (len,Emebed_size)


                t_embedding = all_id_embedding[self.kg_pair[i]]  # dim (len,Emebed_size)
                t_transfer = all_id_transfer[self.kg_pair[i]]

                r_embedding = all_rel_embedding[self.kg_relation[i]]  # dim (len,Emebed_size)
                r_embedding = torch.nn.functional.normalize(r_embedding, 2, -1).detach()


                r_transfer=all_rel_transfer[self.kg_relation[i]]

                h_embedding = self.kg_model._transfer(h_embedding, h_transfer, r_transfer).detach()
                h_embedding = torch.nn.functional.normalize(h_embedding, 2, -1).detach()# dim (len,Emebed_size)

                t_embedding=self.kg_model._transfer(t_embedding,t_transfer,r_transfer).detach()
                t_embedding = torch.nn.functional.normalize(t_embedding, 2, -1).detach()

                r_direction=torch.tensor(self.kg_direction[i],device=cfg.device,dtype=torch.float32) #1Xlen
                r_direction=torch.reshape(r_direction,(1,-1))
                r_direction=r_direction.t() # len X 1
                r_embedding=r_direction.repeat(1,self.embed_size)*r_embedding #len X E
                t_embedding=self.tanh(t_embedding-r_embedding)

                h_embedding=torch.reshape(h_embedding,(h_embedding.shape[0],1,h_embedding.shape[1])) #len X 1 X E
                t_embedding=torch.reshape(t_embedding,(t_embedding.shape[0],t_embedding.shape[1],1)) #len X E X 1

                #update weight
                col = col + self.kg_pair[i]
                row = row + [i - current_id] * len(self.kg_pair[i])

                weight = weight + torch.nn.functional.softmax(
                    torch.reshape(torch.bmm(h_embedding, t_embedding),(-1,1)).t()[0],dim=0).tolist()   #1 X Len

            i = torch.LongTensor([row, col])
            v = torch.FloatTensor(weight)

            self.sp_weight[n_fold] = torch.sparse.FloatTensor(i, v, torch.Size(
                [right_id - current_id, self.n_user + self.n_item + self.n_attr + 1]))

            current_id = current_id + length
            n_fold = n_fold + 1

    def propagation_train(self, user, item, attr, ui_relation, ia_relation):
        all_id = torch.tensor([ii for ii in range(self.n_user + self.n_item + self.n_attr + 1)], device=cfg.device)
        h_embedding = self.kg_model.ent_embeddings(all_id)

        all_embedding = [h_embedding]
        n_embedding = []
        embed = []

        for k in range(1, cfg.ProP_n_Layer):
            length = 1000
            current_id = 0
            n_fold = 0
            while current_id < self.n_user:
                if current_id == 0:
                    n_embedding = torch.sparse.mm(self.sp_weight[n_fold].cuda(), h_embedding)
                else:
                    n_embedding = torch.cat((n_embedding, torch.sparse.mm(self.sp_weight[n_fold].cuda(), h_embedding)),
                                            0)
                current_id = current_id + length
                n_fold = n_fold + 1

            current_id = self.n_user
            while current_id < self.n_user + self.n_item:
                n_embedding = torch.cat((n_embedding, torch.sparse.mm(self.sp_weight[n_fold].cuda(), h_embedding)), 0)
                current_id = current_id + length
                n_fold = n_fold + 1

            current_id = self.n_user + self.n_item
            while current_id < self.n_user + self.n_item + self.n_attr:
                n_embedding = torch.cat((n_embedding, torch.sparse.mm(self.sp_weight[n_fold].cuda(), h_embedding)), 0)
                current_id = current_id + length
                n_fold = n_fold + 1

            add_embedding = h_embedding + n_embedding
            if k == 1:
                h_embedding = torch.nn.functional.relu(self.fc1(add_embedding))
            elif k == 2:
                h_embedding = torch.nn.functional.relu(self.fc2(add_embedding))
            else:
                h_embedding = torch.nn.functional.relu(self.fc3(add_embedding))
            norm_embedding = torch.nn.functional.normalize(h_embedding, 2, -1)
            all_embedding += [norm_embedding]

        ##修改
        # ui_relation = torch.tensor([cfg.UI_RELATION_ID] * len(item))
        # ia_relation = torch.tensor([cfg.IA_RELATION_ID] * len(item))
        # ai_relation = torch.tensor([[cfg.IA_RELATION_ID] * attr.size(1)] * attr.size(0))

        u_transfer = self.kg_model.ent_transfer(cuda_(user))  # dim(user_len,embedin_len)
        ui_r_transfer = self.kg_model.rel_transfer(cuda_(ui_relation))  # dim(item_lsit_len,embedin_len)
        i_transfer = self.kg_model.ent_transfer(cuda_(item))  # dim(item_list_len,embedin_len)
        ia_r_transfer = self.kg_model.rel_transfer(
            cuda_(ia_relation))  # 3D dim(item_list_len,user+item_attr_len,embedding)
        a_transfer = self.kg_model.ent_transfer(cuda_(attr))  # 3D dim(item_list_len,user+item_attr_len,embedding)
        ai_r_transfer = self.kg_model.rel_transfer(cuda_(ia_relation))

        for k in range(cfg.ProP_n_Layer):
            user_embed = all_embedding[k][user]
            item_embed = all_embedding[k][item]
            attr_embed = all_embedding[k][attr]

            if self.map == 0:
                user_embed = torch.reshape(user_embed, (user_embed.size(0), 1, user_embed.size(1)))

                item_embed = torch.reshape(item_embed, (item_embed.size(0), 1, item_embed.size(1)))
                item_embed_u = item_embed
                item_embed_a = item_embed

            else:
                user_embed = self.kg_model._transfer(user_embed, u_transfer, ui_r_transfer)
                user_embed = torch.nn.functional.normalize(user_embed, 2, -1)
                user_embed = torch.reshape(user_embed, (user_embed.size(0), 1, user_embed.size(1)))

                item_embed_u = self.kg_model._transfer(item_embed, i_transfer, ui_r_transfer)
                item_embed_u = torch.nn.functional.normalize(item_embed_u, 2, -1)
                item_embed_u = torch.reshape(item_embed_u, (item_embed_u.size(0), 1, item_embed_u.size(1)))

                item_embed = torch.reshape(item_embed, (item_embed.size(0), 1, item_embed.size(1)))
                # expand to 3D
                item_embed = item_embed.repeat(1, ia_r_transfer.size(1), 1)
                i_transfer = i_transfer.repeat(1, ia_r_transfer.size(1), 1)
                item_embed_a = self.kg_model._transfer(item_embed, i_transfer, ia_r_transfer)
                item_embed_a = torch.nn.functional.normalize(item_embed_a, 2, -1)
                # item_embed_a = torch.reshape(item_embed_a, (item_embed_a.size(0), 1, item_embed_a.size(1)))

                attr_embed = self.kg_model._transfer(attr_embed, a_transfer, ai_r_transfer)
                attr_embed = torch.nn.functional.normalize(attr_embed, 2, -1)

            temp = torch.cat((user_embed, item_embed_u, item_embed_a, attr_embed), dim=1)

            if k == 0:
                embed = temp
            else:
                embed = torch.cat((embed, temp), dim=2)

        score = get_sim(embed)
        return score

    def propagation_eval(self, user, item, attr):
        embed = []
        ui_relation = torch.tensor([cfg.UI_RELATION_ID] * len(item))
        ia_relation = torch.tensor([cfg.IA_RELATION_ID] * len(item))
        ai_relation = torch.tensor([[cfg.IA_RELATION_ID] * attr.size(1)] * attr.size(0))

        for k in range(cfg.ProP_n_Layer):
            user_embed = self.all_embedding_eval[k][user]
            item_embed = self.all_embedding_eval[k][item]
            attr_embed = self.all_embedding_eval[k][attr]

            if self.map == 0:
                user_embed = torch.reshape(user_embed, (user_embed.size(0), 1, user_embed.size(1)))

                item_embed = torch.reshape(item_embed, (item_embed.size(0), 1, item_embed.size(1)))
                item_embed_u = item_embed
                item_embed_a = item_embed
            else:
                u_transfer = self.kg_model.ent_transfer(cuda_(user)).detach()
                r_transfer = self.kg_model.rel_transfer(cuda_(ui_relation)).detach()
                user_embed = self.kg_model._transfer(user_embed, u_transfer, r_transfer).detach()
                user_embed = torch.nn.functional.normalize(user_embed, 2, -1)
                user_embed = torch.reshape(user_embed, (user_embed.size(0), 1, user_embed.size(1)))

                i_transfer = self.kg_model.ent_transfer(cuda_(item)).detach()
                item_embed_u = self.kg_model._transfer(item_embed, i_transfer, r_transfer).detach()
                item_embed_u = torch.nn.functional.normalize(item_embed_u, 2, -1)
                item_embed_u = torch.reshape(item_embed_u, (item_embed_u.size(0), 1, item_embed_u.size(1)))
                r_transfer = self.kg_model.rel_transfer(cuda_(ia_relation)).detach()
                item_embed_a = self.kg_model._transfer(item_embed, i_transfer, r_transfer).detach()
                item_embed_a = torch.nn.functional.normalize(item_embed_a, 2, -1)
                item_embed_a = torch.reshape(item_embed_a, (item_embed_a.size(0), 1, item_embed_a.size(1)))

                a_transfer = self.kg_model.ent_transfer(cuda_(attr)).detach()
                r_transfer = self.kg_model.rel_transfer(cuda_(ai_relation)).detach()
                attr_embed = self.kg_model._transfer(attr_embed, a_transfer, r_transfer).detach()
                attr_embed = torch.nn.functional.normalize(attr_embed, 2, -1)

            temp = torch.cat((user_embed, item_embed_u, item_embed_a, attr_embed), dim=1)

            if k == 0:
                embed = temp
            else:
                embed = torch.cat((embed, temp), dim=2)

        score = get_sim(embed)
        return score

    def eval_embed(self):
        all_id = torch.tensor([ii for ii in range(self.n_user + self.n_item + self.n_attr + 1)], device=cfg.device)
        h_embedding = self.kg_model.ent_embeddings(all_id).detach()

        self.all_embedding_eval = [h_embedding]
        n_embedding = []

        for k in range(1, cfg.ProP_n_Layer):
            length = 1000
            current_id = 0
            n_fold = 0
            while current_id < self.n_user:
                if current_id == 0:
                    n_embedding = torch.sparse.mm(self.sp_weight[n_fold].cuda(), h_embedding)
                else:
                    n_embedding = torch.cat((n_embedding, torch.sparse.mm(self.sp_weight[n_fold].cuda(), h_embedding)),
                                            0)
                current_id = current_id + length
                n_fold = n_fold + 1

            current_id = self.n_user
            while current_id < self.n_user + self.n_item:
                n_embedding = torch.cat((n_embedding, torch.sparse.mm(self.sp_weight[n_fold].cuda(), h_embedding)), 0)
                current_id = current_id + length
                n_fold = n_fold + 1

            current_id = self.n_user + self.n_item
            while current_id < self.n_user + self.n_item + self.n_attr:
                n_embedding = torch.cat((n_embedding, torch.sparse.mm(self.sp_weight[n_fold].cuda(), h_embedding)), 0)
                current_id = current_id + length
                n_fold = n_fold + 1

            add_embedding = h_embedding + n_embedding
            if k == 1:
                h_embedding = torch.nn.functional.relu(self.fc1(add_embedding)).detach()
            elif k == 2:
                h_embedding = torch.nn.functional.relu(self.fc2(add_embedding)).detach()
            else:
                h_embedding = torch.nn.functional.relu(self.fc3(add_embedding)).detach()
            norm_embedding = torch.nn.functional.normalize(h_embedding, 2, -1)
            self.all_embedding_eval += [norm_embedding]

    def get_embedding(self, e_id, r_id):
        e = self.kg_model.ent_embeddings(e_id)

        e_transfer = self.kg_model.ent_transfer(e_id)
        r_transfer = self.kg_model.rel_transfer(r_id)

        e = self.kg_model._transfer(e, e_transfer, r_transfer)
        e = torch.nn.functional.normalize(e, 2, -1)

        return e

    def propagation_eval_attr_update(self, user, candidate, attr, current_sp_weight, neg_set, flag):
        # update weight begin
        if len(neg_set) == 0 or flag is True:
            sp_weight = current_sp_weight
        else:
            sp_weight = dict()

            length = 1000
            current_id = 0
            n_fold = 0
            while current_id < self.n_user:
                i = current_sp_weight[n_fold]._indices()
                v = current_sp_weight[n_fold]._values()
                right_id = min(current_id + length, self.n_user)

                ind_set = None
                for neg_node in neg_set:
                    ind_row = (i[0] == neg_node - current_id)
                    ind_col = (i[1] == neg_node)
                    if ind_set is None:
                        ind_set = ind_row | ind_col
                    else:
                        ind_set = ind_set | ind_row | ind_col
                ind_set = ~ind_set
                ind_set = ind_set.float()

                v = v * ind_set

                sp_weight[n_fold] = torch.sparse.FloatTensor(i, v, torch.Size(
                    [right_id - current_id, self.n_user + self.n_item + self.n_attr + 1]))

                current_id = current_id + length
                n_fold = n_fold + 1

            current_id = self.n_user
            while current_id < self.n_user + self.n_item:
                i = current_sp_weight[n_fold]._indices()
                v = current_sp_weight[n_fold]._values()
                right_id = min(current_id + length, self.n_user + self.n_item)

                ind_set = None
                for neg_node in neg_set:
                    ind_row = (i[0] == neg_node - current_id)
                    ind_col = (i[1] == neg_node)
                    if ind_set is None:
                        ind_set = ind_row | ind_col
                    else:
                        ind_set = ind_set | ind_row | ind_col
                ind_set = ~ind_set
                ind_set = ind_set.float()

                v = v * ind_set

                sp_weight[n_fold] = torch.sparse.FloatTensor(i, v, torch.Size(
                    [right_id - current_id, self.n_user + self.n_item + self.n_attr + 1]))

                current_id = current_id + length
                n_fold = n_fold + 1

            current_id = self.n_user + self.n_item
            while current_id < self.n_user + self.n_item + self.n_attr:
                i = current_sp_weight[n_fold]._indices()
                v = current_sp_weight[n_fold]._values()
                right_id = min(current_id + length, self.n_user + self.n_item + self.n_attr + 1)

                ind_set = None
                for neg_node in neg_set:
                    ind_row = (i[0] == neg_node - current_id)
                    ind_col = (i[1] == neg_node)
                    if ind_set is None:
                        ind_set = ind_row | ind_col
                    else:
                        ind_set = ind_set | ind_row | ind_col
                ind_set = ~ind_set
                ind_set = ind_set.float()

                v = v * ind_set

                sp_weight[n_fold] = torch.sparse.FloatTensor(i, v, torch.Size(
                    [right_id - current_id, self.n_user + self.n_item + self.n_attr + 1]))

                current_id = current_id + length
                n_fold = n_fold + 1
        # update weight end

        # get embedding based on current weight begin
        if flag is True:
            all_embedding_eval = self.all_embedding_eval
        else:
            all_id = torch.tensor([ii for ii in range(self.n_user + self.n_item + self.n_attr + 1)], device=cfg.device)
            h_embedding = self.kg_model.ent_embeddings(all_id).detach()

            all_embedding_eval = [h_embedding]
            n_embedding = []

            for k in range(1, cfg.ProP_n_Layer):
                length = 1000
                current_id = 0
                n_fold = 0
                while current_id < self.n_user:
                    if current_id == 0:
                        n_embedding = torch.sparse.mm(self.sp_weight[n_fold].cuda(), h_embedding)
                    else:
                        n_embedding = torch.cat(
                            (n_embedding, torch.sparse.mm(self.sp_weight[n_fold].cuda(), h_embedding)),
                            0)
                    current_id = current_id + length
                    n_fold = n_fold + 1

                current_id = self.n_user
                while current_id < self.n_user + self.n_item:
                    n_embedding = torch.cat((n_embedding, torch.sparse.mm(self.sp_weight[n_fold].cuda(), h_embedding)),
                                            0)
                    current_id = current_id + length
                    n_fold = n_fold + 1

                current_id = self.n_user + self.n_item
                while current_id < self.n_user + self.n_item + self.n_attr:
                    n_embedding = torch.cat((n_embedding, torch.sparse.mm(self.sp_weight[n_fold].cuda(), h_embedding)),
                                            0)
                    current_id = current_id + length
                    n_fold = n_fold + 1

                add_embedding = h_embedding + n_embedding
                if k == 1:
                    h_embedding = torch.nn.functional.relu(self.fc1(add_embedding)).detach()
                elif k == 2:
                    h_embedding = torch.nn.functional.relu(self.fc2(add_embedding)).detach()
                else:
                    h_embedding = torch.nn.functional.relu(self.fc3(add_embedding)).detach()
                norm_embedding = torch.nn.functional.normalize(h_embedding, 2, -1)
                all_embedding_eval += [norm_embedding]

            self.all_embedding_eval = all_embedding_eval
            self.sp_weight = sp_weight

        # get embedding based on current weight end

        # calculate score begin
        embed = []
        ua_relation = torch.tensor([cfg.UA_RELATION_ID] * len(candidate))
        ia_relation = torch.tensor([cfg.IA_RELATION_ID] * len(candidate))
        ai_relation = torch.tensor([[cfg.IA_RELATION_ID] * attr.size(1)] * attr.size(0))

        for k in range(cfg.ProP_n_Layer):
            user_embed = all_embedding_eval[k][user]
            candidate_embed = all_embedding_eval[k][candidate]
            attr_embed = all_embedding_eval[k][attr]

            if self.map == 0:
                user_embed = torch.reshape(user_embed, (user_embed.size(0), 1, user_embed.size(1)))

                candidate_embed = torch.reshape(candidate_embed, (candidate_embed.size(0), 1, candidate_embed.size(1)))
                candidate_embed_u = candidate_embed
                candidate_embed_a = candidate_embed
            else:
                u_transfer = self.kg_model.ent_transfer(cuda_(user)).detach()
                r_transfer = self.kg_model.rel_transfer(cuda_(ua_relation)).detach()
                user_embed = self.kg_model._transfer(user_embed, u_transfer, r_transfer).detach()
                user_embed = torch.nn.functional.normalize(user_embed, 2, -1)
                user_embed = torch.reshape(user_embed, (user_embed.size(0), 1, user_embed.size(1)))

                i_transfer = self.kg_model.ent_transfer(cuda_(candidate)).detach()
                candidate_embed_u = self.kg_model._transfer(candidate_embed, i_transfer, r_transfer).detach()
                candidate_embed_u = torch.nn.functional.normalize(candidate_embed_u, 2, -1)
                candidate_embed_u = torch.reshape(candidate_embed_u,
                                                  (candidate_embed_u.size(0), 1, candidate_embed_u.size(1)))
                r_transfer = self.kg_model.rel_transfer(cuda_(ia_relation)).detach()
                candidate_embed_a = self.kg_model._transfer(candidate_embed, i_transfer, r_transfer).detach()
                candidate_embed_a = torch.nn.functional.normalize(candidate_embed_a, 2, -1)
                candidate_embed_a = torch.reshape(candidate_embed_a,
                                                  (candidate_embed_a.size(0), 1, candidate_embed_a.size(1)))

                a_transfer = self.kg_model.ent_transfer(cuda_(attr)).detach()
                r_transfer = self.kg_model.rel_transfer(cuda_(ai_relation)).detach()
                attr_embed = self.kg_model._transfer(attr_embed, a_transfer, r_transfer).detach()
                attr_embed = torch.nn.functional.normalize(attr_embed, 2, -1)

            temp = torch.cat((user_embed, candidate_embed_u, candidate_embed_a, attr_embed), dim=1)

            if k == 0:
                embed = temp
            else:
                embed = torch.cat((embed, temp), dim=2)

        score = get_sim(embed).detach().cpu().reshape(-1)
        score = torch.nn.functional.normalize(score, 2, -1)

        score_w_u = get_sim_w_u(embed).detach().cpu().reshape(-1)
        score_w_u = torch.nn.functional.normalize(score_w_u, 2, -1)

        score_wo_u = get_sim_wo_u(embed).detach().cpu().reshape(-1)
        score_wo_u = torch.nn.functional.normalize(score_wo_u, 2, -1)
        # calculate score end

        return score.tolist(), score_w_u.tolist(), score_wo_u.tolist(), sp_weight
