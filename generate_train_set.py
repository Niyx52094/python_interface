from util import *
import random
import numpy as np


#(h,r,t)->(user,item,rel),(user,user,rel),(item,attr,rel)
def generate_htr(entity_path=None,relation_path=None,pool_path=None,user_hist=None,attribute_path=None):

    ENTITY_COUNT=0
    ATTR_COUNT=0
    POOL_COUNT=0
    HIST_COUNT=0
    RELATION_LIST_COUNT=0
    #import coresponding files
    if relation_path!=None:
        relation_list=load_file(relation_path)
        RELATION_LIST_COUNT=len(relation_list)
    if entity_path!=None:
        entity_list=load_file(entity_path)
        ENTITY_COUNT = len(entity_list)
    if pool_path!=None:
        pool_list=load_file(pool_path)
        POOL_COUNT=len(pool_list)
    if user_hist!=None:
        hist_list=load_file(user_hist)
        HIST_COUNT=len(hist_list)
    if attribute_path!=None:
        attr_list=load_file(attribute_path)
        ATTR_COUNT = len(attr_list)

    user_ids=[]
    item_ids=[]
    attr_ids=[]
    user_user_rel_ids=[]
    user_item_rel_ids=[]
    item_attr_rel_ids=[]
    for i in range(len(entity_list)):
        temp=entity_list[i].split(" ")
        if temp[1]=="人":
            user_ids.append(int(temp[0]))
        else:
            item_ids.append(int(temp[0]))
    USER_COUNT=len(user_ids)
    ITEM_COUNT=len(item_ids)
    user_trans_ids_dict= transfor_originial_id_to_new_id(user_ids)
    item_trans_ids_dict = transfor_originial_id_to_new_id(item_ids)

    output_item_trans_id=dict()
    for key in item_trans_ids_dict.keys():
        output_item_trans_id[key]=item_trans_ids_dict[key]+USER_COUNT
    save_file(user_trans_ids_dict,'./new_user_id_with_original_id.txt')
    save_file(output_item_trans_id, './new_item_id_with_original_item_id.txt')

    #relation.txt
    uu_rel_and_ui_rel=[]
    uu_ui_rel=dict()
    visited_objs=[]
    index=0
    for i in range(RELATION_LIST_COUNT):
        temp=relation_list[i].split(' ')
        # print(temp)
        left_id=int(temp[0])
        right_id=int(temp[1])

        direction=int(temp[-1])
        #relation name and its id, keep it unique
        if uu_ui_rel.__contains__(temp[3])==False:
            uu_ui_rel[temp[3]]=index

            if user_trans_ids_dict.__contains__(left_id) and user_trans_ids_dict.__contains__(right_id):
                user_user_rel_ids.append(index)
            else:
                user_item_rel_ids.append(index)# 这里只考虑两种user-user 和user-item的关系情况，因为给我的样本就这两种，没考虑item-item
            index+=1
        #user_user_rel and user_item_rel
        if (left_id,right_id) not in visited_objs or (right_id,left_id) not in visited_objs:
            # if direction==1:
            #     tup=(left_id,right_id,uu_ui_rel[temp[3]])
            # elif direction==-1:
            #     tup=(right_id,left_id,)
            # elif direction==0:
            if left_id in user_ids and right_id in user_ids:
                uu_rel_and_ui_rel.append((user_trans_ids_dict[left_id],user_trans_ids_dict[right_id],uu_ui_rel[temp[3]]))
            else:
                uu_rel_and_ui_rel.append((user_trans_ids_dict[left_id],USER_COUNT+item_trans_ids_dict[right_id],uu_ui_rel[temp[3]]))
            visited_objs.append((left_id,right_id))

    RELATION_COUNT=len(uu_ui_rel.keys())

    # attribute.txt
    attr_entity = dict()
    attr_rel = dict()
    ##generate attr entities,从0开始
    index = 0
    for i in range(ATTR_COUNT):
        temp = attr_list[i].split(' ')
        # attr_rel and it's id
        attr_rel[temp[0]] = str(i + RELATION_COUNT)

        for j in range(1, len(temp)):
            # attr_entity and it's id
            attr_entity[temp[0] + ':' + temp[j]] = str(index)
            attr_ids.append(index)
            index += 1
    ATTR_ENTITY_COUNT=index
    ##(item,attr,rel)
    item_attr_rel = []
    for i in range(POOL_COUNT):
        temp = pool_list[i].split(' ')
        item_id = int(temp[0])
        keys = attr_entity.keys()
        for j in range(1, len(temp)):
            if temp[j] in keys:
                temp_temp = temp[j].split(":")
                attr_entity_id = int(attr_entity[temp[j]])
                item_attr_rel_id = int(attr_rel[temp_temp[0]])
                item_attr_rel_ids.append(item_attr_rel_id)
                item_attr_rel.append((item_trans_ids_dict[item_id]+USER_COUNT, attr_entity_id+USER_COUNT+ITEM_COUNT, item_attr_rel_id))  # tuple（item,attr,relation)
            else:
                print("The format of pool.txt and attribute.txt do not match.")
                return -1

    final_hrt=uu_rel_and_ui_rel+item_attr_rel
    print("Finish positive sampling, please wait for negative sampling...")
    #output
    save_pickle(final_hrt,"./h_pt_r_ant.pkl")

    all_rels=dict(uu_ui_rel,**attr_rel)
    # #save the relation id and name, and entity id and its name for validation

    output_attr_entity_trans_id=dict()
    for key in attr_entity.keys():
        output_attr_entity_trans_id[key]=int(attr_entity[key])+USER_COUNT+ITEM_COUNT
    save_file(output_attr_entity_trans_id,"./attr_entity_and_IDS_start_from "+str(USER_COUNT+ITEM_COUNT)+".txt")
    save_file(all_rels, "./all_relations_andIDs.txt")

    #negative sampling,final_hrt_list format: [(X,positive_Y,negative_Y,relation)]
    # final_hrt_list=negative_sampling(final_hrt,user_ids,item_ids,attr_ids,50)
    final_hrt_list=negative_sampling(final_hrt,USER_COUNT,ITEM_COUNT,ATTR_ENTITY_COUNT,50)
    print("Finish the negative sampling for pre-training...")

    generate_h_pt_nt1_nt2_for_rec(USER_COUNT,ITEM_COUNT,ATTR_ENTITY_COUNT)
    print("Finish the negative sampling for REC Modules...")

    return final_hrt_list

def negative_sampling(final_hrt,USER_COUNT,ITEM_COUNT,ATTR_ENTITY_COUNT,epoches):

    #positive ids
    positive_sample=final_hrt
    user_contain_user_ids=dict()
    user_contain_item_ids=dict()
    item_contain_attr_ids=dict()

    #outputfile,the true value == index+(USER_COUNT or USER_COUNT+ITEM_COUNT)
    user_contain_user_ids_output=dict()
    user_contain_item_ids_output=dict()
    item_contain_attr_ids_output=dict()


    #negative ids
    user_not_have_user_ids=dict()
    user_not_have_item_ids=dict()
    item_not_have_attr_ids=dict()

    #outputfile,the true value == index+(USER_COUNT or USER_COUNT+ITEM_COUNT)
    user_not_have_user_ids_output=dict()
    user_not_have_item_ids_output=dict()
    item_not_have_attr_ids_output=dict()


    for i in range(USER_COUNT):
        user_contain_item_ids[i]=[]
        user_contain_user_ids[i]=[]

    for i in range(ITEM_COUNT):
        item_contain_attr_ids[i]=[]

    #get the linked tails for each head
    for i in range(len(final_hrt)):
        if final_hrt[i][0]<USER_COUNT:
            if final_hrt[i][1]<USER_COUNT:
                #user-user
                user_contain_user_ids[final_hrt[i][0]].append(final_hrt[i][1])
            else:
                #user-item
                user_contain_item_ids[final_hrt[i][0]].append(final_hrt[i][1])
        else:
            #item-attr
            item_contain_attr_ids[final_hrt[i][0]-USER_COUNT].append(final_hrt[i][1])


    for key in user_contain_user_ids.keys():
        user_contain_user_ids_output[key]=user_contain_user_ids[key]

    for key in user_contain_item_ids.keys():
        user_contain_item_ids_output[key] = list(np.array(user_contain_item_ids[key])-USER_COUNT)

    for key in item_contain_attr_ids.keys():
        item_contain_attr_ids_output[key] = list(np.array(item_contain_attr_ids[key])-USER_COUNT-ITEM_COUNT)

        #字典key，value编号是其index+USER_COUNT or USER_COUNT+ITEM_COUNT
    save_pickle(user_contain_user_ids_output,'./pickle/user_user_dict.pkl')
    save_pickle(user_contain_item_ids_output, './pickle/user_item_dict.pkl')
    save_pickle(item_contain_attr_ids_output, './pickle/item_attr_dict.pkl')



    #save reverse graph
    reverse_user_contain_user_ids=dict()
    item_user_ids_dict=dict()
    attr_item_ids_dict=dict()
    for i in range(USER_COUNT):
        reverse_user_contain_user_ids[i]=[]
    for i in range(ITEM_COUNT):
        item_user_ids_dict[i]=[]
    for i in range(ATTR_ENTITY_COUNT):
        attr_item_ids_dict[i]=[]


    for key in user_contain_user_ids.keys():
        for value in user_contain_user_ids[i]:
            reverse_user_contain_user_ids[value].append(key)

    for key in user_contain_item_ids_output.keys():
        for value in user_contain_item_ids_output[key]:
            item_user_ids_dict[value].append(key)

    for key in item_contain_attr_ids_output.keys():
        for value in item_contain_attr_ids_output[key]:
            attr_item_ids_dict[value].append(key)
    #save reversed grapph
    save_pickle(reverse_user_contain_user_ids,'./pickle/reversed_user_user_dict.pkl')
    save_pickle(item_user_ids_dict, './pickle/item_user_dict.pkl')
    save_pickle(attr_item_ids_dict, './pickle/attr_item_dict.pkl')

    #get negative samples
        ##get negative user ids
    for key in user_contain_user_ids.keys():
        positive_list=user_contain_user_ids[key]
        negative_list=set(list(range(USER_COUNT)))-set(positive_list)-set([key])
        user_not_have_user_ids[key]=list(negative_list)


        ##get negative item ids
    for key in user_contain_item_ids.keys():
        positive_list=user_contain_item_ids[key]
        negative_list=set(list(range(USER_COUNT,USER_COUNT+ITEM_COUNT)))-set(positive_list)
        user_not_have_item_ids[key]=list(negative_list)

        ##get negative attr ids
    for key in item_contain_attr_ids.keys():
        positive_list=item_contain_attr_ids[key]
        negative_list=set(list(range(USER_COUNT+ITEM_COUNT,USER_COUNT+ITEM_COUNT+ATTR_ENTITY_COUNT)))-set(positive_list)
        item_not_have_attr_ids[key]=list(negative_list)

    #output negative dict
    for key in user_not_have_user_ids.keys():
        user_not_have_user_ids_output[key]=user_not_have_user_ids[key]

    for key in user_not_have_item_ids.keys():
        user_not_have_item_ids_output[key] = list(np.array(user_not_have_item_ids[key]) - USER_COUNT)

    for key in item_not_have_attr_ids.keys():
        item_not_have_attr_ids_output[key] = list(np.array(item_not_have_attr_ids[key]) - USER_COUNT - ITEM_COUNT)

    #字典key，value编号是其index+USER_COUNT
    save_pickle(user_not_have_user_ids_output,'./pickle/neg_user_user_dict.pkl')
    save_pickle(user_not_have_item_ids_output, './pickle/neg_user_item_dict.pkl')
    save_pickle(item_not_have_attr_ids_output, './pickle/neg_item_attr_dict.pkl')


    final_pos_neg_htr=[]
    for ii in range(epoches):
        pos_neg_htr=[]
        #build final (h,posi_t,neg_t,r) tuples
        for i in range(len(positive_sample)):
            if positive_sample[i][0] <USER_COUNT:
                if positive_sample[i][1] <USER_COUNT:
                    #user-user
                    if len(user_not_have_user_ids[positive_sample[i][0]]) > 0: #检查有没有negative samples
                        neg_sample = random.sample(user_not_have_user_ids[positive_sample[i][0]],1)
                        pos_neg_htr.append((positive_sample[i][0],positive_sample[i][1],neg_sample[0],positive_sample[i][2]))
                    else:
                        pos_neg_htr.append(
                            (positive_sample[i][0], positive_sample[i][1], "No_neg", positive_sample[i][2]))
                else:
                    #user-item
                    if len(user_not_have_item_ids[positive_sample[i][0]]) > 0:
                        neg_sample = random.sample(user_not_have_item_ids[positive_sample[i][0]], 1)
                        pos_neg_htr.append((positive_sample[i][0], positive_sample[i][1], neg_sample[0], positive_sample[i][2]))
                    else:
                        pos_neg_htr.append(
                            (positive_sample[i][0], positive_sample[i][1], "No_neg", positive_sample[i][2]))
            else:
                #item-attr
                if len(item_not_have_attr_ids[positive_sample[i][0]-USER_COUNT]) > 0:
                    neg_sample = random.sample(item_not_have_attr_ids[positive_sample[i][0]-USER_COUNT], 1)
                    pos_neg_htr.append((positive_sample[i][0], positive_sample[i][1], neg_sample[0], positive_sample[i][2]))
                # else:
                #     pos_neg_htr.append(
                #         (positive_sample[i][0], positive_sample[i][1], "No_neg", positive_sample[i][2]))
        save_pickle(pos_neg_htr,"./train_set/h_pt_nt_r_ant_train_dataset_{}.pkl".format(str(ii)))
        final_pos_neg_htr.append(pos_neg_htr)

    save_pickle(final_pos_neg_htr,"./train_set/h_pt_nt_r_ant_train_dataset_list.pkl")
    return final_pos_neg_htr

def generate_h_pt_nt1_nt2_for_rec(USER_COUNT,ITEM_COUNT,ATTR_ENTITY_COUNT):
    final_hrt=load_pickle("./h_pt_r_ant.pkl")
    user_contain_user_ids=load_pickle('./pickle/user_user_dict.pkl')
    user_contain_item_ids=load_pickle('./pickle/user_item_dict.pkl')
    item_contain_attr_ids=load_pickle('./pickle/item_attr_dict.pkl')

    #import reversed graph
    reverse_user_contain_user_ids=load_pickle('./pickle/reversed_user_user_dict.pkl')
    item_user_ids_dict=load_pickle('./pickle/item_user_dict.pkl')
    attr_item_ids_dict=load_pickle('./pickle/attr_item_dict.pkl')


    #import negative dict

    neg_user_user_ids=load_pickle('./pickle/neg_user_user_dict.pkl')
    neg_user_item_ids=load_pickle('./pickle/neg_user_item_dict.pkl')
    neg_item_attr_ids=load_pickle('./pickle/neg_item_attr_dict.pkl')


    for iii in range(50):
        user, p_item, n_item, n_item2, attr = [], [], [], [], []
        random.shuffle(final_hrt)
        for one_htr in final_hrt:
            if one_htr[0]<USER_COUNT:
                if one_htr[1]>=USER_COUNT:
                    #user-item
                    user.append(one_htr[0])
                    p_item.append(one_htr[1])
                    #random_choose_negative_items
                    n_item.append(random.sample(neg_user_item_ids[one_htr[0]],1))

                    #get the attribute in this item
                    attr_ids=item_contain_attr_ids[one_htr[1]-USER_COUNT]
                    temp_attr = attr_ids
                    random.shuffle(temp_attr)
                    max_attr = min(len(attr_ids), 15)
                    i=0
                    temp=neg_user_item_ids[one_htr[0]]
                    for i in range(max_attr):
                        #aggregate the other items that has same attrs
                        temp_1 = list(set(temp) & set(attr_item_ids_dict[temp_attr[i]]))
                        if len(temp_1) < 10 and i > 0:
                            i = i - 1
                            break
                        temp = temp_1
                    attr.append(temp_attr[:i + 1])
                    if len(temp) == 0:
                        n_item2.append(None)
                    else:
                        n_item2.append(np.random.choice(temp))
        train_for_rec_item = [user, p_item, n_item, n_item2, attr]
        save_pickle(train_for_rec_item, './train_set/v1-speed-train-{}.pickle'.format(iii))
        print('data', iii, 'end ... ', 'Length:', len(train_for_rec_item[0]), len(train_for_rec_item[1])
              , len(train_for_rec_item[2]), len(train_for_rec_item[3]),len(train_for_rec_item[4]))
                # else:
                #     #user-user

            # else:
            #     #item-attr
