# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np 
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import torch.nn.functional as F

import time

from configparser import ConfigParser
config = ConfigParser()
config.read('conf', encoding='UTF-8')

from RedaData import RedaData
from RedaUtils import test_util

FType = torch.FloatTensor
LType = torch.LongTensor


class Reda(nn.Module):

    def __init__(self): 
        super(Reda,self).__init__()

        self.learning_rate = float(config['DEFAULT'].get("learning_rate"))
        self.epochs = int(config['DEFAULT'].get("epochs"))
        
        self.vec_size = int(config['DEFAULT'].get("vec_size"))
        self.memory_slices = int(config['DEFAULT'].get("memory_size"))
        self.K = int(config['DEFAULT'].get("K"))
        self.weight_size = int(config['DEFAULT'].get("weight_size"))
        
        self.start_epoch = int(config['DEFAULT'].get("start_epoch"))
        self.inter_epoch = int(config['DEFAULT'].get("inter_epoch"))

        self.dirname = config['DEFAULT'].get("dirname")
        self.batch_size = int(config['DEFAULT'].get("batch_size"))
        self.data = RedaData(self.dirname)
        self.num_users, self.num_items = self.data.get_user_item_dim()
        
        self.user_matrix = torch.tensor(torch.from_numpy(np.random.uniform(-0.1,0.1,(self.num_users, self.vec_size))).type(FType),requires_grad=False)
        
        self.item_matrix = torch.tensor(torch.from_numpy(np.random.uniform(-0.1,0.1,(self.num_items, self.K, self.vec_size))).type(FType),requires_grad=True)
        
        self.memory_keys = torch.tensor(torch.from_numpy(np.random.uniform(-0.1,0.1,(self.vec_size, self.memory_slices))).type(FType),requires_grad=True)

        self.memory_matrix = torch.tensor(torch.from_numpy(np.random.uniform(-0.1,0.1,(self.memory_slices, self.vec_size))).type(FType),requires_grad=True)

        self.weight_net = torch.nn.Sequential(
                    nn.Linear(self.vec_size,self.weight_size),
                    nn.ReLU())

        self.weight_factor = torch.tensor(torch.from_numpy(np.random.uniform(-0.1,0.1,(self.weight_size, 1))).type(FType),requires_grad=True)

        self.opt = Adam(lr=self.learning_rate, params=[{"params":self.item_matrix},{"params":self.memory_matrix},\
            {"params":self.memory_keys},{"params":self.weight_net.parameters()},{"params":self.weight_factor}], weight_decay=1e-5)
        
        self.loss = torch.FloatTensor()
        self.metrics = [0.0,0.0,0.0,0.0,0.0,0.0]


    def pairwise_interaction(self,item_matrix,thresold_matrix):
        batch = item_matrix.size()[0]
        vec_size = self.vec_size
        K = self.K
        interaction_matrix = torch.zeros((batch,K**2,vec_size), dtype=torch.float)
        for i in range(K):
            partial_item_mat = item_matrix[:,i,:]
            partial_item_mat = partial_item_mat.view(batch,1,vec_size)
            partial_item_mat = partial_item_mat.expand(batch,K,vec_size)
            sub_interaction_matrix = partial_item_mat.mul(thresold_matrix)
            interaction_matrix[:,(i*K):(i+1)*K,:] = sub_interaction_matrix
        return interaction_matrix

    def pooling_operation(self,memory_attention_matrix,weight_attention_matrix):
        batch, attr_size, _ = weight_attention_matrix.size()     #1,9,
        weight_attention_matrix = weight_attention_matrix.expand(batch,attr_size,self.vec_size)
        weighted_relation_emb = weight_attention_matrix.mul(memory_attention_matrix)          #1,9,64
        relation_emb = weighted_relation_emb.sum(dim=1)                                 #1,64
        return relation_emb

    def weight_attention(self,interaction_matrix): 
        hidden = self.weight_net(interaction_matrix)          #1*9*10
        #print(hidden.size())
        attention = hidden.matmul(self.weight_factor)   #1*9*1
        soft_attention = F.softmax(attention,dim=1)
        return soft_attention

    def memory_attention(self,interaction_matrix):
        #print(self.memory_keys.size())
        #print(interaction_matrix.size())
        attention = interaction_matrix.matmul(self.memory_keys)
        #print(attention.size())
        soft_attention = F.softmax(attention,dim=2)         #taking softmax along the third dimension i.e., 't'/'m' acc to the paper 
        memory_attention_matrix = soft_attention.matmul(self.memory_matrix)     #1*9*64
        return memory_attention_matrix

    def primitive_layer(self,item_matrix,thresold_matrix):
        interaction_matrix = self.pairwise_interaction(item_matrix,thresold_matrix)
        memory_attention_matrix = self.memory_attention(interaction_matrix) 
        weight_attention_matrix = self.weight_attention(interaction_matrix) 
        relation_emb = self.pooling_operation(memory_attention_matrix, weight_attention_matrix)
        return relation_emb

    def loss_function(self,pos_sum_relation_emb,neg_sum_relation_emb,base_sum_relation_emb):
        pos_diff = torch.sum(pos_sum_relation_emb.mul(base_sum_relation_emb),dim=1)
        neg_diff = torch.sum(neg_sum_relation_emb.mul(base_sum_relation_emb),dim=1)

        rank_loss = (neg_diff - pos_diff)
        rank_loss = rank_loss.sigmoid()
        rank_loss = torch.sum(rank_loss)
        return rank_loss

    def forward_pass(self, common_his_item, pos_item, neg_item, base_item):
        batch = pos_item.size()[0]

        pos_item_matrix = self.item_matrix.index_select(0, torch.tensor(pos_item.view(-1)))
        pos_item_matrix = pos_item_matrix.view(batch,self.K,self.vec_size)

        neg_item_matrix = self.item_matrix.index_select(0, torch.tensor(neg_item.view(-1)))
        neg_item_matrix = neg_item_matrix.view(batch,self.K,self.vec_size)

        thresold_matrix = self.item_matrix.index_select(0, torch.tensor(common_his_item.view(-1)))
        thresold_matrix = thresold_matrix.view(batch,self.K,self.vec_size)

        base_item_matrix = self.item_matrix.index_select(0, torch.tensor(base_item.view(-1)))
        base_item_matrix = base_item_matrix.view(batch,self.K,self.vec_size)

        pos_relation_emb = self.primitive_layer(pos_item_matrix,thresold_matrix)
        neg_relation_emb = self.primitive_layer(neg_item_matrix,thresold_matrix)
        base_relation_emb = self.primitive_layer(base_item_matrix,thresold_matrix)

        rank_loss = self.loss_function(pos_relation_emb,neg_relation_emb,base_relation_emb)
        return rank_loss

    def optimize_parameters(self, common_his_item, pos_item, neg_item, base_item):
        self.opt.zero_grad()
        #print(common_his_item)
        #print(pos_item.size())
        loss = self.forward_pass(common_his_item, pos_item, neg_item, base_item)

        self.loss += loss.data
        loss.backward()
        self.opt.step()

    def model_train(self):
        for epoch in range(self.epochs):
            self.loss =  0.0
            loader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True)

            start = time.time()

            for i_batch, sample_batched in enumerate(loader):
                end = time.time()
                print("=======================> sample_data", i_batch, end-start)

                self.optimize_parameters(sample_batched['common_his_item'],sample_batched['pos_item'],
                    sample_batched['neg_item'],sample_batched["base_item"])

                end = time.time()
                print("=======================> training", i_batch, end-start)

            print("\repoch "+ str(epoch) +" : avg loss = " + str(self.loss/len(self.data)))
            print("=============================> cosing time ", time.time()-start)
            
            if epoch >= self.start_epoch and (epoch+1) % self.inter_epoch == 0:
                test_start = time.time()
                self.model_test()
                print("One Test Has Finished", time.time()-test_start)

    def user_matrix_build(self,user,pair_one,pair_two):
        batch = pair_one.size()[0]
        pair_one = self.item_matrix.index_select(0, torch.tensor(pair_one.view(-1)))
        pair_one = pair_one.view(batch,self.K,self.vec_size)
        pair_two = self.item_matrix.index_select(0, torch.tensor(pair_two.view(-1)))
        pair_two = pair_two.view(batch,self.K,self.vec_size)
        relation_emb = self.primitive_layer(pair_one,pair_two)
        relation_emb = relation_emb.sum(dim=0)

        self.user_matrix[user] = relation_emb

    def forward_test(self, user, target_item_pair_one, target_item_pair_two):
        user = torch.LongTensor(user)
        target_item_pair_one = torch.LongTensor(target_item_pair_one)
        target_item_pair_two = torch.LongTensor(target_item_pair_two)

        batch = target_item_pair_one.size()[0]

        user_matrix = self.user_matrix.index_select(0, torch.tensor(user.view(-1)))
        user_matrix = user_matrix.view(1, self.vec_size) # 1*emb

        target_item_pair_one = self.item_matrix.index_select(0, torch.tensor(target_item_pair_one.view(-1)))
        target_item_pair_one = target_item_pair_one.view(batch,self.K,self.vec_size)

        target_item_pair_two = self.item_matrix.index_select(0, torch.tensor(target_item_pair_two.view(-1)))
        target_item_pair_two = target_item_pair_two.view(batch,self.K,self.vec_size) 

        target_relation_emb = self.primitive_layer(target_item_pair_one,target_item_pair_two)
        target_relation_emb = target_relation_emb.sum(dim=0)

        prefer_score = torch.sum(user_matrix.mul(target_relation_emb))

        return prefer_score.cpu().data.numpy()

    def model_test(self):
        with torch.no_grad():
            for user in self.data.user2item:
                his_items = list(self.data.user2item[user])
                pair_one, pair_two = [], []

                max_len = len(his_items)
                for i in range(max_len):
                    for j in range(i+1,max_len):
                        pair_one.append(his_items[i])
                        pair_two.append(his_items[j])

                pair_one = torch.LongTensor(pair_one)
                pair_two = torch.LongTensor(pair_two)
                user = torch.LongTensor([user])
                self.user_matrix_build(user,pair_one,pair_two)

            hr_5,ndcg_5,hr_10,ndcg_10,hr_15,ndcg_15 = test_util(self,self.dirname)
            max_hr_5 = max(self.metrics[0],hr_5)
            max_ndcg_5 = max(self.metrics[1],ndcg_5)
            max_hr_10 = max(self.metrics[2],hr_10)
            max_ndcg_10 = max(self.metrics[3],ndcg_10)
            max_hr_15 = max(self.metrics[4],hr_15)
            max_ndcg_15 = max(self.metrics[5],ndcg_15)

            print("<=================================================>")
            print("Max_HR@5: ", max_hr_5)
            print("Max_ndcg@5: ", max_ndcg_5)
            print("Max_HR@10: ", max_hr_10)
            print("Max_ndcg@10: ", max_ndcg_10)
            print("Max_HR@15: ", max_hr_15)
            print("Max_ndcg@15: ", max_ndcg_15)
            print("<=================================================>")
            self.metrics = [max_hr_5, max_ndcg_5, max_hr_10, max_ndcg_10, max_hr_15, max_ndcg_15]

    def pred(self, user, pair_one,pair_two):
        #user = torch.LongTensor(user)
        target_item_pair_one = torch.LongTensor(pair_one)
        target_item_pair_two = torch.LongTensor(pair_two)

        batch = target_item_pair_one.size()[0]

        #user_matrix = self.user_matrix.index_select(0, torch.tensor(user.view(-1)))
        #user_matrix = user_matrix.view(1, self.vec_size) # 1*emb

        target_item_pair_one = self.item_matrix.index_select(0, torch.tensor(target_item_pair_one.view(-1)))
        target_item_pair_one = target_item_pair_one.view(batch,self.K,self.vec_size)

        target_item_pair_two = self.item_matrix.index_select(0, torch.tensor(target_item_pair_two.view(-1)))
        target_item_pair_two = target_item_pair_two.view(batch,self.K,self.vec_size) 

        target_relation_emb = self.primitive_layer(target_item_pair_one,target_item_pair_two)
        target_relation_emb = target_relation_emb.sum(dim=0)

        prefer_score = torch.sum(user.mul(target_relation_emb))

        return prefer_score.cpu().data.numpy()
    
    def prediction(self, list_user, target):
        result = {}
        pair_one_ll=[]
        pair_two_ll=[]
        for i in range(len(list_user)):
            for j in range(i+1,len(list_user)):
                    pair_one_ll.append(list_user[i])
                    pair_two_ll.append(list_user[j])
        batch = len(list_user)
        pair_one_l = torch.LongTensor(pair_one_ll)
        pair_two_l = torch.LongTensor(pair_two_ll)
        pair_one = self.item_matrix.index_select(0, torch.tensor(pair_one_l.view(-1)))
        pair_one = pair_one.view(batch,self.K,self.vec_size)
        pair_two = self.item_matrix.index_select(0, torch.tensor(pair_two_l.view(-1)))
        pair_two = pair_two.view(batch,self.K,self.vec_size)
        relation_emb = self.primitive_layer(pair_one,pair_two)
        relation_emb = relation_emb.sum(dim=0)

        for t in target:
            pair_two = list_user
            pair_one = []
            for i in range(len(pair_two)):
                pair_one.append(t)
            score = self.pred(relation_emb, pair_one , pair_two)
            result[t] = score
        
        result = sorted(result.items(), key=lambda item:item[1], reverse=True)

        for i in range(10):
            print(result[i][0])


if __name__ == '__main__':
    mr_model = Reda()
    mr_model.model_train()
    mr_model.model_test()

    