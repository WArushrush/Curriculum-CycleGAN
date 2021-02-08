import numpy as np
import sklearn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.optim
import time
import argparse
import pickle
import random
import numpy as np
import time
import argparse
from scipy.sparse import coo_matrix
from models import Generator
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
matrix = []
data_labels = []
languages = ['de', 'en','fr','jp']
domains = ['books','dvd','music']
root = "/root/data/xiaoyang/PyTorch-CycleGAN-master/cls-acl10-unprocessed"
for lan in languages:
    for domain in domains:
        f = open(root+"/"+lan+"/"+domain+"_train_embedding_4.pkl",'rb')
        temp = pickle.load(f)
        matrix.append(temp[0])
        data_labels.append(temp[1])
test_matrix = []
test_data_labels = []
for lan in languages:
    for domain in domains:
        f = open(root+"/"+lan+"/"+domain+"_test_embedding_4.pkl",'rb')
        temp = pickle.load(f)
        test_matrix.append(temp[0])
        test_data_labels.append(temp[1])
print("import finished")

class SoftMax(nn.Module):
    def __init__(self, n_feature=1024,n_out=2):
        super(SoftMax, self).__init__()
        self.hidden1 = nn.Linear(n_feature, 512)
        # self.hidden2=nn.Linear(80,60)
        # self.hidden3=nn.Linear(60,40)
        # self.hidden4=nn.Linear(40,20)
        self.hidden2 = nn.Linear(512,256)
        self.hidden3 = nn.Linear(256,128)
        self.out = nn.Linear(128, n_out)
        self.model=nn.Sequential(self.hidden1, nn.ReLU(),
                                 self.hidden2, nn.ReLU(),
                                 self.hidden3, nn.ReLU(),
                                 # self.hidden3, nn.ReLU(),
                                 # self.hidden4, nn.ReLU(),
                                  self.out)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)  # 返回的是每个类的概率

print("net topology designed")

epoches = 20
batch_size=64
print("training...")
res = [0,0,0]*4
for target in range(4):
    for domain in range(3):
        net_A = SoftMax(n_feature=1024, n_out=2).to("cuda")
        optimizer_A = torch.optim.Adam(net_A.parameters(), lr=0.0001, betas=(0.5, 0.999))
        lr_schedular_A = torch.optim.lr_scheduler.StepLR(optimizer_A,
                                                               step_size=200, gamma=0.5)
        source_reviews = [matrix[idx*3+domain] for idx in range(4) if not idx == target]
        target_reviews = test_matrix[target*3+domain]
        source_sentiments = [data_labels[idx*3+domain] for idx in range(4) if not idx == target]
        target_sentiments = test_data_labels[target*3+domain]
        net_G = Generator().to("cuda")
        net_G.load_state_dict(
            torch.load('/root/data/xiaoyang/PyTorch-CycleGAN-master/model/GAN4/' + str(target) + '/netG_A2B.pth'))
        for epoch in range(epoches):
            train_loss=0
            for source in range(3):
                source_loss=0
                max_len = len(source_sentiments[source])
                print("maxlen: ", max_len)
                for batch in range(max_len//batch_size):
                    optimizer_A.zero_grad()
                    # Set model input
                    real_A = []
                    A_sentiments = []
                    for idx in range(batch * batch_size, (batch+1) * batch_size):
                        # temp = random.randint(0,max_len-1)
                        cur = source_reviews[source][idx % max_len]
                        cur = torch.tensor(cur, requires_grad=False).view(1, -1).to("cuda")
                        real_A.append(cur)
                        A_sentiments.append(source_sentiments[source][idx % max_len])
                    real_A = torch.cat(real_A, 0)
                    real_A = net_G(real_A)
                    A_sentiments = torch.tensor(A_sentiments, requires_grad=False).to("cuda")
                    dirty = False
                    temp = torch.isnan(real_A)
                    for i in range(batch_size):
                        for j in range(100):
                            if temp[i][j] == torch.tensor(True):
                                dirty = True
                    if dirty:
                        continue
                    pred = net_A(real_A)
                    loss = F.nll_loss(pred, A_sentiments)
                    source_loss+=loss
                    print("target: ", target, "domain: ", domain, "epoch: ", epoch, "source ", source, " batch: ", batch, " loss: ", loss)
                    optimizer_A.zero_grad()
                    loss.backward()
                    # nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
                    optimizer_A.step()
                train_loss+=source_loss
            max_len = len(target_sentiments)
            total_loss=0
            acc = 0
            for batch in range(max_len):
                cur = target_reviews[batch]
                cur = torch.tensor(cur, requires_grad=False).view(1, -1).to("cuda")
                pred = net_A(cur)
                total_loss+=F.nll_loss(pred,torch.tensor([target_sentiments[batch]], requires_grad=False).to("cuda"))
                pred = int(pred[0][1]>pred[0][0])
                if pred == int(target_sentiments[batch]):
                    acc+=1
            dev_avg_loss = total_loss
            res[target*3+domain] = max(res[target*3+domain], acc/max_len)
            print("target: ", target, "domain: ", domain, "epoch: ", epoch, "acc: ", acc/max_len, "res: ", res,"train_loss: ",float(train_loss), "dev_loss: ",float(dev_avg_loss))
            lr_schedular_A.step()
