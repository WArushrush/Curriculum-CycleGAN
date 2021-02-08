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
from models import Generator
print("import finished")

class SoftMax(nn.Module):
    def __init__(self, n_feature=1024,n_out=2):
        super(SoftMax, self).__init__()
        self.hidden1 = nn.Linear(n_feature, 512)
        self.hidden2=nn.Linear(512,256)
        self.hidden3=nn.Linear(256,128)
        self.out = nn.Linear(128, n_out)
        self.model=nn.Sequential(self.hidden1,nn.ReLU(),nn.Dropout(0.5),
                                  self.hidden2,nn.ReLU(),nn.Dropout(0.3),
                                 self.hidden3,nn.ReLU(),nn.Dropout(0.2),
                                  self.out)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)  # 返回的是每个类的概率

print("net topology designed")

epoches = 20
batch_size=64
root="/root/data/xiaoyang/five-domains"
data_name=["custrev","laptop","restaurant","rt-polarity","stsa_binary"]
res = [0]*5
print("training...")
for target in range(5):
    srcs = [open(root + "/" + data_name[target] + "_train_embedding_3.pkl", 'rb')]
    target_domain = open(root + "/" + data_name[target] + "_test_embedding_3.pkl", 'rb')
    source_reviews = []
    source_sentiments = []
    for src in srcs:
        temp = pickle.load(src)
        source_reviews.append([review[0] for review in temp[0]])
        source_sentiments.append(temp[1])
    temp = pickle.load(target_domain)
    target_reviews = [review[0] for review in temp[0]]
    target_sentiments = temp[1]
    net_G = Generator().to("cuda")
    net_G.load_state_dict(
        torch.load('/root/data/xiaoyang/PyTorch-CycleGAN-master/model/GAN/' + str(target) + '/netG_A2B.pth'))
    net = SoftMax(n_feature=1024, n_out=2).to("cuda")
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=0.001)
    lr_schedular = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=20, gamma=0.5)
    for source in range(1):
        for epoch in range(epoches):
            max_len = len(source_sentiments[source])
            for batch in range(max_len // batch_size):
                optimizer.zero_grad()
                # Set model input
                real_A = []
                A_sentiments = []
                for idx in range(batch * batch_size, (batch + 1) * batch_size):
                    temp = random.randint(0,max_len-1)
                    cur = source_reviews[source][idx % max_len]
                    cur = torch.tensor(cur, requires_grad=False).view(1, -1).to("cuda")
                    real_A.append(cur)
                    A_sentiments.append(source_sentiments[source][idx % max_len])
                real_A = torch.cat(real_A, 0)
                # real_A = net_G(real_A)
                A_sentiments = torch.tensor(A_sentiments, requires_grad=False).to("cuda")
                pred = net(real_A)
                loss = F.nll_loss(pred, A_sentiments)
                # print("target: ", target, "epoch: ", epoch, "source: ", source, "batch: ", batch, "loss: ", loss,
                #       "res: ", res)
                loss.backward()
                # nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
                optimizer.step()
            lr_schedular.step()
            max_len = len(target_sentiments)
            total_loss=0
            acc = 0
            for batch in range(max_len):
                cur = target_reviews[batch]
                cur = torch.tensor(cur, requires_grad=False).view(1, -1).to("cuda")
                pred = net(cur)
                total_loss+=F.nll_loss(pred,torch.tensor([target_sentiments[batch]], requires_grad=False).to("cuda"))
                pred = int(pred[0][1]>pred[0][0])
                if pred == int(target_sentiments[batch]):
                    acc+=1
            dev_avg_loss = total_loss
            res[target] = max(res[target], acc/max_len)
            print("target: ", target, "epoch: ", epoch, "acc: ", acc/max_len,"res: ",res, "dev_loss: ",float(dev_avg_loss))