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
from models_amazon import Generator
amazon = np.load("/root/data/xiaoyang/PyTorch-CycleGAN-master/datasets/amazon.npz")
amazon_xx = coo_matrix((amazon['xx_data'], (amazon['xx_col'], amazon['xx_row'])),
                       shape=amazon['xx_shape'][::-1]).tocsc()
amazon_xx = amazon_xx[:, :5000]
amazon_yy = amazon['yy']
amazon_yy = (amazon_yy + 1) / 2
amazon_offset = amazon['offset'].flatten()
data_name = ["books", "dvd", "electronics", "kitchen"]
num_data_sets = 4
data_insts, data_labels, num_insts = [], [], []
for i in range(num_data_sets):
    data_insts.append(amazon_xx[amazon_offset[i]: amazon_offset[i+1], :])
    data_labels.append(amazon_yy[amazon_offset[i]: amazon_offset[i+1], :])
    num_insts.append(amazon_offset[i+1] - amazon_offset[i])
    # Randomly shuffle.
    r_order = np.arange(num_insts[i])
    np.random.shuffle(r_order)
    data_insts[i] = data_insts[i][r_order, :]
    data_labels[i] = data_labels[i][r_order, :]
matrix = [i.todense() for i in data_insts]
data_labels = [i.tolist() for i in data_labels]
matrix = [i.tolist() for i in matrix]
data_labels = [[int(j[0]) for j in i] for i in data_labels]
print("import finished")

class SoftMax(nn.Module):
    def __init__(self, n_feature=5000,n_out=2):
        super(SoftMax, self).__init__()
        self.hidden1 = nn.Linear(n_feature, 1000)
        self.hidden2=nn.Linear(1000,500)
        self.hidden3=nn.Linear(500,100)
        self.out = nn.Linear(100, n_out)
        self.model=nn.Sequential(self.hidden1,nn.Dropout(0.5),nn.LeakyReLU(),
                                  self.hidden2,nn.Dropout(0.3),nn.LeakyReLU(),
                                 self.hidden3,nn.Dropout(0.2),nn.LeakyReLU(),
                                  self.out)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)  # 返回的是每个类的概率

print("net topology designed")

epoches = 200
batch_size=16
root="/root/data/xiaoyang/five-domains"
data_name=["custrev","laptop","restaurant","rt-polarity","stsa_binary"]
res = [0]*5
print("training...")
for target in range(1,3):
    net = SoftMax(n_feature=5000, n_out=2).to("cuda")
    optimizer = torch.optim.Adadelta(net.parameters(), lr=1.0)
    lr_schedular = torch.optim.lr_scheduler.StepLR(optimizer,
                                                           step_size=20, gamma=0.8)
    source_reviews = [matrix[idx] for idx in range(4) if not idx == target]
    target_reviews = matrix[target]
    source_sentiments = [data_labels[idx] for idx in range(4) if not idx == target]
    target_sentiments = data_labels[target]
    net_G = Generator().to("cuda")
    net_G.load_state_dict(
        torch.load('/root/data/xiaoyang/PyTorch-CycleGAN-master/model/GAN2/' + str(target) + '/netG_A2B.pth'))
    for epoch in range(epoches):
        train_loss=0
        for source in range(3):
            source_loss=0
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
                source_loss+=loss
                # print("target: ", target, "epoch: ", epoch, "source: ", source, "batch: ", batch, "loss: ", loss,
                #       "res: ", res)
                loss.backward()
                # nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
                optimizer.step()
            train_loss+=source_loss
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
        print("target: ", target, "epoch: ", epoch, "acc: ", acc/max_len,"res: ",res,"train_loss: ",float(train_loss), "dev_loss: ",float(dev_avg_loss))