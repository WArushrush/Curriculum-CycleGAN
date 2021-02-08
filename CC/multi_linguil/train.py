#!/usr/bin/python3

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from models import Generator
from models import Discriminator
from models import CMSS
from utils import ReplayBuffer
from utils import LambdaLR
import os
import random
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
root = "/root/data/xiaoyang/PyTorch-CycleGAN-master/cls-acl10-unprocessed"
languages = ['de','en','fr','jp']
data_name = ['books','dvd','music']
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=64, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=71, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


###### Definition of variables ######
# Networks
# Inputs & targets memory allocation
# Loss plot
###################################

def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl)

res = [0,0,0]*4

###### Training ######
for lan in range(4):
    for target in range(3):
        netG_A2B = Generator()
        netG_B2A = Generator()
        netD_A = Discriminator()
        netD_B = Discriminator()
        # netS_A = Discriminator()
        netS_B = Discriminator()
        # netC_A = CMSS()
        # netC_B = CMSS()
        if opt.cuda:
            netG_A2B.cuda()
            netG_B2A.cuda()
            netD_A.cuda()
            netD_B.cuda()
            # netS_A.cuda()
            netS_B.cuda()
            # netC_A.cuda()
            # netC_B.cuda()
        netG_A2B = nn.DataParallel(netG_A2B, device_ids=[0, 1])
        netG_B2A = nn.DataParallel(netG_B2A, device_ids=[0, 1])
        netD_A = nn.DataParallel(netD_A, device_ids=[0, 1])
        netD_B = nn.DataParallel(netD_B, device_ids=[0, 1])
        # netS_A = nn.DataParallel(netS_A, device_ids=[0, 1])
        netS_B = nn.DataParallel(netS_B, device_ids=[0, 1])
        Tensor = torch.tensor
        target_real = torch.ones(opt.batchSize, requires_grad=False).long().to("cuda")
        target_fake = torch.zeros(opt.batchSize, requires_grad=False).long().to("cuda")
        target_distribution = torch.tensor([1.0/opt.batchSize for _ in range(opt.batchSize)], requires_grad=False).view(1,-1).to("cuda")

        fake_A_buffer = ReplayBuffer()
        fake_B_buffer = ReplayBuffer()

        # Dataset loader
        # Losses
        criterion_GAN = F.nll_loss
        criterion_cycle = torch.nn.L1Loss()
        criterion_identity = torch.nn.L1Loss()
        criterion_kl = nn.KLDivLoss()

        # Optimizers & LR schedulers
        optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                       lr=opt.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        # optimizer_S_A = torch.optim.Adam(netS_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        optimizer_S_B = torch.optim.Adam(netS_B.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        # optimizer_C_A = torch.optim.Adam(netC_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        # optimizer_C_B = torch.optim.Adam(netC_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        lr_scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G,
                                                         step_size=10, gamma=0.5)
        lr_scheduler_D_A = torch.optim.lr_scheduler.StepLR(optimizer_D_A,
                                                           step_size=10, gamma=0.5)
        lr_scheduler_D_B = torch.optim.lr_scheduler.StepLR(optimizer_D_B,
                                                           step_size=10, gamma=0.5)
        # lr_scheduler_S_A = torch.optim.lr_scheduler.StepLR(optimizer_S_A,
        #                                                    step_size=2, gamma=0.5)
        lr_scheduler_S_B = torch.optim.lr_scheduler.StepLR(optimizer_S_B,
                                                           step_size=10, gamma=0.5)
        # lr_scheduler_C_A = torch.optim.lr_scheduler.StepLR(optimizer_C_A,
        #                                                    step_size=40, gamma=0.5)
        # lr_scheduler_C_B = torch.optim.lr_scheduler.StepLR(optimizer_C_B,
        #                                                    step_size=40, gamma=0.5)
        for epoch in range(opt.epoch, opt.n_epochs):
            srcs = [open(root + "/" + languages[idx] + "/" + data_name[target] + "_train_embedding_5.pkl", 'rb') for idx in range(4) if
                    idx != lan]
            target_domain = open(root + "/" + languages[lan] + "/" + data_name[target] + "_train_embedding_5.pkl", 'rb')
            test_domain = open(root + "/" + languages[lan] + "/" + data_name[target] + "_test_embedding_5.pkl", 'rb')
            source_reviews = []
            source_sentiments = []
            target_reviews = []
            target_sentiments = []
            test_reviews = []
            test_sentiments = []
            for src in srcs:
                temp = pickle.load(src)
                source_reviews.append([review[0] for review in temp[0]])
                source_sentiments.append(temp[1])
            temp = pickle.load(target_domain)
            target_reviews = [review[0] for review in temp[0]]
            target_sentiments = temp[1]
            temp = pickle.load(test_domain)
            test_reviews = [review[0] for review in temp[0]]
            test_sentiments = temp[1]
            temp_reviews = []
            for source_review in source_reviews:
                temp_reviews += source_review
            temp_sentiments = []
            for source_sentiment in source_sentiments:
                temp_sentiments += source_sentiment
            randnum = random.randint(0, 1000)
            random.seed(randnum)
            random.shuffle(temp_reviews)
            random.seed(randnum)
            random.shuffle(temp_sentiments)
            randnum = random.randint(0, 1000)
            random.seed(randnum)
            random.shuffle(target_reviews)
            random.seed(randnum)
            random.shuffle(target_sentiments)
            source_reviews = [temp_reviews]
            source_sentiments = [temp_sentiments]
            for source in range(1):
                max_len = len(source_sentiments[source])
                for batch in range(max_len // opt.batchSize):
                    # Set model input
                    real_A = []
                    A_sentiments = []
                    for idx in range(batch * opt.batchSize, (batch + 1) * opt.batchSize):
                        cur = source_reviews[source][idx%max_len]
                        cur = torch.tensor(cur, requires_grad=False).view(1, -1).to("cuda")
                        real_A.append(cur)
                        A_sentiments.append(source_sentiments[source][idx%max_len])
                    real_A = torch.cat(real_A, 0)
                    A_sentiments = torch.tensor(A_sentiments, requires_grad=False).to("cuda")
                    real_B = []
                    B_sentiments = []
                    len_target = len(target_sentiments)
                    for idx in range(batch * opt.batchSize, (batch + 1) * opt.batchSize):
                        cur = target_reviews[idx%len_target]
                        cur = torch.tensor(cur, requires_grad=False).view(1, -1).to("cuda")
                        real_B.append(cur)
                        B_sentiments.append(target_sentiments[idx%len_target])
                    real_B = torch.cat(real_B, 0)
                    B_sentiments = torch.tensor(B_sentiments, requires_grad=False).to("cuda")
                    ###### Generators A2B and B2A ######

                    # Identity loss
                    # G_A2B(B) should equal B if real B is fed
                    same_B = netG_A2B(real_B)
                    loss_identity_B = criterion_identity(same_B, real_B) * 5.0
                    # G_B2A(A) should equal A if real A is fed
                    same_A = netG_B2A(real_A)
                    loss_identity_A = criterion_identity(same_A, real_A) * 5.0

                    # GAN loss
                    fake_B = netG_A2B(real_A)
                    pred_fake = netD_B(fake_B)
                    # weight_A = F.softmax(netC_A(fake_B),dim=0)
                    # weight_real_B = F.log_softmax(netC_A(real_B), dim=0).view(1,-1)
                    # loss_kl_A = criterion_kl(weight_real_B, target_distribution)*500
                    init_loss = torch.tensor([criterion_GAN(pred_fake[_].view(1,-1), target_real[_].view(1)) for _ in range(opt.batchSize)]).to("cuda")
                    weight_A = F.softmax(-init_loss*10)
                    # print("weight_real_B: ", weight_real_B, " log_softmax: ", weight_real_B, " kl loss: ", loss_kl_A)
                    loss_GAN_A2B = 0
                    for kkk in range(opt.batchSize):
                        temp = criterion_GAN(pred_fake[kkk].view(1, -1), target_real[kkk].view(1))
                        loss_GAN_A2B += temp * weight_A[kkk].float()
                        if batch % 100 == 0:
                            print("sample_A loss: ", float(temp), " weight_A: ", float(weight_A[kkk]))
                    # loss_GAN_A2B /= opt.batchSize
                    # loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

                    fake_A = netG_B2A(real_B)
                    pred_fake = netD_A(fake_A)
                    # weight_B = F.softmax(netC_B(fake_A),dim=0)
                    # weight_real_A = F.log_softmax(netC_B(real_A), dim=0).view(1,-1)
                    # loss_kl_B = criterion_kl(weight_real_A, target_distribution)*500
                    init_loss = torch.tensor(
                        [criterion_GAN(pred_fake[_].view(1, -1), target_real[_].view(1)) for _ in range(opt.batchSize)]).to(
                        "cuda")
                    weight_B = F.softmax(-init_loss*10)
                    loss_GAN_B2A = 0
                    for kkk in range(opt.batchSize):
                        loss_GAN_B2A += criterion_GAN(pred_fake[kkk].view(1, -1), target_real[kkk].view(1)) * weight_B[kkk].float()
                    # loss_GAN_B2A /= opt.batchSize
                    # loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

                    # pred_sentiment_real_B = netS_B(real_B)
                    # pred_sentiment_real_A = netS_A(real_A)
                    pred_sentiment_fake_B = netS_B(fake_B)
                    # pred_sentiment_fake_A = netS_A(fake_A)
                    sentiment_loss_A2B = criterion_GAN(pred_sentiment_fake_B, A_sentiments)
                    # sentiment_loss_B2A = criterion_GAN(pred_sentiment_fake_A, B_sentiments)
                    # S_B_loss = criterion_GAN(pred_sentiment_real_B, B_sentiments)
                    # S_A_loss = criterion_GAN(pred_sentiment_real_A, A_sentiments)

                    # Cycle loss
                    recovered_A = netG_B2A(fake_B)
                    loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

                    recovered_B = netG_A2B(fake_A)
                    loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

                    # Total loss
                    loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB \
                             + sentiment_loss_A2B + loss_identity_A + loss_identity_B
                    if batch % 10 > 2:
                        optimizer_G.zero_grad()
                        optimizer_S_B.zero_grad()
                        # optimizer_C_A.zero_grad()
                        # optimizer_C_B.zero_grad()
                        loss_G.backward(retain_graph=True)

                        optimizer_G.step()
                        optimizer_S_B.step()

                        # ###### CMSS A ######
                        # optimizer_C_A.step()
                        #
                        # ###### CMSS B ######
                        # optimizer_C_B.step()


                        ######Sentiment B######
                        # optimizer_S_B.zero_grad()
                        # S_B_loss.backward()
                        # optimizer_S_B.step()
                        #
                        # ######Sentiment A######
                        # optimizer_S_A.zero_grad()
                        # S_A_loss.backward()
                        # optimizer_S_A.step()

                        # Progress report (http://localhost:8097)
                        print({'lan': lan, 'target': target, 'epoch': epoch, 'batch': batch})
                        print({'loss_G': float(loss_G), 'loss_G_GAN': float(loss_GAN_A2B + loss_GAN_B2A),
                               'loss_G_cycle': float(loss_cycle_ABA + loss_cycle_BAB),
                               'loss_sentiment': float(sentiment_loss_A2B)})
                    ###################################
                    else:

                        ###### Discriminator A ######
                        optimizer_D_A.zero_grad()

                        # Real loss
                        pred_real = netD_A(real_A)
                        loss_D_real = criterion_GAN(pred_real, target_real)

                        # Fake loss
                        fake_A = fake_A_buffer.push_and_pop(fake_A)
                        pred_fake = netD_A(fake_A.detach())
                        loss_D_fake = criterion_GAN(pred_fake, target_fake)

                        # Total loss
                        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
                        loss_D_A.backward()

                        optimizer_D_A.step()
                        ###################################

                        ###### Discriminator B ######
                        optimizer_D_B.zero_grad()

                        # Real loss
                        pred_real = netD_B(real_B)
                        loss_D_real = criterion_GAN(pred_real, target_real)

                        # Fake loss
                        fake_B = fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = netD_B(fake_B.detach())
                        loss_D_fake = criterion_GAN(pred_fake, target_fake)

                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
                        loss_D_B.backward()

                        optimizer_D_B.step()
                        print({'lan': lan, 'target': target, 'epoch': epoch, 'batch': batch})
                        print({'loss_D_A': float(loss_D_A), 'loss_D_B': float(loss_D_B)})
                    ###################################

            test_len = len(test_sentiments)
            acc = 0
            for batch in range(test_len):
                review = torch.tensor([test_reviews[batch]], requires_grad=False).to("cuda")
                sentiment = test_sentiments[batch]
                pred = netS_B(review)[0]
                pred_sentiment = int(pred[1]>pred[0])
                if pred_sentiment == sentiment:
                    acc +=1
            res[target*3+lan] = max(res[target*3+lan], acc/test_len)
            print("language: ", lan, "domain: ", target, "epoch: ", epoch, "accuracy: ", acc/test_len, "res: ", res)

            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()
            # lr_scheduler_S_A.step()
            lr_scheduler_S_B.step()
            # lr_scheduler_C_A.step()
            # lr_scheduler_C_B.step()



            # Save models checkpoints
            torch.save(netG_A2B.module.state_dict(),
                       '/root/data/xiaoyang/PyTorch-CycleGAN-master/model/GAN5/' + str(lan) + '/' + str(target) + '/netG_A2B.pth')
            torch.save(netG_B2A.module.state_dict(),
                       '/root/data/xiaoyang/PyTorch-CycleGAN-master/model/GAN5/' + str(lan) + '/' + str(target) + '/netG_B2A.pth')
            torch.save(netD_A.module.state_dict(),
                       '/root/data/xiaoyang/PyTorch-CycleGAN-master/model/GAN5/' + str(lan) + '/' + str(target) + '/netD_A.pth')
            torch.save(netD_B.module.state_dict(),
                       '/root/data/xiaoyang/PyTorch-CycleGAN-master/model/GAN5/' + str(lan) + '/' + str(target) + '/netD_B.pth')
    ###################################
