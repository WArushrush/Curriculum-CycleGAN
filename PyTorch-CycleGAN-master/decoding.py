import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Generator
from model2 import Encoder, Decoder
import pickle

device = torch.device("cuda")
root="/root/data/xiaoyang/five-domains"
data_name=["custrev","laptop","restaurant","rt-polarity","stsa_binary"]

generator = Generator().to("cuda")
encoder = Encoder().to("cuda")
decoder = Decoder().to("cuda")
generator.load_state_dict(
        torch.load('/root/data/xiaoyang/PyTorch-CycleGAN-master/model/GAN/1/netG_A2B.pth'))
encoder.load_state_dict(
    torch.load('/root/data/xiaoyang/PyTorch-CycleGAN-master/model/seq2seq_3/2/encoder_15.pth'))
decoder.load_state_dict(
    torch.load('/root/data/xiaoyang/PyTorch-CycleGAN-master/model/seq2seq_3/1/decoder_15.pth'))
f = open(root+"/restaurant_train.pkl",'rb')
temp=pickle.load(f)
reviews = temp[0]
f=open("/root/data/xiaoyang/five-domains/index_dict.pkl",'rb')
index_dict=pickle.load(f)
f.close()
MAX_LENGTH=106
SOS_token=29205
EOS_token=29206


def train(input_tensor,encoder, decoder):
    original=[]
    for word in input_tensor.tolist():
        original.append(index_dict[word])
    print("original_sentence: ",original)
    target_length = MAX_LENGTH
    encoder_output,encoder_hidden = encoder(input_tensor)
    embedding = encoder_hidden[0].view(1,-1)
    embedding = generator(embedding).view(1,1,-1)
    embedding = embedding.view(1,1,-1)
    decoder_input = torch.tensor([[SOS_token]]).to(device) # SOS为标记句首
    decoder_hidden = (embedding,torch.zeros(1,1,1024).to(device)) # 把编码的最终状态作为解码的初始状态
    sentence=[]
    for di in range(target_length): # 每次预测一个元素
        decoder_output, decoder_hidden= decoder(
        decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1) # 将可能性最大的预测值加入译文序列
        temp = topi.squeeze().detach()
        sentence.append(int(temp))
        decoder_input=temp
        if decoder_input.item()==EOS_token:
            break
    for idx in range(len(sentence)):
        sentence[idx]=index_dict[sentence[idx]]
    print("transfered sentence: ",sentence)
    return


for review in reviews:
    cur = torch.tensor(review).view(-1).to("cuda")
    train(cur, encoder, decoder)
