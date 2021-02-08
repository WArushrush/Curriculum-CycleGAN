import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from model2 import Encoder

root="/root/data/xiaoyang/five-domains"
data_name=["custrev","laptop","restaurant","rt-polarity","stsa_binary"]
encoder_index=[13,13,9,14,12]
print("import finished")

for i in range(5):
    print("source: ",i)
    encoder = Encoder()
    encoder.load_state_dict(torch.load("/root/data/xiaoyang/PyTorch-CycleGAN-master/model/seq2seq/"
                                       +str(i)+"/encoder_"+str(encoder_index[i])+".pth"))
    encoder=encoder.to("cuda")
    print("train set")
    f=open(root+"/"+data_name[i]+"_train.pkl",'rb')
    temp=pickle.load(f)
    f.close()
    reviews=temp[0]
    sentiments=temp[1]
    embeddings=[]
    f=open(root+"/"+data_name[i]+"_train_embedding.pkl",'wb+')
    cnt=0
    for review in reviews:
        review=torch.tensor(review,requires_grad=False).to("cuda")
        encoder_output,encoder_hidden = encoder(review)
        embedding = encoder_hidden[0].view(1,-1).tolist()
        cnt+=1
        print("train set sentence: ",cnt)
        embeddings.append(embedding)
    temp=[embeddings,sentiments]
    pickle.dump(temp,f)
    f.close()
    print("test set")
    f = open(root + "/" + data_name[i] + "_test.pkl", 'rb')
    temp = pickle.load(f)
    f.close()
    reviews = temp[0]
    sentiments = temp[1]
    embeddings = []
    f = open(root + "/" + data_name[i] + "_test_embedding.pkl", 'wb+')
    cnt=0
    for review in reviews:
        review = torch.tensor(review, requires_grad=False).to("cuda")
        encoder_output, encoder_hidden = encoder(review)
        embedding = encoder_hidden[0].view(1,-1).tolist()
        cnt+=1
        print("test set sentence: ",cnt)
        embeddings.append(embedding)
    temp = [embeddings, sentiments]
    pickle.dump(temp, f)
    f.close()