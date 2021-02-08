import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random
from model2 import Encoder,Decoder,Discriminator
device=torch.device("cuda")

MAX_LENGTH=106
SOS_token=55583
EOS_token=55584
f=open("/root/data/xiaoyang/five-domains/index_dict.pkl",'rb')
index_dict=pickle.load(f)
f.close()
root="/root/data/xiaoyang/five-domains"
data_name=["custrev","laptop","restaurant","rt-polarity","stsa_binary"]
print("finished")

def train(input_tensor,encoder, decoder,prob=0.5,criterion=F.nll_loss):
    target_tensor=input_tensor.clone()
    original=[]
    for word in input_tensor.tolist():
        original.append(index_dict[word])
    # print("original_sentence: ",original)
    target_length = target_tensor.size(0)
    loss = 0
    encoder_output,encoder_hidden = encoder(input_tensor)
    embedding = encoder_hidden[0].view(1,-1)
    decoder_input = torch.tensor([[SOS_token]]).to(device) # SOS为标记句首
    decoder_hidden = (encoder_hidden[0].view(1,1,-1),torch.zeros(1,1,1024).to("cuda")) # 把编码的最终状态作为解码的初始状态
    sentence=[]
    for di in range(target_length): # 每次预测一个元素
        decoder_output, decoder_hidden= decoder(
        decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1) # 将可能性最大的预测值加入译文序列
        temp = topi.squeeze().detach()
        sentence.append(int(temp))
        rand=random.random()
        if rand>prob:
            decoder_input=temp
        else:
            decoder_input=torch.tensor([[input_tensor.tolist()[di]]]).to(device)
        loss+=criterion(decoder_output, torch.tensor([target_tensor.tolist()[di]]).to(device))
        if decoder_input.item()==EOS_token:
            break
    for idx in range(len(sentence)):
        sentence[idx]=index_dict[sentence[idx]]
    # print("autoencoded sentence: ",sentence)
    return loss/len(sentence),embedding


num_epoch = 30
batch_size = 16
print("start")
for source_idx in range(1,2):
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    sentiment_classifier = Discriminator().to(device)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001, betas=(0.5, 0.999))
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001, betas=(0.5, 0.999))
    sentiment_classifier_optimizer = torch.optim.Adam(sentiment_classifier.parameters(),
                                                      lr=0.0001, betas=(0.5, 0.999))
    lr_schedular_encoder = torch.optim.lr_scheduler.StepLR(encoder_optimizer,
                                                           step_size=20, gamma=0.5)
    lr_schedular_decoder = torch.optim.lr_scheduler.StepLR(decoder_optimizer,
                                                           step_size=20, gamma=0.5)
    lr_schedular_sentiment_classifier = torch.optim.lr_scheduler.StepLR(sentiment_classifier_optimizer,
                                                           step_size=20, gamma=0.5)
    f=open(root+"/"+data_name[source_idx]+"_train.pkl",'rb')
    temp=pickle.load(f)
    reviews=temp[0]
    sentiments=temp[1]
    f.close()
    f = open(root + "/" + data_name[source_idx] + "_test.pkl", 'rb')
    temp = pickle.load(f)
    test_reviews = temp[0]
    test_sentiments = temp[1]
    f.close()
    f = open(root + "/yahoo_train.pkl", 'rb')
    temp = pickle.load(f)
    yahoo_reviews = temp[0]
    yahoo_reviews = [yahoo_reviews[i] for i in range(10000)]
    f.close()
    f = open(root + "/yelp_train.pkl", 'rb')
    temp = pickle.load(f)
    yelp_reviews = temp[0]
    yelp_reviews = [yelp_reviews[i] for i in range(10000)]
    f.close()
    for batch in range(len(yahoo_reviews) // batch_size):
        try:
            feature_loss = 0
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            for review_idx in range(batch * batch_size, (batch + 1) * batch_size):
                review_idx %= len(yahoo_reviews)
                review = yahoo_reviews[review_idx]
                original = torch.tensor(review).to(device)
                loss, embedding = train(original, encoder, decoder)
                feature_loss += loss
            feature_loss /= batch_size
            print(" yahoo batch: ", batch, " feature loss: ",
                  feature_loss.item())
            feature_loss.backward(retain_graph=True)
            encoder_optimizer.step()
            decoder_optimizer.step()
        except Exception as e:
            print(e)
    for batch in range(len(yelp_reviews) // batch_size):
        try:
            feature_loss = 0
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            for review_idx in range(batch * batch_size, (batch + 1) * batch_size):
                review_idx %= len(yelp_reviews)
                review = yelp_reviews[review_idx]
                original = torch.tensor(review).to(device)
                loss, embedding = train(original, encoder, decoder)
                feature_loss += loss
            feature_loss /= batch_size
            print(" yelp batch: ", batch, " feature loss: ",
                  feature_loss.item())
            feature_loss.backward(retain_graph=True)
            encoder_optimizer.step()
            decoder_optimizer.step()
        except Exception as e:
            print(e)
    for epoch in range(num_epoch):
        for batch in range(len(reviews)//batch_size):
            try:
                feature_loss = 0
                sentiment_loss = 0
                # encoder_optimizer.zero_grad()
                # decoder_optimizer.zero_grad()
                # sentiment_classifier_optimizer.zero_grad()
                for review_idx in range(batch*batch_size,(batch+1)*batch_size):
                    review_idx%=len(reviews)
                    review=reviews[review_idx]
                    sentiment=sentiments[review_idx]
                    original=torch.tensor(review).to(device)
                    loss,embedding = train(original,encoder,decoder)
                    feature_loss+=loss
                    predict = sentiment_classifier(embedding)
                    sentiment_loss += F.nll_loss(predict,torch.tensor([sentiment]).to(device))*10
                feature_loss/=batch_size
                sentiment_loss/=batch_size
                print("source: ", source_idx, "epoch: ",epoch," batch: ",batch," feature loss: ", feature_loss.item(),
                      " sentiment loss: ",sentiment_loss)
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                feature_loss.backward(retain_graph=True)
                encoder_optimizer.step()
                decoder_optimizer.step()
                sentiment_classifier_optimizer.zero_grad()
                sentiment_loss.backward()
                encoder_optimizer.step()
                sentiment_classifier_optimizer.step()
            except Exception as e:
                print(e)
        cnt=0
        acc=0
        for batch in range(len(test_reviews)//batch_size):
            try:
                feature_loss = 0
                sentiment_loss = 0
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                sentiment_classifier_optimizer.zero_grad()
                for review_idx in range(batch*batch_size,(batch+1)*batch_size):
                    review_idx%=len(test_reviews)
                    review=test_reviews[review_idx]
                    sentiment=test_sentiments[review_idx]
                    original=torch.tensor(review).to(device)
                    loss,embedding = train(original,encoder,decoder)
                    feature_loss+=loss
                    predict = sentiment_classifier(embedding)
                    predict_label = int(int(predict[0][1]) > int(predict[0][0]))
                    sentiment_loss += F.nll_loss(predict,torch.tensor([sentiment]).to(device))*10
                    cnt += 1
                    if predict_label == int(sentiment):
                        acc += 1
                feature_loss/=batch_size
                sentiment_loss/=batch_size
                # feature_loss.backward()
                # encoder_optimizer.step()
                # decoder_optimizer.step()
            except Exception as e:
                print(e)
    
        lr_schedular_encoder.step()
        lr_schedular_decoder.step()
        lr_schedular_sentiment_classifier.step()
        print("epoch: ",epoch," test set accuracy: ",str(acc/cnt))
        torch.save(encoder.state_dict(), '/root/data/xiaoyang/PyTorch-CycleGAN-master/model/seq2seq/'+str(
            source_idx)+"/encoder_"+str(epoch+1)+".pth")
        torch.save(decoder.state_dict(), '/root/data/xiaoyang/PyTorch-CycleGAN-master/model/seq2seq/' + str(
            source_idx) + "/decoder_" + str(epoch+1) + ".pth")
        torch.save(sentiment_classifier.state_dict(), '/root/data/xiaoyang/PyTorch-CycleGAN-master/model/seq2seq/' + str(
            source_idx) + "/sentiment_classifier_" + str(epoch+1) + ".pth")