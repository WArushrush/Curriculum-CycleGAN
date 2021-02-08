import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random
import os
import torch.distributed as dist
import torch.utils.data as Data
from model2 import Encoder, Discriminator, Emb
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device=torch.device("cuda")

MAX_LENGTH=-1
SOS_token=-1
EOS_token=-1
root="/root/data/xiaoyang/PyTorch-CycleGAN-master/cls-acl10-unprocessed"
languages = ['de', 'en', 'fr', 'jp']
data_name=['books', 'dvd', 'music']
# word_cnt = [
#     [79255, 81021, 77199],
#     [64666, 68853, 59093],
#     [54871, 58168, 63119],
#     [21540, 21877, 20928]
# ]
word_cnt = [
    [481745, 480348, 461154],
    [321175, 240515, 205236],
    [203857, 124814, 158917],
    [310947, 290222, 242701]
]
# word_cnt = [
#     [192058, 204704, 186648],
#     [141591, 132336, 117507],
#     [118209, 124814, 131403],
#     [87156, 82905, 75042]
# ]


print("finished")

def train(input_tensor,encoder, decoder,prob=0.5,criterion=F.nll_loss):
    target_tensor=input_tensor.clone()
    # original=[]
    # for word in input_tensor.tolist():
    #     original.append(index_dict[word])
    # print("original_sentence: ",original)
    target_length = target_tensor.size(0)
    loss = 0
    encoder_output,encoder_hidden = encoder(input_tensor)
    # print("encoder_output: ", encoder_output.shape)
    # print("encoder_hidden: ", encoder_hidden[0].shape, encoder_hidden[1].shape)
    embedding = encoder_hidden[0].view(1,-1)
    decoder_input = torch.tensor([[SOS_token]]).to(device) # SOS为标记句首
    decoder_hidden = (encoder_hidden[0].view(1,1,-1),torch.zeros(1,1,1200).to("cuda")) # 把编码的最终状态作为解码的初始状态
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
            decoder_input=torch.tensor([[input_tensor.squeeze(0).tolist()[di]]]).to(device)
        loss+=criterion(decoder_output, torch.tensor([target_tensor.squeeze(0).tolist()[di]]).to(device))
        if decoder_input.item()==EOS_token:
            break
    # for idx in range(len(sentence)):
    #     sentence[idx]=index_dict[sentence[idx]]
    # print("autoencoded sentence: ",sentence)
    return loss/len(sentence),embedding


num_epoch = 20
batch_size = 32
print("start")
for language_idx in range(4):
    for source_idx in range(3):
        encoder = Encoder(language=languages[language_idx], domain=data_name[source_idx], input_size=word_cnt[language_idx][source_idx] + 3).to("cuda")
        emb = Emb(language=languages[language_idx], domain=data_name[source_idx], input_size=word_cnt[language_idx][source_idx] + 3).to("cuda")
        # decoder = Decoder(language=languages[language_idx], domain=data_name[source_idx], output_size=word_cnt[language_idx][source_idx] + 2).to("cuda")
        sentiment_classifier = Discriminator().to("cuda")
        # encoder = nn.DataParallel(encoder, device_ids=[0, 1])
        emb = nn.DataParallel(emb, device_ids=[0,1])
        # decoder = nn.DataParallel(decoder, device_ids=[0, 1])
        sentiment_classifier = nn.DataParallel(sentiment_classifier, device_ids=[0, 1])
        SOS_token = word_cnt[language_idx][source_idx]
        EOS_token = word_cnt[language_idx][source_idx] + 1
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)
        # decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001, betas=(0.5, 0.999))
        sentiment_classifier_optimizer = torch.optim.Adam(sentiment_classifier.parameters(),
                                                          lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)
        lr_schedular_encoder = torch.optim.lr_scheduler.StepLR(encoder_optimizer,
                                                               step_size=10, gamma=0.5)
        # lr_schedular_decoder = torch.optim.lr_scheduler.StepLR(decoder_optimizer,
        #                                                       step_size=20, gamma=0.5)
        lr_schedular_sentiment_classifier = torch.optim.lr_scheduler.StepLR(sentiment_classifier_optimizer,
                                                               step_size=10, gamma=0.5)
        f = open(root + "/" + languages[language_idx] + "/" + data_name[source_idx] + "/train.pkl",'rb')
        temp=pickle.load(f)
        reviews=temp[0]
        sentiments=temp[1]
        f.close()
        f = open(root + "/" + languages[language_idx] + "/" + data_name[source_idx] + "/test.pkl", 'rb')
        temp = pickle.load(f)
        test_reviews = temp[0]
        test_sentiments = temp[1]
        f.close()
        # f = open(root + "/yahoo_train.pkl", 'rb')
        # temp = pickle.load(f)
        # yahoo_reviews = temp[0]
        # yahoo_reviews = [yahoo_reviews[i] for i in range(10000)]
        # f.close()
        # f = open(root + "/yelp_train.pkl", 'rb')
        # temp = pickle.load(f)
        # yelp_reviews = temp[0]
        # yelp_reviews = [yelp_reviews[i] for i in range(10000)]
        # f.close()
        for epoch in range(num_epoch):
            for batch in range(len(reviews)//batch_size):
                # feature_loss = 0
                sentiment_loss = 0
                original = []
                sentiment = []
                max_len = 0
                pad_idx = word_cnt[language_idx][source_idx] + 2
                review_lengths = []
                for review_idx in range(batch*batch_size,(batch+1)*batch_size):
                    review_idx%=len(reviews)
                    review=reviews[review_idx]
                    # sentiment=sentiments[review_idx]
                    # original=torch.tensor(review, requires_grad=False).to(device)
                    sentiment.append(sentiments[review_idx])
                    original.append(review)
                    max_len = max(max_len, len(review))
                    review_lengths.append(len(review))
                for idx in range(batch_size):
                    original[idx] += [pad_idx] * (max_len - len(original[idx]))
                original = torch.tensor(original, requires_grad=False).to(device)
                # print("before emb: ", original.shape)
                # print("review_lengths: ", sum(review_lengths))
                original = emb(original)
                # print("embed", original.shape)
                _, idx_sort = torch.sort(torch.tensor(review_lengths), dim=0, descending=True)
                # _, idx_unsort = torch.sort(idx_sort, dim=0)
                idx_sort = idx_sort.to(device)

                original = original.index_select(0, idx_sort)
                sentiment = [sentiment[int(idx)] for idx in idx_sort]
                review_lengths = [review_lengths[int(idx)] for idx in idx_sort]
                # original = nn.utils.rnn.pack_padded_sequence(input=original, lengths=review_lengths, batch_first=True)
                # loss,embedding = train(original,encoder,decoder)
                original = nn.utils.rnn.pack_padded_sequence(input=original, lengths=review_lengths, batch_first=True)
                encoder_output, encoder_hidden = encoder(original)
                # print(encoder_hidden[0].shape)
                embedding = torch.transpose(encoder_hidden[0], 0, 1).reshape(batch_size, -1)
                # feature_loss+=loss
                predict = sentiment_classifier(embedding)
                # print(predict.shape)
                sentiment_loss = F.nll_loss(predict,torch.tensor(sentiment).to(device))
                # feature_loss/=batch_size
                # sentiment_loss/=batch_size
                print("language: ", language_idx, "domain: ", source_idx, "epoch: ",epoch," batch: ", batch,
                      " sentiment loss: ",sentiment_loss)

                # encoder_optimizer.zero_grad()
                # decoder_optimizer.zero_grad()
                # feature_loss.backward(retain_graph=True)
                # encoder_optimizer.step()
                # decoder_optimizer.step()

                sentiment_classifier_optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                sentiment_loss.backward()
                nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=20, norm_type=2)
                encoder_optimizer.step()
                sentiment_classifier_optimizer.step()
                # except Exception as e:
                #     print(e)
            # for batch in range(len(yahoo_reviews)//batch_size):
            #     try:
            #         feature_loss = 0
            #         encoder_optimizer.zero_grad()
            #         decoder_optimizer.zero_grad()
            #         for review_idx in range(batch*batch_size,(batch+1)*batch_size):
            #             review_idx%=len(yahoo_reviews)
            #             review=yahoo_reviews[review_idx]
            #             original=torch.tensor(review).to(device)
            #             loss,embedding = train(original,encoder,decoder)
            #             feature_loss+=loss
            #         feature_loss/=batch_size
            #         print("source: ", source_idx, "epoch: ",epoch," yahoo batch: ",batch," feature loss: ", feature_loss.item())
            #         feature_loss.backward(retain_graph=True)
            #         encoder_optimizer.step()
            #         decoder_optimizer.step()
            #     except Exception as e:
            #         print(e)
            # for batch in range(len(yelp_reviews) // batch_size):
            #     try:
            #         feature_loss = 0
            #         encoder_optimizer.zero_grad()
            #         decoder_optimizer.zero_grad()
            #         for review_idx in range(batch * batch_size, (batch + 1) * batch_size):
            #             review_idx %= len(yelp_reviews)
            #             review = yelp_reviews[review_idx]
            #             original = torch.tensor(review).to(device)
            #             loss, embedding = train(original, encoder, decoder)
            #             feature_loss += loss
            #         feature_loss /= batch_size
            #         print("source: ", source_idx, "epoch: ", epoch, " yelp batch: ", batch, " feature loss: ", feature_loss.item())
            #         feature_loss.backward(retain_graph=True)
            #         encoder_optimizer.step()
            #         decoder_optimizer.step()
            #     except Exception as e:
            #         print(e)
            cnt=0
            acc=0
            for batch in range(len(test_reviews)//batch_size):
                try:
                    # feature_loss = 0
                    sentiment_loss = 0
                    encoder_optimizer.zero_grad()
                    # decoder_optimizer.zero_grad()
                    sentiment_classifier_optimizer.zero_grad()
                    for review_idx in range(batch*batch_size,(batch+1)*batch_size):
                        review_idx%=len(test_reviews)
                        review=[test_reviews[review_idx]]
                        sentiment=test_sentiments[review_idx]
                        original=torch.tensor(review, requires_grad=False).to(device)
                        # loss,embedding = train(original,encoder,decoder)
                        # feature_loss+=loss
                        original = emb(original)
                        encoder_output, encoder_hidden = encoder(original)
                        embedding = encoder_hidden[0].view(1,-1)
                        predict = sentiment_classifier(embedding)
                        predict_label = int(int(predict[0][1]) > int(predict[0][0]))
                        sentiment_loss += F.nll_loss(predict,torch.tensor([sentiment]).to(device))*10
                        cnt += 1
                        if predict_label == int(sentiment):
                            acc += 1
                    # feature_loss/=batch_size
                    # sentiment_loss/=batch_size
                    # feature_loss.backward()
                    # encoder_optimizer.step()
                    # decoder_optimizer.step()
                except Exception as e:
                    print(e)

            lr_schedular_encoder.step()
            # lr_schedular_decoder.step()
            lr_schedular_sentiment_classifier.step()
            print("epoch: ",epoch," test set accuracy: ",str(acc/cnt))
            torch.save(emb.module.state_dict(),
                       '/root/data/xiaoyang/PyTorch-CycleGAN-master/model/seq2seq_5/' + str(language_idx) + "/" + str(
                           source_idx) + "/embedding_" + str(epoch + 1) + ".pth")
            torch.save(encoder.state_dict(), '/root/data/xiaoyang/PyTorch-CycleGAN-master/model/seq2seq_5/'+str(language_idx)+"/"+str(
                source_idx)+"/encoder_"+str(epoch+1)+".pth")
            # torch.save(decoder.module.state_dict(), '/root/data/xiaoyang/PyTorch-CycleGAN-master/model/seq2seq_5/'+str(language_idx)+"/" + str(
            #     source_idx) + "/decoder_" + str(epoch+1) + ".pth")
            torch.save(sentiment_classifier.module.state_dict(), '/root/data/xiaoyang/PyTorch-CycleGAN-master/model/seq2seq_5/'+str(language_idx)+"/" + str(
                source_idx) + "/sentiment_classifier_" + str(epoch+1) + ".pth")