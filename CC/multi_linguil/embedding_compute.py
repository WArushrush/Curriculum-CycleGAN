import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from model2 import Encoder, Emb

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
print("import finished")
for lan in range(4):
    print("lan: ", lan)
    for i in range(3):
        print("    source: ",i)
        emb = Emb(language=languages[lan], domain=data_name[i], input_size=word_cnt[lan][i] + 3)
        encoder = Encoder(language=lan, domain=i, input_size=word_cnt[lan][i] + 3)
        emb.load_state_dict(torch.load("/root/data/xiaoyang/PyTorch-CycleGAN-master/model/seq2seq_5/"
                                           +str(lan)+"/"+str(i)+"/embedding_20.pth"))
        encoder.load_state_dict(torch.load("/root/data/xiaoyang/PyTorch-CycleGAN-master/model/seq2seq_5/"
                                           +str(lan)+"/"+str(i)+"/encoder_20.pth"))
        emb=emb.to("cuda")
        encoder=encoder.to("cuda")
        print("        train set")
        f=open(root+"/"+languages[lan]+"/"+data_name[i]+"/train.pkl",'rb')
        temp=pickle.load(f)
        f.close()
        reviews=temp[0]
        sentiments=temp[1]
        embeddings=[]
        f=open(root+"/"+languages[lan]+"/"+data_name[i]+"_train_embedding_5.pkl",'wb+')
        cnt=0
        for review in reviews:
            review=torch.tensor([review],requires_grad=False).to("cuda")
            review = emb(review)
            encoder_output,encoder_hidden = encoder(review)
            embedding = encoder_hidden[0].view(1,-1).tolist()
            cnt+=1
            print("train set sentence: ",cnt)
            embeddings.append(embedding)
        temp=[embeddings,sentiments]
        pickle.dump(temp,f)
        f.close()
        print("        test set")
        f = open(root + "/" + languages[lan]+"/" + data_name[i] + "/test.pkl", 'rb')
        temp = pickle.load(f)
        f.close()
        reviews = temp[0]
        sentiments = temp[1]
        embeddings = []
        f = open(root + "/" + languages[lan]+"/" + data_name[i] + "_test_embedding_5.pkl", 'wb+')
        cnt=0
        for review in reviews:
            review = torch.tensor([review], requires_grad=False).to("cuda")
            review = emb(review)
            encoder_output, encoder_hidden = encoder(review)
            embedding = encoder_hidden[0].view(1,-1).tolist()
            cnt+=1
            print("test set sentence: ",cnt)
            embeddings.append(embedding)
        temp = [embeddings, sentiments]
        pickle.dump(temp, f)
        f.close()