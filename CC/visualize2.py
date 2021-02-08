import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from models import Generator
from model2 import Encoder
import matplotlib.pyplot as plt
# encoder = Encoder().to("cuda")
# encoder.load_state_dict(
#     torch.load('/root/data/xiaoyang/PyTorch-CycleGAN-master/model/GAN/0/encoder.pth'))
net_G = Generator().to("cuda")
net_G.load_state_dict(
    torch.load('/root/data/xiaoyang/PyTorch-CycleGAN-master/model/GAN/0/netG_A2B.pth'))

switch = 1

f = open("/root/data/xiaoyang/five-domains/custrev_train_embedding_3.pkl", 'rb')
temp = pickle.load(f)
reviews1 = [review[0] for review in temp[0][:200]]
sentiments1 = temp[1]
f.close()
f = open("/root/data/xiaoyang/five-domains/restaurant_train_embedding_3.pkl", 'rb')
temp = pickle.load(f)
reviews2 = [review[0] for review in temp[0][:200]]
sentiments2 = temp[1]
f.close()
f = open("/root/data/xiaoyang/five-domains/laptop_train_embedding_3.pkl", 'rb')
temp = pickle.load(f)
reviews3 = [review[0] for review in temp[0][:200]]
sentiments3 = temp[1]
f.close()
f = open("/root/data/xiaoyang/five-domains/rt-polarity_train_embedding_3.pkl", 'rb')
temp = pickle.load(f)
reviews4 = [review[0] for review in temp[0][:200]]
sentiments4 = temp[1]
f.close()
f = open("/root/data/xiaoyang/five-domains/stsa_binary_train_embedding_3.pkl", 'rb')
temp = pickle.load(f)
reviews5 = [review[0] for review in temp[0][:200]]
sentiments5 = temp[1]
f.close()
total = reviews1 + reviews2 + reviews3 + reviews4 + reviews5
# reviews1 = np.array(reviews1)
# reviews2 = np.array(reviews2)
# reviews3 = np.array(reviews3)
print("training PCA")
pca = PCA(n_components=2)
pca.fit(total)
total = pca.fit_transform(total)
reviews1 = total[:200]
reviews2 = total[200:400]
reviews3 = total[400:600]
reviews4 = total[600:800]
reviews5 = total[800:1000]
print("ploting")


plt.scatter(reviews1[:,0], reviews1[:,1], marker = 'o',color = 'red', s = 10 ,label = 'custrev')
plt.scatter(reviews2[:,0], reviews2[:,1], marker = 'o',color = 'orange', s = 10 ,label = 'restaurant')
plt.scatter(reviews3[:, 0], reviews3[:, 1], marker='o', color='yellow', s=10, label='laptop')
plt.scatter(reviews4[:, 0], reviews4[:, 1], marker='o', color='green', s=10, label='rt-polarity')
plt.scatter(reviews5[:, 0], reviews5[:, 1], marker='o', color='blue', s=10, label='stsa_binary')
plt.legend(loc='upper right', title='five domains', framealpha=10)
plt.savefig('/root/data/xiaoyang/five-domains.jpg')
