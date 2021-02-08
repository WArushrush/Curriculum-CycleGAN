import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from models import Generator
from model2 import Encoder
import matplotlib.pyplot as plt

# encoder = Encoder().to("cuda")
# encoder.load_state_dict(
#     torch.load('/root/data/xiaoyang/PyTorch-CycleGAN-master/model/GAN/0/encoder.pth'))
net_G = Generator().to("cuda")
net_G.load_state_dict(
    torch.load('/root/data/xiaoyang/PyTorch-CycleGAN-master/model/GAN/2/netG_A2B.pth'))

switch = 0

f = open("/root/data/xiaoyang/five-domains/restaurant_train_embedding.pkl", 'rb')
temp = pickle.load(f)
# temp_reviews = temp[0][:200]
# temp_sentiments = temp[1][:200]
# for i in range(200):
#     temp_reviews[i] = torch.tensor(temp_reviews[i], requires_grad=False).to("cuda")
#     encoder_output, encoder_hidden = encoder(temp_reviews[i])
#     temp_reviews[i] = encoder_hidden[0].view(1, -1).tolist()
reviews1 = [review[0] for review in temp[0]]
sentiments1 = temp[1]
f.close()
f = open("/root/data/xiaoyang/five-domains/laptop_train_embedding.pkl", 'rb')
temp = pickle.load(f)
# temp_reviews = temp[0][:200]
# temp_sentiments = temp[1][:200]
# for i in range(200):
#     temp_reviews[i] = torch.tensor(temp_reviews[i], requires_grad=False).to("cuda")
#     encoder_output, encoder_hidden = encoder(temp_reviews[i])
#     temp_reviews[i] = encoder_hidden[0].view(1, -1).tolist()
# reviews2 = temp_reviews
reviews2 = [review[0] for review in temp[0]]
sentiments2 = temp[1]
f.close()
reviews3 = []
for i in range(200):
    review = reviews2[i]
    review = torch.tensor([review], requires_grad=False).to("cuda")
    review = net_G(review)
    review = review.squeeze(0).tolist()
    reviews3.append(review)
total = reviews1 + reviews2 + reviews3
total = np.array(total)
# reviews1 = np.array(reviews1)
# reviews2 = np.array(reviews2)
# reviews3 = np.array(reviews3)
# print("training PCA")
# pca = PCA(n_components=2)
# pca.fit(total)
# total = pca.fit_transform(total)
print("training tSNE")
tsne = TSNE(n_components=2)
tsne.fit(total)
total = tsne.fit_transform(total)
reviews1 = total[:200]
reviews2 = total[200:400]
reviews3 = total[400:600]
# pca = PCA(n_components=2)
# pca.fit(reviews1)
# reviews1 = pca.fit_transform(reviews1)
# pca = PCA(n_components=2)
# pca.fit(reviews2)
# reviews2 = pca.fit_transform(reviews2)
# pca = PCA(n_components=2)
# pca.fit(reviews3)
# reviews3 = pca.fit_transform(reviews3)
positive_reviews1 = np.array([reviews1[i].tolist() for i in range(200) if sentiments1[i]==1])
negative_reviews1 = np.array([reviews1[i].tolist() for i in range(200) if sentiments1[i]==0])
positive_reviews2 = np.array([reviews2[i].tolist() for i in range(200) if sentiments2[i]==1])
negative_reviews2 = np.array([reviews2[i].tolist() for i in range(200) if sentiments2[i]==0])
positive_reviews3 = np.array([reviews3[i].tolist() for i in range(200) if sentiments2[i]==1])
negative_reviews3 = np.array([reviews3[i].tolist() for i in range(200) if sentiments2[i]==0])
print("ploting")


if switch:
    plt.scatter(positive_reviews1[:,0], positive_reviews1[:,1], marker = 'o',color = 'red', s = 40 ,label = 'positive_restaurant')
    plt.scatter(negative_reviews1[:,0], negative_reviews1[:,1], marker = 'o',color = 'blue', s = 40 ,label = 'negative_restaurant')
    plt.scatter(positive_reviews2[:,0], positive_reviews2[:,1], marker = 'o',color = 'pink', s = 40 ,label = 'positive_laptop')
    plt.scatter(negative_reviews2[:,0], negative_reviews2[:,1], marker = 'o',color = 'powderblue', s = 40 ,label = 'negative_laptop')
    plt.legend(loc='upper right', title='restaurant & laptop', framealpha=10)
    plt.savefig('/root/data/xiaoyang/laptop_restaurant.jpg')
else:
    plt.scatter(positive_reviews1[:,0], positive_reviews1[:,1], marker = 'o',color = 'red', s = 40 ,label = 'positive_restaurant')
    plt.scatter(negative_reviews1[:,0], negative_reviews1[:,1], marker = 'o',color = 'blue', s = 40 ,label = 'negative_restaurant')
    plt.scatter(positive_reviews3[:,0], positive_reviews3[:,1], marker = 'o',color = 'pink', s = 40 ,label = 'transfered-positive_laptop')
    plt.scatter(negative_reviews3[:,0], negative_reviews3[:,1], marker = 'o',color = 'powderblue', s = 40 ,label = 'transfered-negative_laptop')
    plt.legend(loc='upper right',title='restaurant & transfered-laptop', framealpha=10)
    plt.savefig('/root/data/xiaoyang/laptop2restaurant.jpg')
