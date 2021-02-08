import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from models import Generator
import matplotlib.pyplot as plt
net_G = Generator().to("cuda")
net_G.load_state_dict(
    torch.load('/root/data/xiaoyang/PyTorch-CycleGAN-master/model/GAN/0/netG_A2B.pth'))

switch = 1

f = open("/root/data/xiaoyang/five-domains/custrev_train_embedding_3.pkl", 'rb')
temp = pickle.load(f)
reviews1 = temp[0][:100]
reviews1 = [review[0] for review in reviews1]
sentiments1 = temp[1]
f.close()
f = open("/root/data/xiaoyang/five-domains/restaurant_train_embedding_3.pkl", 'rb')
temp = pickle.load(f)
reviews2 = temp[0][:100]
reviews2 = [review[0] for review in reviews2]
sentiments2 = temp[1]
sentiments3 = sentiments2
f.close()
reviews3 = []
for i in range(100):
    review = reviews2[i]
    review = torch.tensor([review], requires_grad=False).to("cuda")
    review = net_G(review)
    review = review.squeeze(0).tolist()
    reviews3.append(review)
reviews1 = np.array(reviews1)
reviews2 = np.array(reviews2)
reviews3 = np.array(reviews3)
print("training PCA")
pca = PCA(n_components=2)
pca.fit(reviews1)
reviews1 = pca.fit_transform(reviews1)
pca = PCA(n_components=2)
pca.fit(reviews2)
reviews2 = pca.fit_transform(reviews2)
pca = PCA(n_components=2)
pca.fit(reviews3)
reviews3 = pca.fit_transform(reviews3)
positive_reviews1 = np.array([reviews1[i].tolist() for i in range(100) if sentiments1[i]==1])
negative_reviews1 = np.array([reviews1[i].tolist() for i in range(100) if sentiments1[i]==0])
positive_reviews2 = np.array([reviews2[i].tolist() for i in range(100) if sentiments2[i]==1])
negative_reviews2 = np.array([reviews2[i].tolist() for i in range(100) if sentiments2[i]==0])
positive_reviews3 = np.array([reviews3[i].tolist() for i in range(100) if sentiments3[i]==1])
negative_reviews3 = np.array([reviews3[i].tolist() for i in range(100) if sentiments3[i]==0])
print("ploting")


if switch:
    plt.scatter(positive_reviews1[:,0], positive_reviews1[:,1], marker = 'o',color = 'red', s = 40 ,label = 'positive_custrev')
    plt.scatter(negative_reviews1[:,0], negative_reviews1[:,1], marker = 'o',color = 'blue', s = 40 ,label = 'negative_custrev')
    plt.scatter(positive_reviews2[:,0], positive_reviews2[:,1], marker = 'o',color = 'pink', s = 40 ,label = 'positive_restaurant')
    plt.scatter(negative_reviews2[:,0], negative_reviews2[:,1], marker = 'o',color = 'powderblue', s = 40 ,label = 'negative_restaurant')
    plt.legend(loc='upper right', title='custrev & restaurant', framealpha=10)
    plt.savefig('/root/data/xiaoyang/restaurant_custrev.jpg')
else:
    plt.scatter(positive_reviews1[:,0], positive_reviews1[:,1], marker = 'o',color = 'red', s = 40 ,label = 'positive_custrev')
    plt.scatter(negative_reviews1[:,0], negative_reviews1[:,1], marker = 'o',color = 'blue', s = 40 ,label = 'negative_custrev')
    plt.scatter(positive_reviews3[:,0], positive_reviews3[:,1], marker = 'o',color = 'pink', s = 40 ,label = 'transfered-positive_restaurant')
    plt.scatter(negative_reviews3[:,0], negative_reviews3[:,1], marker = 'o',color = 'powderblue', s = 40 ,label = 'transfered-negative_restaurant')
    plt.legend(loc='upper right',title='custrev & transfered-restaurant', framealpha=10)
    plt.savefig('/root/data/xiaoyang/restaurant2custrev.jpg')
