import numpy as np
import time
import argparse
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F
from scipy.sparse import coo_matrix
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
matrix = [i.tolist for i in matrix]
data_labels = [[int(j[0]) for j in i] for i in data_labels]
