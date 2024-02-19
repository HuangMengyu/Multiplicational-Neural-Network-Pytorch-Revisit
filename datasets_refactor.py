# -*- coding: utf-8 -*-
"""
datasets generation

@author: Mengyu Huang
"""

from torch.utils.data import Dataset
import os
import pickle
import scipy.fftpack as FFT
import numpy as np


# def dense_to_one_hot(labels_dense, num_classes):
#     """Convert class labels from scalars to one-hot vectors."""
#     labels_one_hot = []
#     for i in range(num_classes):
#         if i == labels_dense:
#             labels_one_hot.append(1)
#         else:
#             labels_one_hot.append(0)
#     return labels_one_hot


def pickle_2_img_single(data_file, win_size=32): #rewrite current pkl file
    if not os.path.exists(data_file):
        print('file {0} not exists'.format(data_file))
        exit()
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    total_x,  total_y = [], []
    for i in range(len(data)):  #10
        for j in range(len(data[i]['labels'])):  #num of samples
            img = data[i]['img'][j]

            img = FFT.fftn(img)
            img = FFT.fftshift(img)

            img1 = img.real
            #print(img1.shape[0] /2, img1.shape[1]/2, win_size/2)
            #exit(0)
            img1 = img1[int(img1.shape[0]/2 - win_size/2): int(img1.shape[0]/2 + win_size/2),
                   int(img1.shape[1]/2 - win_size/2): int(img1.shape[1]/2 + win_size/2)]
            img1 = np.expand_dims(img1, axis=-1)

            img2 = img.imag
            img2 = img2[int(img2.shape[0] / 2 - win_size / 2): int(img2.shape[0] / 2 + win_size / 2),
                   int(img2.shape[1] / 2 - win_size / 2): int(img2.shape[1] / 2 + win_size / 2)]
            img2 = np.expand_dims(img2, axis=-1)

            x = np.concatenate([img1, img2], axis=-1)

            # print(data[i]['labels'][j])
            label = int(data[i]['labels'][j])

            total_x.append(x)
            total_y.append(label)
    total_x = np.asarray(total_x)
    total_y = np.asarray(total_y)
    return total_x, total_y


read_dirs = ["pkl\ckp_with_img_geometry.pkl", "pkl/oulu_casia_with_img_geometry.pkl"]
out_dirs = ["new_pkl\ckp_6.pkl", "new_pkl/oulu_casia.pkl"]

for i in range(len(read_dirs)):
    print(i)
    x,y = pickle_2_img_single(read_dirs[i], win_size=32)
    print (dir, x.shape, y.shape)
    data = {'imgs': x, 'labels': y}
    with open(out_dirs[i], 'wb') as f:
        pickle.dump(data, f)








