# -*- coding: utf-8 -*-
"""
inherit Dataset class for my own datasets

@author: Mengyu Huang
"""


from torch.utils.data import Dataset
import os
import pickle
import numpy as np
import scipy.fftpack as FFT


def FourierImage(image):
    img = FFT.fftn(image)
    img = FFT.fftshift(img)
    return img


def FeatureSelection(image, win_size):
    img1 = image.real
    # print(img1.shape[0] /2, img1.shape[1]/2, win_size/2)
    # exit(0)
    img1 = img1[int(img1.shape[0] / 2 - win_size / 2): int(img1.shape[0] / 2 + win_size / 2),
           int(img1.shape[1] / 2 - win_size / 2): int(img1.shape[1] / 2 + win_size / 2)]
    img1 = np.expand_dims(img1, axis=-1)

    img2 = image.imag
    img2 = img2[int(img2.shape[0] / 2 - win_size / 2): int(img2.shape[0] / 2 + win_size / 2),
           int(img2.shape[1] / 2 - win_size / 2): int(img2.shape[1] / 2 + win_size / 2)]
    img2 = np.expand_dims(img2, axis=-1)

    x = np.concatenate([img1, img2], axis=-1)
    return x

class FacialExpressionDatasets(Dataset):
    def __init__(self, pkl_file, setting='train', win_size = 32, num_fold = 10, k=0, transform=None):
        super(FacialExpressionDatasets, self).__init__()
        self.pkl_file = pkl_file
        self.transform = transform
        self.win_size = win_size
        self.data = {'images': [], 'labels' : []}

        if not os.path.exists(self.pkl_file):
            print('file {0} not exists'.format(self.pkl_file))
            exit()
        with open(self.pkl_file, 'rb') as f:
            data = pickle.load(f)

        num_samples = data['labels'].shape[0]
        index1 = round(num_samples / num_fold) * k
        index2 = round(num_samples / num_fold) * (k+1)
        if setting == 'train':
            if k != 1:
                self.data['images']  = np.concatenate([ data['images'][:index1], data['images'][index2:] ])
                self.data['labels'] = np.concatenate([data['labels'][:index1], data['labels'][index2:]])
            else:
                self.data['images'] = data['images'][index2:]
                self.data['labels'] = data['labels'][index2:]
        if setting == 'test':
            self.data['images'] = data['images'][index1:index2]
            self.data['labels'] = data['labels'][index1:index2]



    def __len__(self):
        return self.data['labels'].shape[0]

    def __getitem__(self, index):
        image = self.data['images'][index]
        label = self.data['labels'][index]

        if self.transform:
            image = self.transform(image)

        image = FeatureSelection(FourierImage(image), self.win_size)

        sample = {'image': image, 'label': label}

        return sample





###   testing dataset class  ####

# face_dataset = FacialExpressionDatasets('new_pkl/ckp_6.pkl', setting='train', num_fold = 10, k=1)
# face_dataset_test = FacialExpressionDatasets('new_pkl/ckp_6.pkl', setting='test', num_fold = 10, k=1)
#
# print(len(face_dataset), len(face_dataset_test))
# for i in range(len(face_dataset)):
#     sample = face_dataset[i]
#     print(i, sample['image'].shape, sample['label'])
