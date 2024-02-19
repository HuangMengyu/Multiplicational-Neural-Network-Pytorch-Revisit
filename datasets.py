# -*- coding: utf-8 -*-
"""
inherit Dataset class for my own datasets

@author: Mengyu Huang
"""


from torch.utils.data import Dataset
import os
import pickle
import numpy as np

class FacialExpressionDatasets(Dataset):
    def __init__(self, pkl_file, setting='train', num_fold = 10, k=0, transform=None):
        super(FacialExpressionDatasets, self).__init__()
        self.pkl_file = pkl_file
        self.transform = transform
        self.data = {'imgs': [], 'labels' : []}

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
                self.data['imgs']  = np.concatenate([ data['imgs'][:index1], data['imgs'][index2:] ])
                self.data['labels'] = np.concatenate([data['labels'][:index1], data['labels'][index2:]])
            else:
                self.data['imgs'] = data['imgs'][index2:]
                self.data['labels'] = data['labels'][index2:]
        if setting == 'test':
            self.data['imgs'] = data['imgs'][index1:index2]
            self.data['labels'] = data['labels'][index1:index2]


    def __len__(self):
        return self.data['labels'].shape[0]

    def __getitem__(self, index):
        image = self.data['imgs'][index]
        label = self.data['labels'][index]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample



###   testing dataset class  ####

# face_dataset = FacialExpressionDatasets('new_pkl/ckp_6.pkl', setting='train', num_fold = 10, k=1)
# face_dataset_test = FacialExpressionDatasets('new_pkl/ckp_6.pkl', setting='test', num_fold = 10, k=1)
#
# print(len(face_dataset), len(face_dataset_test))
# for i in range(len(face_dataset)):
#     sample = face_dataset[i]
#     print(i, sample['image'].shape, sample['label'])
