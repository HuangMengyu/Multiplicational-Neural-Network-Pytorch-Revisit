# -*- coding: utf-8 -*-
"""
training MNN

@author: Mengyu Huang
"""

import torch
import os
from datasets import FacialExpressionDatasets
import argparse
import torch.optim as optim
import torch.nn as nn
from network import MNN


# parser = argparse.ArgumentParser()
# parser.add_argument('dataset', type=str,help='dataset name', default='ckp_6')
# args = parser.parse_args()
#
# data_dir = ''
# if args.dataset.equals('ckp_6'):
#     data_dir = 'new_pkl/ckp_6.pkl'
# elif args.dataset.equal('oulu_casia'):
#    data_dir = 'new_pkl/oulu_casia.pkl'

data_dir = 'new_pkl/ckp_6.pkl'

# ten-fold validation
num_fold = 10
learning_rate = 0.001
num_epoch = 100

mnn = MNN(height=32, width=32, input_channel=40, classes=6)
params = mnn.parameters()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mnn.parameters(), lr=0.001)

for i in range(num_fold):
    print ('current fold: %d' % (i))
    train_dataset = FacialExpressionDatasets(data_dir, setting='train', k=i)
    test_dataset = FacialExpressionDatasets(data_dir, setting='test', k=i)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=16,
                                              shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=0)

    for epoch in range(num_epoch):
        print('start training epoch : %d' %(epoch))
        running_loss = 0.0
        num_batch = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs = data['image']
            labels = torch.tensor(data['label'], dtype=torch.long)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = mnn(inputs)
            #outputs, _ = outputs.max(dim=1)
            #outputs = torch.squeeze(outputs)
            #print('final outputs', outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            num_batch += 1
            if num_batch % 10 == 0:
                print('epoch: %d, batch: %d, loss: %.3f' %
                      (epoch + 1, num_batch+1, running_loss / num_batch))

        print('epoch: %d, loss: %.3f' %
                    (epoch + 1, running_loss /num_batch ))
    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        print('Begin testing')
        for data in testloader:
            inputs = data['image']
            labels = torch.tensor(data['label'], dtype=torch.long)
            outputs = mnn(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the %d-th fold test images: %d %%' % (i, 100 * correct / total))




