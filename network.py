# -*- coding: utf-8 -*-
"""
network structure of MNN

@author: Mengyu Huang

"""

import torch
import torch.nn as nn
import numpy as np


class mult_layer(nn.Module):
    
    def __init__(self, height, width, channel):
        super(mult_layer,self).__init__()
        self.channel = channel
        self.height = height
        self.width = width
        # weights initialization
        self.weight = nn.Parameter(torch.randn(self.height, self.width, self.channel), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)
        #print('weight and bias', self.weight.shape, self.bias.shape)

    def forward(self, x): #height, width, channels
        #output computation
        x_channels = x.shape[-1]
        output = torch.empty(x.shape[0], self.height, self.width, self.channel*x_channels)
        for k in range(x.shape[0]):
            for i in range(x_channels):
                for j in range(self.channel):
                    #print(x[k,:,:,i].shape, self.weight[ :,:,j].shape)
                    output[k, :, :, self.channel * i + j] = torch.mul(x[k,:,:,i] , self.weight[:,:,j]) + self.bias
        
        return output
            
            
        
class MNN_branch(nn.Module):
    def __init__(self, height, width, input_channel):
        super(MNN_branch, self).__init__()
        self.mul_layer_first = mult_layer(height, width, input_channel)
        self.mul_layer_mid = mult_layer(height, width, 1)
        self.conv_layer = nn.Conv2d(input_channel, input_channel, 5)
        self.max_pool = nn.functional.max_pool2d
        
        
    def forward(self, x):
        output = self.mul_layer_first(x)
        #print('out1', output.shape)
        for _ in range(0,4):
            output = self.mul_layer_mid(output)
            #print('out', output.shape)
        output = output.permute((0,3,1,2))
        #print('permuted', output.shape)
        output = self.conv_layer(output)
        output = nn.functional.relu(output)
        output = self.max_pool(output, kernel_size = 2)
        return output
        
        
        
class MNN(nn.Module):
    def __init__(self, height, width, input_channel, classes):
        super(MNN, self).__init__()
        self.branch = MNN_branch(height, width, input_channel)

        self.linear2 = nn.Linear(2048, 512)
        self.linear3 = nn.Linear(512, classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, fusion='add'): #x: height, width, 2; fusion: 'add','concat'
        #print('x', x.shape)
        x1 = x[:,:,:,0].unsqueeze(-1) #real component
        x2 = x[:,:,:,1].unsqueeze(-1)
        #print('x1', x1.shape, 'x2', x2.shape)
        x1 = self.branch(x1)
        x2 = self.branch(x2)
        
        #assert x1.size().equal(x2.size()), 'the sizes of two branches are not equal'
        
        if fusion == 'add':
            output = x1.add(x2)
        if fusion == 'concat':
            output = x1.concat(x2)

        num_feature = output.shape[1] * output.shape[2] * output.shape[3]
        output = output.view(-1, num_feature)
        #print('output.view', output.shape)
        output = nn.Linear(num_feature, 2048)(output)
        output = nn.Tanh()(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = nn.Tanh()(output)
        output = self.dropout(output)
        output = self.linear3(output)
       # output = nn.Softmax(dim=1)(output)
        
        return output
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
