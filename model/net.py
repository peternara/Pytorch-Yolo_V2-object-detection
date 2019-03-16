#-*- conding:utf-8 -*-

import torch
import torch.nn as nn
import math
import numpy as np


class Yolo_V2(nn.Module):
    def __init__(self):
        super(Yolo_V2,self).__init__()
        # n*3*448*448 -> n*1000*7*7
        self.layer0 = nn.Sequential(
            nn.Conv2d(3,32,3,1,padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,1,padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128,64,1,1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64,128,3,1,padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256,128,1,1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(256,512,3,1,padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.Conv2d(512,256,1,1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256,512,3,1,padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.Conv2d(512,256,1,1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256,512,3,1,padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(512,1024,3,1,padding=1),
            nn.BatchNorm2d(1024),
            nn.PReLU(),
            nn.Conv2d(1024,512,1,1),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.Conv2d(512,1024,3,1,padding=1),
            nn.BatchNorm2d(1024),
            nn.PReLU(),
            nn.Conv2d(1024,512,1,1),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.Conv2d(512,1024,3,1,padding=1),
            nn.BatchNorm2d(1024),
            nn.PReLU()
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(1024,1000,1,1),
        )

        self.cond = nn.Sequential(
            nn.Conv2d(1000,9,1,1),
            nn.Sigmoid()
        )
        self.offset = nn.Sequential(
            nn.Conv2d(1000,36,1,1),
        )

        self.weights_init()

    def forward(self,x):
        x = self.layer0(x)
        x = self.layer1(x)
        _cond = self.cond(x)
        _offset = self.offset(x)
        return _cond, _offset

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            n = m.out_features
            m.weight.data.normal_(0, math.sqrt(1. / n))
            m.bias.data.zero_()

    def clip_weight(params):
        torch.nn.utils.clip_grad_value_(params, 0.01)
