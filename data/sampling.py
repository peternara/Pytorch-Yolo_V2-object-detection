#-*- conding:utf-8 -*-

import os
import numpy as np
from PIL import Image
from torch.utils import data

class MyDataset(data.Dataset):

    def __init__(self,label_txt):
        self.filelists = open(label_txt).readlines()

    def __getitem__(self, index):
        cond = np.zeros(shape=(9,7,7,1))
        offset = np.zeros(shape=(9,7,7,4))

        for i, strs in enumerate(self.filelists[index].split("*")):
            if i == 0:
                img_path = strs
            else:
                str = strs.split(",")
                name = str[0]
                h_index = int(str[1])
                w_index = int(str[2])
                channel_index = int(str[3])
                offset_x1 = np.float32(str[4])
                offset_y1 = np.float32(str[5])
                offset_x2 = np.float32(str[6])
                offset_y2 = np.float32(str[7])

                cond[channel_index][h_index][w_index][0] = 1
                offset[channel_index][h_index][w_index][0] = offset_x1
                offset[channel_index][h_index][w_index][1] = offset_y1
                offset[channel_index][h_index][w_index][2] = offset_x2
                offset[channel_index][h_index][w_index][3] = offset_y2

        img = np.array(Image.open(img_path), dtype=np.float32)/255.0 - 0.5
        img = np.transpose(img, (2,0,1))

        return img, cond, offset

    def __len__(self):
        return len(self.filelists)

    def get_batch(self,dataload):
        dataiter = iter(dataload)
        return dataiter.next()
