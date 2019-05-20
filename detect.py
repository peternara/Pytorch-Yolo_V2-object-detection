#-*- conding:utf-8 -*-

import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image,ImageDraw
from model.net import Yolo_V2
from tool.utils import iou

class Detect:

    def __init__(self,img):
        self.net = Yolo_V2()
        self.net_param = r"./param/yolo_v2.pkl"
        self.img = Variable(torch.Tensor(img))

        # # cuda
        # if torch.cuda.is_available():
        #     self.net = self.net.cuda()
        #     self.img = self.img.cuda()

        self.net.load_state_dict(torch.load(self.net_param))
        self.net.eval()

    def detect(self):
        out_cond,out_offset = self.net(self.img)
        out_cond = out_cond.view(-1, 9, 7, 7, 1)
        out_offset = out_offset.view(-1, 9, 4, 7, 7)
        out_offset = out_offset.permute(0,1,3,4,2)

        index_cond = torch.gt(out_cond,0.9)
        index_nonzero = torch.nonzero(index_cond)

        index_nonzero_np = index_nonzero.cpu().data.numpy()
        out_offset_np = out_offset.cpu().data.numpy()

        boxes = []
        for i in range(len(index_nonzero_np)):
            c,h,w = index_nonzero_np[i][1],index_nonzero_np[i][2],index_nonzero_np[i][3]
            [x1,y1,x2,y2] = self.reverse_boxes(c,h,w,out_offset_np[0][c][h][w][0],out_offset_np[0][c][h][w][1],out_offset_np[0][c][h][w][2],out_offset_np[0][c][h][w][3])
            boxes.append([x1,y1,x2,y2])
        return boxes

    def reverse_boxes(self,c,h,w,offset_x1,offset_y1,offset_x2,offset_y2):

        cx,cy = w*32+16,h*32+16
        offset = [offset_x1,offset_y1,offset_x2,offset_y2]
        if c == 0:
            box = [cx - 16, cy -16, cx + 16, cy + 16]
            x1, y1, x2, y2 = self.get_box_value(box, offset)
        if c == 1:
            box = [cx - 24, cy - 24, cx + 24, cy + 24]
            x1, y1, x2, y2 = self.get_box_value(box, offset)
        if c == 2:
            box = [cx - 32, cy - 32, cx + 32, cy + 32]
            x1, y1, x2, y2 = self.get_box_value(box, offset)
        if c == 3:
            box = [cx - 24, cy - 8, cx + 24, cy + 8]
            x1, y1, x2, y2 = self.get_box_value(box, offset)
        if c == 4:
            box = [cx - 48, cy - 16, cx + 48, cy + 16]
            x1, y1, x2, y2 = self.get_box_value(box, offset)
        if c == 5:
            box = [cx - 96, cy - 32, cx + 96, cy + 32]
            x1, y1, x2, y2 = self.get_box_value(box, offset)
        if c == 6:
            box = [cx - 8, cy - 24, cx + 8, cy + 24]
            x1, y1, x2, y2 = self.get_box_value(box, offset)
        if c == 7:
            box = [cx - 16, cy - 48, cx + 16, cy + 48]
            x1, y1, x2, y2 = self.get_box_value(box, offset)
        if c == 8:
            box = [cx - 32, cy - 96, cx + 32, cy + 96]
            x1, y1, x2, y2 = self.get_box_value(box, offset)

        return [x1,y1,x2,y2]

    def get_box_value(self,box,offset):
        x1 = int(offset[0] * (box[2] - box[0]) + box[0])
        y1 = int(offset[1] * (box[3] - box[1]) + box[1])
        x2 = int(offset[2] * (box[2] - box[0]) + box[2])
        y2 = int(offset[3] * (box[3] - box[1]) + box[3])
        return x1,y1,x2,y2


if __name__ == '__main__':
    img_path = r"./imgs/2007_000027.jpg"
    img = Image.open(img_path)
    img_in = np.array(img, dtype=np.float32)/255.0 - 0.5
    img_in = np.transpose(img_in, (2, 0, 1))
    img_in = np.array([img_in])

    detect = Detect(img_in)
    boxes = detect.detect()
    imDrwa = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        print(boxes[i])

        x1 = int(boxes[i][0])
        y1 = int(boxes[i][1])
        x2 = int(boxes[i][2])
        y2 = int(boxes[i][3])
        imDrwa.rectangle((x1, y1, x2, y2), outline="red")
    img.save("./yolo.jpg")
    img.show()





