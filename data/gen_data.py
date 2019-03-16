#-*- conding:utf-8 -*-
import numpy as np
import os
from PIL import Image,ImageDraw
from xml.etree import ElementTree as ET
from tool import utils
import traceback

def check_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    fp = open(file_path,"w")
    fp.close()

def check_dir(path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            print("delete --",os.path.join(path, filename))
            os.remove(os.path.join(path, filename))
        os.removedirs(path)
    os.makedirs(path)

class gen_Data:
    def __init__(self,xml_path,img_path,save_img_224_path,save_txt):
        self.xml_path = xml_path
        self.img_path = img_path
        self.save_img_224_path = save_img_224_path
        self.save_txt = save_txt
        self.xml_list = []
        for xml_name in os.listdir(self.xml_path):
            self.xml_list.append(os.path.join(self.xml_path,xml_name))

        self.read_xml()

    def read_xml(self):
        for i in range(len(self.xml_list)):
            print(i,"-----------------------")
            xml_file = self.xml_list[i]
            root = ET.parse(xml_file).getroot()
            img_name = root.find("filename").text
            img_path_filename = os.path.join(self.img_path,img_name)
            img_width = int(root.find("size/width").text)
            img_height = int(root.find("size/height").text)

            img = Image.open(img_path_filename)
            re_Img = img.resize((224, 224), Image.ANTIALIAS)
            list_type = []
            list_heigth = []
            list_width = []
            list_channel = []
            list_offset = []
            try:
                save_file = open(self.save_txt, "a+")
                try:
                    for obj in root.iter("object"):
                        box_type = obj.find("name").text
                        list_type.append(box_type)
                        x1 = int(int(obj.find("bndbox/xmin").text) * 224 / img_width)
                        y1 = int(int(obj.find("bndbox/ymin").text) * 224 / img_height)
                        x2 = int(int(obj.find("bndbox/xmax").text) * 224 / img_width)
                        y2 = int(int(obj.find("bndbox/ymax").text) * 224 / img_height)
                        box = np.array([x1,y1,x2,y2])
                        cx = int((x1 + x2)/2)
                        cy = int((y1 + y2)/2)

                        cbox_width_index = cx // 32
                        cbox_height_index = cy // 32

                        list_width.append(cbox_width_index)
                        list_heigth.append(cbox_height_index)

                        box_cx = cbox_width_index * 32 + 16
                        box_cy = cbox_height_index * 32 + 16

                        boxes = []
                        _c_16_x1 = box_cx - 16             
                        _c_16_y1 = box_cy - 16
                        _c_16_x2 = box_cx + 16
                        _c_16_y2 = box_cy + 16

                        _c_24_x1 = box_cx - 24             
                        _c_24_y1 = box_cy - 24
                        _c_24_x2 = box_cx + 24
                        _c_24_y2 = box_cy + 24

                        _c_32_x1 = box_cx - 32             
                        _c_32_y1 = box_cy - 32
                        _c_32_x2 = box_cx + 32
                        _c_32_y2 = box_cy + 32

                        boxes.append([_c_16_x1,_c_16_y1,_c_16_x2,_c_16_y2])
                        boxes.append([_c_24_x1,_c_24_y1,_c_24_x2,_c_24_y2])
                        boxes.append([_c_32_x1,_c_32_y1,_c_32_x2,_c_32_y2])

                        _w_1_x1 = box_cx - 24            
                        _w_1_y1 = box_cx - 8
                        _w_1_x2 = box_cx + 24
                        _w_1_y2 = box_cx + 8

                        _w_2_x1 = box_cx - 48            
                        _w_2_y1 = box_cx - 16
                        _w_2_x2 = box_cx + 48
                        _w_2_y2 = box_cx + 16

                        _w_3_x1 = box_cx - 96            
                        _w_3_y1 = box_cx - 32
                        _w_3_x2 = box_cx + 96
                        _w_3_y2 = box_cx + 32

                        boxes.append([_w_1_x1,_w_1_y1,_w_1_x2,_w_1_y2])
                        boxes.append([_w_2_x1,_w_2_y1,_w_2_x2,_w_2_y2])
                        boxes.append([_w_3_x1,_w_3_y1,_w_3_x2,_w_3_y2])

                        _h_1_x1 = box_cx - 8            
                        _h_1_y1 = box_cx - 24
                        _h_1_x2 = box_cx + 8
                        _h_1_y2 = box_cx + 24

                        _h_2_x1 = box_cx - 16           
                        _h_2_y1 = box_cx - 48
                        _h_2_x2 = box_cx + 16
                        _h_2_y2 = box_cx + 48

                        _h_3_x1 = box_cx - 32           
                        _h_3_y1 = box_cx - 96
                        _h_3_x2 = box_cx + 32
                        _h_3_y2 = box_cx + 96

                        boxes.append([_h_1_x1,_h_1_y1,_h_1_x2,_h_1_y2])
                        boxes.append([_h_2_x1,_h_2_y1,_h_2_x2,_h_2_y2])
                        boxes.append([_h_3_x1,_h_3_y1,_h_3_x2,_h_3_y2])

                        iou = utils.iou(box, np.array(boxes))

                        index_max = np.where(iou == np.max(iou))[0][0]
                        list_channel.append(index_max)

                        max_box = np.array([boxes[index_max][0],boxes[index_max][1],boxes[index_max][2],boxes[index_max][3]])
                        offset_x1, offset_y1, offset_x2, offset_y2 = self.get_offset(box, max_box)
                        list_offset.append([offset_x1, offset_y1, offset_x2, offset_y2])

                    re_Img.save(os.path.join(self.save_img_224_path, img_name))
                    str = ""
                    for num in range(len(list_type)):
                        str += "*{0},{1},{2},{3},{4},{5},{6},{7}".format(list_type[num],list_heigth[num],list_width[num],list_channel[num],
                                                                                  list_offset[num][0],list_offset[num][1],list_offset[num][2],list_offset[num][3])

                    save_file.write("{0}{1}\n".format(os.path.join(self.save_img_224_path, img_name),str))
                    save_file.flush()
                except Exception as e:
                    traceback.print_exc()
            finally:
                save_file.close()


    def get_offset(self,box,max_box):
        offset_x1 = (box[0] - max_box[0]) / (max_box[2] - max_box[0])
        offset_y1 = (box[1] - max_box[1]) / (max_box[3] - max_box[1])
        offset_x2 = (box[2] - max_box[2]) / (max_box[2] - max_box[0])
        offset_y2 = (box[3] - max_box[3]) / (max_box[3] - max_box[1])
        return offset_x1,offset_y1,offset_x2,offset_y2

