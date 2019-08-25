# -*- coding:utf-8 -*-
# !/usr/bin/env python
import json
import glob
import os
import sys

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class PascalVOC2coco(object):
    def __init__(self, xml=[], save_json_path='./new.json'):
        '''
        :param xml: 所有Pascal VOC的xml文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.xml = xml
        self.images = []
        self.categories = []
        self.label = []
        self.real_classes = ['jj', 'st', 'dmyh_gh', 'lbj']
        self.annID = 0
        self.annotations = []
        self.save_json_path = save_json_path

    def data_transfer(self):
        for num, xml_file in enumerate(self.xml):
            # 进度输出
            sys.stdout.write('\r>> Converting image %d/%d' % (
                num + 1, len(self.xml)))
            sys.stdout.flush()

            tree = ET.parse(xml_file)
            root = tree.getroot()
            img_name = root.find('filename').text
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            self.images.append(self.image(height, width, num, img_name))

            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in self.real_classes:
                    continue
                cls_id = self.real_classes.index(cls)
                self.categories.append(self.categorie(cls, cls_id))
                self.label.append(cls_id)

                xml_box = obj.find('bndbox')
                # COCO 对应格式[x,y,w,h]
                x_min = int(xml_box.find('xmin').text)
                y_min = int(xml_box.find('ymin').text)
                x_max = int(xml_box.find('xmax').text)
                y_max = int(xml_box.find('ymax').text)
                w = x_max - x_min
                h = y_max - y_min
                area = w * h
                bbox = [x_min, y_min, w, h]
                self.annID += 1
                self.annotations.append(self.annotation(num, bbox, cls_id, self.annID, area))

        sys.stdout.write('\n')
        sys.stdout.flush()

    def image(self, h, w, num, name):
        _image = {}
        _image['id'] = num + 1
        _image['width'] = w
        _image['height'] = h
        _image['file_name'] = name
        return _image

    def categorie(self, label, label_id):
        _categorie = {}
        _categorie['supercategory'] = label
        _categorie['id'] = label_id
        _categorie['name'] = label
        return _categorie

    def annotation(self, num, bbox, category_id, annID, area):
        _annotation = {}
        _annotation['iscrowd'] = 0
        _annotation['image_id'] = num + 1
        _annotation['bbox'] = bbox
        _annotation['category_id'] = category_id
        _annotation['id'] = annID
        _annotation['area'] = area
        return _annotation

    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        data_coco = self.data2coco()
        # 保存json文件
        json.dump(data_coco, open(self.save_json_path, 'w'), indent=4)  # indent=4 更加美观显示


if __name__ == '__main__':
    xml_file = glob.glob('/home/xuxin/Desktop/train/label/*.xml')
    convert = PascalVOC2coco(xml_file, 'test.json')
    convert.save_json()