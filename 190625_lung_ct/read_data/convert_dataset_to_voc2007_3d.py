# convert mhd voxel to world coordinate


import csv
import math
import cv2

import SimpleITK as sitk
import numpy as np

import matplotlib.pyplot as plt

from glob import glob
from os import path
import os
import argparse

import xml.dom.minidom as minidom
import random


def load_itk(filename):
    '''
    This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
    '''
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # indexes are z,y,x (notice the ordering)
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # z,y,x  Origin in voxel coordinates
    # voxel 体素
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # spacing of voxels to world coor(mm) z,y,x
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


def image_process(image):
    '''
    将最小的数值（图像边界转换为有效像素的最小值）
    :param image:
    :return: image
    '''
    image_max = np.max(image)
    image_min = np.min(image)
    bool_idx_image_min = (image == image_min)
    image_tmp = image
    image_tmp[bool_idx_image_min] = image_max
    image_second_min = np.min(image_tmp)
    image[bool_idx_image_min] = image_second_min
    return image


def image_process_range(image):
    '''
    -1200~600
    :param image:
    :return: image
    '''
    image_max = -200
    image_min = -1024
    bool_idx_image_min = (image <= image_min)
    image[bool_idx_image_min] = image_min
    bool_idx_image_max = (image >= image_max)
    image[bool_idx_image_max] = image_max
    return image


parser = argparse.ArgumentParser()
parser.add_argument("-i", help="input folder", type=str, default="/home/xuxin/simple_data/tiaochi/dataset_raw/train")
parser.add_argument("-ll", help="label file", type=str, default="/home/xuxin/simple_data/tiaochi/dataset_raw/chestCT_round1_annotation.csv")
parser.add_argument("-o", help="output folder", type=str, default="/home/xuxin/simple_data/tiaochi/dataset_image/train_voc2007_rgb")
parser.add_argument("-t", help="dataset type", type=str, default="voc2007")
args = parser.parse_args()

voc_root = args.o
JPEGImages = os.path.join(voc_root, "VOCdevkit2007", "VOC2007", "JPEGImages")
Annatations = os.path.join(voc_root, "VOCdevkit2007", "VOC2007", "Annotations")
Main = os.path.join(voc_root, "VOCdevkit2007", "VOC2007", "ImageSets", "Main")

for t_path in [JPEGImages, Annatations, Main]:
    if not os.path.exists(t_path):
        os.makedirs(t_path)


def main():
    data_path = args.i
    annotation_file = args.ll
    save_dir = args.o
    # jj 结节
    # st 索条
    # dmyh_gh 动脉硬化或钙化
    # lbj 淋巴结钙化
    labels = {1: 'jj', 5: 'st', 31: 'dmyh_gh', 32: 'lbj'}

    with open(annotation_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        anno_rows = [row for row in reader]
    file_list = glob(data_path + "/*.mhd")
    for file_name in file_list:
        ct_scan, origin, spacing = load_itk(file_name)

        file_id = path.basename(file_name).split('.')[0]

        # file_anno = []  # label, Z_slice(0~n), x,y(left_up_point),width,heigth

        def file_anno():
            for row in anno_rows:
                if str(file_id) == row[0]:
                    voxel_center = np.array([float(row[3]), float(row[2]), float(row[1])])  # z,y,x
                    world_center = (voxel_center - origin) / spacing  # z,y,x
                    diameterX = int(float(row[4]) / spacing[2])
                    diameterY = int(float(row[5]) / spacing[1])
                    diameterZ = int(math.ceil(float(row[6]) / spacing[0]))
                    label = int(row[7])

                    left_up_point_x = int(world_center[2] - diameterX / 2)
                    left_up_point_y = int(world_center[1] - diameterY / 2)
                    point_z = world_center[0]

                    start = math.ceil(point_z - diameterZ / 2)
                    end = math.floor(point_z + diameterZ / 2)

                    while start <= end:
                        yield ([label, start, left_up_point_x, left_up_point_y, diameterX, diameterY])
                        start += 1

        ct_scan = image_process_range(ct_scan)
        show_ids = {}

        for anno in iter(file_anno()):
            height, width = ct_scan[anno[1]].shape[:2]
            pt1 = (anno[2] if anno[2] > 0 else 0,
                   anno[3] if anno[3] > 0 else 0)
            pt2 = (anno[2] + anno[4] if anno[2] + anno[4] < width else width - 1,
                   anno[3] + anno[5] if anno[3] + anno[5] < height else height - 1)
            if pt1[0] < 0 or pt1[1] < 0 or pt2[0] > width - 1 or pt2[1] > height - 1:
                continue

            if anno[1] in show_ids.keys():
                show_ids[anno[1]].append([pt1, pt2, anno[0]])
            else:
                show_ids[anno[1]] = [[pt1, pt2, anno[0]]]

        for id, pts in show_ids.items():
            image_count = len(os.listdir(JPEGImages))
            image_name = os.path.join(JPEGImages, "%.6d.jpg" % (image_count + 1))
            for pt in pts:
                if pt[0][0] < 0 or pt[0][1] < 0 or pt[1][0] > 512 - 1 or pt[1][1] > 512 - 1:
                    continue
                add_xml_doc(os.path.join(Annatations, "%.6d.xml" % (image_count + 1)), ct_scan[id].shape, pt[0], pt[1],
                            labels[pt[2]])
                plt.imsave(image_name, ct_scan[id], cmap=plt.cm.gray)
                im = cv2.imread(image_name, 0)
                #im = cv2.equalizeHist(im)
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(image_name, im)
                print("save", image_name)

            # cv2.putText(ct_scan[anno[1]], labels[anno[0]], pt1, cv2.FONT_HERSHEY_COMPLEX, 0.3, (127, 0, 0))
            # cv2.rectangle(ct_scan[anno[1]], pt1, pt2, (255, 0, 0), 1)
            #
            # plt.imshow(ct_scan[anno[1]], cmap='gray')
            # plt.show()
    splite_data()


def add_xml_doc(path, size, pt1, pt2, label, image_type="jpg"):  # size-width,height,depth
    if not os.path.exists(path):
        dom = minidom.getDOMImplementation().createDocument(None, 'annotation', None)
        root = dom.documentElement

        # add folder
        element = dom.createElement('folder')
        element.appendChild(dom.createTextNode('VOC2007'))
        root.appendChild(element)

        # add filename
        element = dom.createElement('filename')
        file_name = os.path.basename(path).replace("xml", image_type)
        element.appendChild(dom.createTextNode(file_name))
        root.appendChild(element)

        # add source
        element = dom.createElement('source')
        child = dom.createElement("database")
        child.appendChild(dom.createTextNode("The VOC2007 Database"))
        element.appendChild(child)
        child = dom.createElement("annotation")
        child.appendChild(dom.createTextNode("PASCAL VOC2007"))
        element.appendChild(child)
        child = dom.createElement("image")
        child.appendChild(dom.createTextNode("flickr"))
        element.appendChild(child)
        child = dom.createElement("flickrid")
        child.appendChild(dom.createTextNode("326445091"))
        element.appendChild(child)
        root.appendChild(element)

        element = dom.createElement('owner')
        child = dom.createElement("flickrid")
        child.appendChild(dom.createTextNode("TIANCHI"))
        element.appendChild(child)
        child = dom.createElement("name")
        child.appendChild(dom.createTextNode("?"))
        element.appendChild(child)
        root.appendChild(element)

        element = dom.createElement('size')
        child = dom.createElement("width")
        child.appendChild(dom.createTextNode(str(size[1])))
        element.appendChild(child)
        child = dom.createElement("height")
        child.appendChild(dom.createTextNode(str(size[0])))
        element.appendChild(child)
        child = dom.createElement("depth")
        child.appendChild(dom.createTextNode(str(3)))
        element.appendChild(child)
        root.appendChild(element)

        element = dom.createElement('segmented')
        element.appendChild(dom.createTextNode("0"))
        root.appendChild(element)

        element = dom.createElement('object')
        child = dom.createElement("name")
        child.appendChild(dom.createTextNode(label))
        element.appendChild(child)
        child = dom.createElement("pose")
        child.appendChild(dom.createTextNode("Unspecified"))
        element.appendChild(child)
        child = dom.createElement("truncated")
        child.appendChild(dom.createTextNode("0"))
        element.appendChild(child)
        child = dom.createElement("difficult")
        child.appendChild(dom.createTextNode("0"))
        element.appendChild(child)

        child = dom.createElement("bndbox")
        grand_child = dom.createElement("xmin")
        grand_child.appendChild(dom.createTextNode(str(pt1[0])))
        child.appendChild(grand_child)
        grand_child = dom.createElement("ymin")
        grand_child.appendChild(dom.createTextNode(str(pt1[1])))
        child.appendChild(grand_child)
        grand_child = dom.createElement("xmax")
        grand_child.appendChild(dom.createTextNode(str(pt2[0])))
        child.appendChild(grand_child)
        grand_child = dom.createElement("ymax")
        grand_child.appendChild(dom.createTextNode(str(pt2[1])))
        child.appendChild(grand_child)
        element.appendChild(child)
        root.appendChild(element)
    else:
        dom = minidom.parse(path)
        root = dom.documentElement
        names = root.getElementsByTagName('annotation')

        element = dom.createElement('object')
        child = dom.createElement("name")
        child.appendChild(dom.createTextNode(label))
        element.appendChild(child)
        child = dom.createElement("pose")
        child.appendChild(dom.createTextNode("Unspecified"))
        element.appendChild(child)
        child = dom.createElement("truncated")
        child.appendChild(dom.createTextNode("0"))
        element.appendChild(child)
        child = dom.createElement("difficult")
        child.appendChild(dom.createTextNode("0"))
        element.appendChild(child)

        child = dom.createElement("bndbox")
        grand_child = dom.createElement("xmin")
        grand_child.appendChild(dom.createTextNode(str(pt1[0])))
        child.appendChild(grand_child)
        grand_child = dom.createElement("ymin")
        grand_child.appendChild(dom.createTextNode(str(pt1[1])))
        child.appendChild(grand_child)
        grand_child = dom.createElement("xmax")
        grand_child.appendChild(dom.createTextNode(str(pt2[0])))
        child.appendChild(grand_child)
        grand_child = dom.createElement("ymax")
        grand_child.appendChild(dom.createTextNode(str(pt2[1])))
        child.appendChild(grand_child)
        element.appendChild(child)
        root.appendChild(element)

    with open(path, 'w+', encoding='utf-8') as f:
        dom.writexml(f, addindent='\t', newl='\n', encoding='utf-8')


def splite_data(rate=(0.7, 0.2, 0.1)):
    image_list = os.listdir(JPEGImages)
    training_list = random.sample(image_list, int(len(image_list) * rate[0]))
    image_set = set(image_list)
    training_set = set(training_list)
    remain_set = image_set - training_set
    val_list = random.sample(list(remain_set), int(len(image_list) * rate[1]))
    val_set = set(val_list)
    test_list = list(remain_set - val_set)

    with open(os.path.join(Main, "test.txt"), "w+") as f:
        for item in test_list:
            f.write(item.split(".")[0] + "\n")
    with open(os.path.join(Main, "val.txt"), "w+") as f:
        for item in val_list:
            f.write(item.split(".")[0] + "\n")
    with open(os.path.join(Main, "train.txt"), "w+") as f:
        for item in training_list:
            f.write(item.split(".")[0] + "\n")
    with open(os.path.join(Main, "trainval.txt"), "w+") as f:
        for item in (training_list + val_list):
            f.write(item.split(".")[0] + "\n")


if __name__ == '__main__':
    main()
