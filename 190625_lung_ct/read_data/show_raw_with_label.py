# convert mhd voxel to world coordinate
import csv
import math
import cv2

import matplotlib.pyplot as plt

from glob import glob
from os import path

import sys

sys.path.append('../lib')
from preprocess import *


def main():
    # data_path = '/home/dls1/simple_data/tiaochi/test/mhd'
    data_path = '/T3/data/train_data/public/tianchi/190625_CT/dataset/split/test'
    annotation_file = '/T3/data/train_data/public/tianchi/190625_CT/dataset/chestCT_round1_annotation.csv'
    save_dir = '/home/dls1/simple_data/tiaochi/test/images'
    save_empty_dir = "/home/dls1/simple_data/tiaochi/test/empty"
    choose_label = [1, 5]
    labels, labels_to_id = label_select('preliminary')
    with open(annotation_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        anno_rows = [row for row in reader]
    file_list = glob(data_path + "/*.mhd")
    for file_name in file_list:
        ct_scan, origin, spacing = load_itk(file_name)

        file_id = path.basename(file_name).split('.')[0]
        file_anno = []  # label, Z_slice(0~n), x,y(left_up_point),width,heigth
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
                    file_anno.append([label, start, left_up_point_x, left_up_point_y, diameterX, diameterY])
                    start += 1

        ct_scan = image_process_range(ct_scan, -750, -250)
        show_ids = []
        for anno in file_anno:
            # if anno[1] not in show_ids:
            #     show_ids.append(anno[1])
            if anno[0] in choose_label:
                save_file_name = path.join(save_dir, "%s_%d.jpg" % (file_id, anno[1]))
                save_empty_file_name = path.join(save_empty_dir, "%s_%d_empty.jpg" % (file_id, anno[1]))
                # plt.imsave(save_file_name, ct_scan[anno[1]], cmap=plt.cm.gray)
                plt.imsave(save_empty_file_name, ct_scan[anno[1]], cmap=plt.cm.gray)
                image = cv2.imread(save_empty_file_name, 0)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                pt1 = (anno[2], anno[3])
                pt2 = (anno[2] + anno[4], anno[3] + anno[5])
                cv2.putText(image, labels[anno[0]], (anno[2], anno[3]-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
                cv2.rectangle(image, pt1, pt2, (255, 0, 0), 2)
                cv2.imwrite(save_file_name, image)
        # for id in show_ids:
        #     plt.imshow(ct_scan[id], cmap='gray')
        #     plt.show()
        # save_file_name = path.join(save_dir, "%s_%d.jpg" % (file_id, id))


if __name__ == '__main__':
    main()
