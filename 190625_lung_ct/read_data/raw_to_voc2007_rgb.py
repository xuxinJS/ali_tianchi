# convert mhd and raw to voc2007 format dataset
import math
import cv2
import argparse

import matplotlib.pyplot as plt

from glob import glob
import sys

sys.path.append('../lib')
from preprocess import *
from csvTools import readCSV


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input folder", type=str,
                        default="/home/dls1/simple_data/tiaochi/test/mhd")
    parser.add_argument("-l", "--label", help="label file", type=str,
                        default="/home/dls1/simple_data/tiaochi/test/train.csv")
    parser.add_argument("-o", "--output_folder", help="output folder", type=str,
                        default="/home/dls1/simple_data/tiaochi/test/images")
    parser.add_argument("-lc", "--label_choose", help="choose preliminary and rematch label mapping",
                        type=str, default="preliminary", choices=["preliminary", 'rematch'])
    parser.add_argument("-imin", "--image_min", help="image max value", type=int, default=-1000)
    parser.add_argument("-imax", "--image_max", help="image max value", type=int, default=400)
    parser.add_argument("-im", "--image_mode", help="image mode", type=str, default='rgb', choices=['gray', 'rgb'])
    parser.add_argument("-sl", "--save_label", help="save specified label", type=int)
    args = parser.parse_args()
    return args


def main():
    save_label = [31, 32, 33]
    args = parse_args()
    flag_save_label = args.save_label
    if flag_save_label == 0:
        save_label = [1, 2, 3, 5]
    else:
        save_label = [31, 32, 33]
    data_path = args.input
    annotation_file = args.label
    voc_root = args.output_folder
    label_choose = args.label_choose
    image_min = args.image_min
    image_max = args.image_max
    image_mode = args.image_mode
    JPEGImages = os.path.join(voc_root, "VOCdevkit2007", "VOC2007", "JPEGImages")
    Annatations = os.path.join(voc_root, "VOCdevkit2007", "VOC2007", "Annotations")
    Main = os.path.join(voc_root, "VOCdevkit2007", "VOC2007", "ImageSets", "Main")
    labels, _ = label_select(label_choose)

    for t_path in [JPEGImages, Annatations, Main]:
        if not os.path.exists(t_path):
            os.makedirs(t_path)
    anno_rows = readCSV(annotation_file)
    raw_list = glob(data_path + "/*.mhd")
    for file_name in raw_list:
        ct_scan, origin, spacing = load_itk(file_name)
        file_id = os.path.basename(file_name).split('.')[0]
        ct_height, ct_width = ct_scan.shape[1:]
        file_annos = []  # label, Z_slice(0~n), xmin,ymin,xmax,ymax
        #row :seriesuid	coordX	coordY	coordZ	diameterX	diameterY	diameterZ	label
        for row in anno_rows:
            if str(file_id) == row[0] and int(row[7]) in save_label:
                voxel_center = np.array([float(row[3]), float(row[2]), float(row[1])])  # z,y,x
                world_center = (voxel_center - origin) / spacing  # z,y,x
                diameterX = float(row[4]) / spacing[2]
                diameterY = float(row[5]) / spacing[1]
                diameterZ = math.ceil(float(row[6]) / spacing[0])
                label = int(row[7])

                left_up_point_x = int(world_center[2] - diameterX / 2)
                left_up_point_y = int(world_center[1] - diameterY / 2)
                right_down_point_x = int(left_up_point_x + diameterX)
                right_down_point_y = int(left_up_point_y + diameterY)
                xmin = left_up_point_x if left_up_point_x >= 0 else 0
                ymin = left_up_point_y if left_up_point_y >= 0 else 0
                xmax = right_down_point_x if right_down_point_x < ct_width else ct_width - 1
                ymax = right_down_point_y if right_down_point_y < ct_height else ct_height - 1

                point_z = world_center[0]

                start = math.ceil(point_z - diameterZ / 2)
                end = math.floor(point_z + diameterZ / 2)

                while start <= end:
                    file_annos.append([label, start, xmin, ymin, xmax, ymax])
                    start += 1

        ct_scan = image_process_range(ct_scan, image_min, image_max)
        all_slices_list = range(ct_scan.shape[0])
        all_slices_set = set(all_slices_list)
        valid_slices_set = set()
        for i in file_annos:
            valid_slices_set.add(i[1])
        invalid_slices_set = all_slices_set - valid_slices_set
        invalid_slices_list = list(invalid_slices_set)
        valid_slices_list = list(valid_slices_set)
        if len(invalid_slices_list) > len(valid_slices_list):
            invalid_slices_list = invalid_slices_list[:len(valid_slices_list)]

        for anno in file_annos:
            save_name = file_id + '_' + str(anno[1])
            image_name = os.path.join(JPEGImages, "%s.jpg" % save_name)
            annotation_name = os.path.join(Annatations, "%s.xml" % save_name)
            if not os.path.exists(image_name):
                plt.imsave(image_name, ct_scan[anno[1]], cmap=plt.cm.gray)
                depth = 1
                if image_mode == 'rgb':
                    im = cv2.imread(image_name, 0)
                    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(image_name, im)
                    depth = 3
            add_xml_doc(annotation_name, ct_scan[anno[1]].shape, (anno[2], anno[3]),
                        (anno[4], anno[5]), labels[anno[0]], depth)
            print("save valid", image_name)

        #save no label image
        for z_slice in invalid_slices_list:
            save_name = file_id + '_' + str(z_slice)
            image_name = os.path.join(JPEGImages, "%s.jpg" % save_name)
            annotation_name = os.path.join(Annatations, "%s.xml" % save_name)
            if not os.path.exists(image_name):
                plt.imsave(image_name, ct_scan[z_slice], cmap=plt.cm.gray)
                depth = 1
                if image_mode == 'rgb':
                    im = cv2.imread(image_name, 0)
                    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(image_name, im)
                    depth = 3
            add_xml_doc(annotation_name, ct_scan[anno[1]].shape, None,
                        None, None, depth)
            print("save invalid", image_name)

    splite_data(JPEGImages, Main)



if __name__ == '__main__':
    main()
