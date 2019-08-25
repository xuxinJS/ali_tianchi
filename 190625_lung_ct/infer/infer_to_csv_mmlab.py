from mmdet.apis import init_detector, inference_detector, show_result
import os
import argparse
import sys
import torch
import cv2
import numpy as np
import time
import csv
import SimpleITK as sitk
from glob import glob

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

def parse_args():
    parser = argparse.ArgumentParser(description='in and out imgs')
    parser.add_argument('--config', dest='config', help='config_file', default=None, type=str)
    parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint_file', default=None, type=str)
    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config_file = args.config
    checkpoint_file = args.checkpoint
    model = init_detector(config_file, checkpoint_file)
    classes = ('jj', 'st', 'dmyh_gh', 'lbj')
    labels_to_id = {0: 1, 1: 5, 2: 31, 3: 32}
    directory = '/home/xuxin/simple_data/tiaochi/dataset_image/trainA'
    mhd_directory = '/home/xuxin/simple_data/tiaochi/dataset_raw/train'
    result_name = 'faster_rcnn_x101_64x4d_fpn_1x_epoch_20_nms20_0.299.csv'
    threshold = 0.5
    show_flag = False
    flag_save_csv = True

    if flag_save_csv:
        csvfile = open(result_name, "w")
        writer = csv.writer(csvfile)
        writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'class', 'probability'])
        coor_convert = {}
        file_list = glob(mhd_directory + "/*.mhd")
        for file_name in file_list:
            file_id = os.path.basename(file_name).split('.')[0]
            _, origin, spacing = load_itk(file_name)  # z,y,x
            coor_convert[file_id] = [origin, spacing]
        # print(coor_convert)

    for name in os.listdir(directory):
        # seriesuid,coordX,coordY,coordZ,class,probability
        print(name)
        if flag_save_csv:
            mhd_id = name.split('_')[0]
            z_slice = name.split('.')[0].split('_')[-1]
            origin = (coor_convert[mhd_id])[0]
            spacing = (coor_convert[mhd_id])[1]

        image_name = os.path.join(directory, name)
        if show_flag:
            image = cv2.imread(image_name)
        result = inference_detector(model, image_name)
        for index, bboxes in enumerate(result):
            if len(bboxes):
                for box in bboxes:
                    if box[-1] >= threshold:
                        x1 = box[0]
                        y1 = box[1]
                        x2 = box[2]
                        y2 = box[3]
                        score = box[-1]

                        if flag_save_csv:
                            box_center_x = (box[0] + box[2]) / 2
                            box_center_y = (box[1] + box[3]) / 2
                            voxel_x = box_center_x * spacing[2] + origin[2]
                            voxel_y = box_center_y * spacing[1] + origin[1]
                            voxel_z = int(z_slice) * spacing[0] + origin[0]
                            voxel_label = labels_to_id[index]
                            # print(box) # x1,y1,x2,y2
                            # seriesuid,coordX,coordY,coordZ,class,probability
                            print(mhd_id, voxel_x, voxel_y, voxel_z, voxel_label, score)
                            writer.writerow([mhd_id, voxel_x, voxel_y, voxel_z, voxel_label, score])

                        if show_flag:
                            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 1)
                            cv2.putText(image, classes[index], (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0))

        if show_flag:
            cv2.imshow('show', image)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
    # savename = savedir + 'pic.png'
    # show_result(img, result, classes, out_file=savename)
    if flag_save_csv:
        csvfile.close()

if __name__ == '__main__':
    main()
