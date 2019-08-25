# save mhd and raw data to filename_zslice.jpg

import cv2
import matplotlib.pyplot as plt
import argparse

from glob import glob
from os import path

import sys
sys.path.append('../lib')
from preprocess import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input folder", type=str,
                        default="/home/dls1/simple_data/tiaochi/test/mhd")
    parser.add_argument("-o", "--output_folder", help="output folder", type=str,
                        default="/home/dls1/simple_data/tiaochi/test/images")
    parser.add_argument("-imin", "--image_min", help="image max value", type=int, default=-1000)
    parser.add_argument("-imax", "--image_max", help="image max value", type=int, default=400)
    parser.add_argument("-im", "--image_mode", help="image mode", type=str, default='rgb', choices=['gray', 'rgb'])
    args = parser.parse_args()
    return args

def image_save():
    args = parse_args()
    data_path = args.input
    save_dir = args.output_folder
    if not path.exists(save_dir):
        os.makedirs(save_dir)
    image_mode = args.image_mode
    image_min = args.image_min
    image_max = args.image_max
    file_list = glob(data_path + "/*.mhd")
    for file_name in file_list:
        file_id = path.basename(file_name).split('.')[0]
        ct_scan, origin, spacing = load_itk(file_name)
        ct_scan = image_process_range(ct_scan, image_min, image_max)
        slices = range(ct_scan.shape[0])
        for id in slices:
            #plt.imshow(ct_scan[id], cmap='gray')
            save_file_name = path.join(save_dir, "%s_%d.jpg" % (file_id, id))
            # plt.imsave(save_file_name, ct_scan[id], cmap=plt.cm.gray, format='jpg')
            plt.imsave(save_file_name, ct_scan[id], cmap=plt.cm.gray)
            if image_mode == 'rgb':
                im = cv2.imread(save_file_name, 0)
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(save_file_name, im)
            print("save", save_file_name)

if __name__ == '__main__':
    image_save()
