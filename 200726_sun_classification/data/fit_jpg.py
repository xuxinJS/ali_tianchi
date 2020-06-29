# coding=utf-8

import os
import cv2
import warnings
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
from matplotlib.colors import LogNorm

warnings.filterwarnings(action='ignore')
figure = plt.figure()


def convert_data(input_name, output_name):
    # ---------------read fits file------------------
    fits_file = fits.open(input_name)
    fits_file.verify("fix")  # slove error
    image_data = fits_file[1].data
    fits_file.close()

    # ---------------plot show----------------------
    # plt.subplot(311)
    # plt.imshow(image_data, cmap='gray')
    # plt.subplot(312)
    # plt.imshow(image_data, cmap='gray', norm=LogNorm())
    # plt.subplot(313)
    # NBINS = 1000
    # histogram = plt.hist(image_data.flatten(), NBINS)
    # plt.colorbar()
    # plt.show()

    # ---------------cv2 nomalized show-------------
    imin = np.min(image_data)
    imax = np.max(image_data)
    cv_im = ((image_data - imin) / (imax - imin) * 256).astype(np.uint8)
    # cv2.imshow('cv', cv_im)
    # cv2.waitKey(0)
    cv2.imwrite(output_name, cv_im)


input_folder = '/T3/data/train_data/public/tianchi/20200726_sun_classification/test_input'
output_folder = '/home/dls1/simple_data/classification/test'
flag = 'test'  # train test
cores = 12
pool = multiprocessing.Pool(processes=cores)
if flag == 'train':
    for folder in os.listdir(input_folder):
        folder_name = os.path.join(input_folder, folder)
        for cls in os.listdir(folder_name):
            class_folder = os.path.join(folder_name, cls)
            for file in os.listdir(class_folder):
                print(file)
                input_file_name = os.path.join(class_folder, file)
                file_split = file.split('.')
                target = file_split[2]
                target_time = file_split[3].split('_')[:2]
                target_time.insert(0, target)
                save_name = '_'.join(target_time) + '.jpg'
                save_folder = os.path.join(output_folder, folder, cls)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                full_save_name = os.path.join(save_folder, save_name)
                pool.apply_async(convert_data, args=(input_file_name, full_save_name))
elif flag == 'test':
    for folder in os.listdir(input_folder):
        folder_name = os.path.join(input_folder, folder)
        for file in os.listdir(folder_name):
            print(file)
            input_file_name = os.path.join(folder_name, file)
            file_split = file.split('.')
            target_time = file_split[3]
            save_name = target_time.split('_')[1] + '.jpg'
            save_folder = os.path.join(output_folder, folder)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            full_save_name = os.path.join(save_folder, save_name)
            pool.apply_async(convert_data, args=(input_file_name, full_save_name))

pool.close()
pool.join()
