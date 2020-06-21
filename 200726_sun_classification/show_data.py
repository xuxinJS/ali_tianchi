# coding=utf-8
# 归一化：将图片缩放到最大值和最小值之间，再缩放到-1～1
# 是否要过滤在直方图上少的像素
# 是否要裁剪图片
# 将con和mag的数据用传统cv合并
import os
import cv2
import warnings

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



input_folder = '/home/xuxin/data/sun_classification/t'
output_folder = '/home/xuxin/data/sun_classification/t_c'
for folder in os.listdir(input_folder):
    folder_name = os.path.join(input_folder, folder)
    for cls in os.listdir(folder_name):
        class_folder = os.path.join(folder_name, cls)
        for file in os.listdir(class_folder):
            input_file_name = os.path.join(class_folder, file)
            file_split = file.split('.')
            save_split = file_split[2:5]
            save_name = ('_').join(save_split) + '.jpg'
            save_folder = os.path.join(output_folder, folder, cls)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            full_save_name = os.path.join(save_folder, save_name)
            convert_data(input_file_name, full_save_name)



