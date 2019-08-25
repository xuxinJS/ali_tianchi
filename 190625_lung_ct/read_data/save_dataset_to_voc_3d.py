# -*- coding: UTF-8 -*-
#
# save mhd and raw data to filename_zslice.jpg
import csv
import math
import cv2

import SimpleITK as sitk
import numpy as np

import matplotlib.pyplot as plt

from glob import glob
from os import path


def load_itk(filename):
    '''
    This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
    '''
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # indexes are z,y,x (notice the ordering)
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # z,y,x  Origin in voxel coordinates
    # voxel
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # spacing of voxels to world coor(mm) z,y,x
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing



def load_itk_withResample(filename, image_distance=1.0):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # set resample info
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(itkimage.GetDirection())
    resample.SetOutputOrigin(itkimage.GetOrigin())

    # set new spacing
    orig_spacing = itkimage.GetSpacing()
    new_spacing = [orig_spacing[0], orig_spacing[1], image_distance]
    resample.SetOutputSpacing(new_spacing)

    # set new size
    orig_size = np.array(itkimage.GetSize(), dtype=np.int)
    new_size = orig_size * [a / b for a, b in zip(orig_spacing, new_spacing)]
    new_size = np.ceil(new_size).astype(np.int)
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)

    new_images = resample.Execute(itkimage)

    # indexes are z,y,x (notice the ordering)
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # z,y,x  Origin in voxel coordinates
    # voxel
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # spacing of voxels to world coor(mm) z,y,x
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


def resample(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # set resample info
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(itkimage.GetDirection())
    resample.SetOutputOrigin(itkimage.GetOrigin())

    # set new spacing
    orig_spacing = itkimage.GetSpacing()
    new_spacing = [orig_spacing[0], orig_spacing[1], 1.0]
    resample.SetOutputSpacing(new_spacing)

    # set new size
    orig_size = np.array(itkimage.GetSize(), dtype=np.int)
    new_size = orig_size * [a / b for a, b in zip(orig_spacing, new_spacing)]
    new_size = np.ceil(new_size).astype(np.int)
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)

    new_images = resample.Execute(itkimage)

    return new_images




def image_process_range(image):
    '''
    -1000~400:结节 索条 动脉硬化或钙化 淋巴结钙化
    :param image:
    :return: image
    '''
    image_max = 400
    image_min = -1000
    bool_idx_image_min = (image <= image_min)
    image[bool_idx_image_min] = image_min
    bool_idx_image_max = (image >= image_max)
    image[bool_idx_image_max] = image_max
    return image


labels = {0: 1, 1: 5, 2: 31, 3: 32}


def image_save():
    # data_path = '/home/xuxin/work/tianchi/code/read_data/data'
    # save_dir = '/home/xuxin/simple_data/tiaochi/simple_test'
    # data_path = '/home/zexin/project/med_lung/datasets/test/testA'
    # save_dir = '/home/zexin/project/med_lung/datasets/test/testA_images_rgb'
    # data_path = '/Users/app/Desktop/all/lab/project/tainchi/datasets/round_one/mhd/'
    # save_dir = '/Users/app/Desktop/all/lab/project/tainchi/datasets/round_one/image'
    data_path = '/home/admin/jupyter/Demo/DataSets/split/train/'
    save_dir = '/Users/app/Desktop/all/lab/project/tainchi/datasets/round_one/image'
    file_list = glob(data_path + "/*.mhd")

    for file_name in file_list:
        file_id = path.basename(file_name).split('.')[0]
        resample(file_name)
        ct_scan, origin, spacing = load_itk(file_name)
        ct_scan = image_process_range(ct_scan)
        slices = range(ct_scan.shape[0])
        for id in slices:
            if (id == 0):
                start_id = 0
                end_id = 3
            elif (id == ct_scan.shape[0]-1):
                start_id = ct_scan.shape[0]-3
                end_id = ct_scan.shape[0]
            else:
                start_id = id - 1
                end_id = id + 2
            print(ct_scan[start_id:end_id].shape,start_id)
            # plt.imshow(ct_scan[start_id, end_id], cmap='gray')
            plt.imshow(np.transpose(ct_scan[start_id:end_id],(1,2,0)))
            save_file_name = path.join(save_dir, "%s_%d.jpg" % (file_id, id))
            # plt.imsave(save_file_name, ct_scan[id], cmap=plt.cm.gray, format='jpg')
            plt.imsave(save_file_name, ct_scan[id], cmap=plt.cm.gray)
            im = cv2.imread(save_file_name, 0)
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(save_file_name, im)
            # plt.show()
            # print("save", save_file_name)

        # for image_id, image in enumerate(ct_scan[])
        #     cv2.putText(ct_scan[anno[1]], labels[anno[0]], pt1, cv2.FONT_HERSHEY_COMPLEX, 0.3, (127, 0, 0))
        #     cv2.rectangle(ct_scan[anno[1]], pt1, pt2, (255, 0, 0), 1)
        #
        #
        # plt.imshow(ct_scan[id], cmap='gray')
        # plt.show()
        # save_file_name = path.join(save_dir, "%s_%d.jpg" % (file_id, id))
        # plt.imsave(save_file_name, ct_scan[id], cmap=plt.cm.gray, format='jpg')


def write_csv():
    mhd_directory = '/home/xuxin/work/tianchi/code/read_data/data'

    coor_convert = {}
    file_list = glob(mhd_directory + "/*.mhd")
    for file_name in file_list:
        file_id = path.basename(file_name).split('.')[0]
        _, origin, spacing = load_itk(file_name)  # z,y,x
        coor_convert[file_id] = [origin, spacing]
    print((coor_convert['691784'])[0], (coor_convert['691784'])[1])

    csvfile = open("test.csv", "w")
    writer = csv.writer(csvfile)

    # 先写入columns_name
    writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ'])
    # 写入多行用writerows
    writer.writerow(
        ['691784', (coor_convert['691784'])[0][0], (coor_convert['691784'])[0][1], (coor_convert['691784'])[0][2]])

    csvfile.close()


if __name__ == '__main__':
    image_save()
