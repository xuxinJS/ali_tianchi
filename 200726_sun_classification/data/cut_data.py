# coding=utf-8
# 将大图裁剪下来，好让roi所占的比重更大
import os
import cv2
from tqdm import tqdm
from lib.find_roi import FindRoi


def image_cut(roi_obj, image_name, output_name, min_h, min_w):
    image = cv2.imread(image_name)
    ret, mask = roi_obj.find_roi(image, roi_obj.min_roi)
    dilate_mask = roi_obj.dilate_mask(mask, roi_obj.max_dilate_kernel)
    xmin, xmax, ymin, ymax = roi_obj.valid_coor(dilate_mask, min_height=min_h, min_width=min_w)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cut = gray[ymin:ymax, xmin:xmax]
    cv2.imwrite(output_name, cut)


def iter_cut(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for i in tqdm(os.listdir(src)):
        src_path = os.path.join(src, i)
        if os.path.isdir(src_path):
            dst_path = os.path.join(dst, i)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            iter_cut(src_path, dst_path)
        else:
            src_name = os.path.join(src, i)
            dst_name = os.path.join(dst, i)
            image_cut(roi, src_name, dst_name, min_height, min_width)


def iter_concat(src, dst):
    for i in tqdm(os.listdir(src)):
        src_path = os.path.join(src, i)
        if os.path.isdir(src_path):
            dst_path = os.path.join(dst, i)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            iter_cut(src_path, dst_path)
        else:
            src_name = os.path.join(src, i)
            dst_name = os.path.join(dst, i)
            image = cv2.imread(src_name)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = roi.find_roi(image, min_area=roi.min_roi)
            dilate_mask = roi.dilate_mask(mask, roi.min_dilate_kernel)
            concat_image = roi.concat_data(gray, None, dilate_mask)
            cv2.imwrite(dst_name, concat_image)



if __name__ == '__main__':
    min_height = 299
    min_width = 299
    roi = FindRoi()
    input_folder = '/home/dls1/simple_data/data_gen/0705_con'
    output_folder = '/home/dls1/simple_data/data_gen/0705_con_cut'
    iter_cut(input_folder, output_folder)
