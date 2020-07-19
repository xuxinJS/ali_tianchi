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


def image_cut2(roi_obj, image_name, output_name, image_name2, output_name2, min_h, min_w):
    # 同时迭代裁剪con和mag图片
    image = cv2.imread(image_name)
    ret, mask = roi_obj.find_roi(image, roi_obj.min_roi)
    dilate_mask = roi_obj.dilate_mask(mask, roi_obj.max_dilate_kernel)

    xmin, xmax, ymin, ymax = roi_obj.valid_coor(dilate_mask, min_height=min_h, min_width=min_w)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cut = gray[ymin:ymax, xmin:xmax]
    cv2.imwrite(output_name, cut)

    image2 = cv2.imread(image_name2, 0)
    cut2 = image2[ymin:ymax, xmin:xmax]
    cv2.imwrite(output_name2, cut2)


def image_cut_concat(roi_obj, image_name, image_name2, output_name, min_h, min_w):
    image = cv2.imread(image_name)
    image2 = cv2.imread(image_name2, 0)
    ret, mask = roi_obj.find_roi(image, roi_obj.min_roi)
    dilate_mask = roi_obj.dilate_mask(mask, roi_obj.min_dilate_kernel)
    coor_mask = roi.dilate_mask(mask, roi.max_dilate_kernel)
    xmin, xmax, ymin, ymax = roi_obj.valid_coor(coor_mask, min_height=min_h, min_width=min_w)
    b_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b_cut = b_gray[ymin:ymax, xmin:xmax]
    g_cut = image2[ymin:ymax, xmin:xmax]
    r_cut = dilate_mask[ymin:ymax, xmin:xmax]
    merge_cut = cv2.merge([b_cut, g_cut, r_cut])
    cv2.imwrite(output_name, merge_cut)


def iter_cut(src, dst):
    # 迭代文件夹裁剪图片
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


def iter_cut2(src, src2, dst, dst2):
    # 同时迭代裁剪con和mag文件夹的图片
    if not os.path.exists(dst):
        os.makedirs(dst)
    if not os.path.exists(dst2):
        os.makedirs(dst2)
    for i in tqdm(os.listdir(src)):
        src_path = os.path.join(src, i)
        src_path2 = os.path.join(src2, i)
        if os.path.isdir(src_path):
            dst_path = os.path.join(dst, i)
            dst_path2 = os.path.join(dst2, i)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            if not os.path.exists(dst_path2):
                os.makedirs(dst_path2)
            iter_cut2(src_path, src_path2, dst_path, dst_path2)
        else:
            src_name = os.path.join(src, i)
            dst_name = os.path.join(dst, i)
            src_name2 = os.path.join(src2, i)
            dst_name2 = os.path.join(dst2, i)
            image_cut2(roi, src_name, dst_name, src_name2, dst_name2, min_height, min_width)


def iter_concat(src, dst):
    # 迭代文件夹将mask和图片拼接在一起
    if not os.path.exists(dst):
        os.makedirs(dst)
    for i in tqdm(os.listdir(src)):
        src_path = os.path.join(src, i)
        if os.path.isdir(src_path):
            dst_path = os.path.join(dst, i)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            iter_concat(src_path, dst_path)
        else:
            src_name = os.path.join(src, i)
            dst_name = os.path.join(dst, i)
            image = cv2.imread(src_name)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, mask = roi.find_roi(image, roi.min_roi)
            dilate_mask = roi.dilate_mask(mask, roi.min_dilate_kernel)
            concat_image = roi.concat_data(gray, None, dilate_mask)
            cv2.imwrite(dst_name, concat_image)


def iter_mask(src, dst):
    # 迭代文件夹生成mask
    if not os.path.exists(dst):
        os.makedirs(dst)
    for i in tqdm(os.listdir(src)):
        src_path = os.path.join(src, i)
        if os.path.isdir(src_path):
            dst_path = os.path.join(dst, i)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            iter_mask(src_path, dst_path)
        else:
            src_name = os.path.join(src, i)
            dst_name = os.path.join(dst, i)
            image = cv2.imread(src_name)
            ret, mask = roi.find_roi(image, min_area=roi.min_big_roi)
            dilate_mask = roi.dilate_mask(mask, roi.min_dilate_kernel)
            cv2.imwrite(dst_name, dilate_mask)


def iter_cut_concat(src, src2, dst):
    # 迭代文件夹裁剪图片
    # 将con mag con生成的mask合并
    # src放con目录，src2放mag的目录，两个目录里的内容要保持一致
    if not os.path.exists(dst):
        os.makedirs(dst)
    for i in tqdm(os.listdir(src)):
        src_path = os.path.join(src, i)
        src2_path = os.path.join(src2, i)
        if os.path.isdir(src_path):
            dst_path = os.path.join(dst, i)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            iter_cut_concat(src_path, src2_path, dst_path)
        else:
            dst_name = os.path.join(dst, i)
            image_cut_concat(roi, src_path, src2_path, dst_name, min_height, min_width)


if __name__ == '__main__':
    min_height = 299
    min_width = 299
    roi = FindRoi()
    input_folder = '/home/dls1/simple_data/classification/test_cut/con'
    input_folder2 = '/home/dls1/simple_data/classification/test/mag'
    output_folder = '/home/dls1/simple_data/classification/test_concat/con_cv'
    output_folder2 = '/home/dls1/simple_data/classification/test_cut/mag'

    # iter_cut2(input_folder, input_folder2, output_folder, output_folder2)
    # iter_mask(input_folder, output_folder)
    # iter_cut_concat(input_folder, input_folder2, output_folder)
    iter_concat(input_folder, output_folder)