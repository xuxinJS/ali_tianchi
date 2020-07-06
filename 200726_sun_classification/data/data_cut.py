# coding=utf-8
# todo 消除无用黑色边界, 过滤地更好一些

import os
import cv2
import numpy as np
from random import randint


def find_roi_min_area(input_image):
    # 找出图片中所有黑色点的大致轮廓,并做一个大框将其所有都包含进去
    # config
    kernel = np.ones((20, 20), np.uint8)
    min_area = 40

    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    dilation = cv2.dilate(gray, kernel, iterations=2)
    # cv2.imshow('dilate', gray2)
    erosion = cv2.erode(dilation, kernel, iterations=2)
    # cv2.imshow('erode', gray2)
    edges = cv2.absdiff(gray, erosion)
    # cv2.imshow('edges', edges)
    x = cv2.Sobel(edges, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(edges, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # cv2.imshow('dst', dst)
    thresh = np.mean(dst) * 2  # dynamic threshold
    final_thresh = thresh if thresh < 200 else 200
    ret, ddst = cv2.threshold(dst, final_thresh, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(ddst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    big_area_list = []
    find_flag = False
    if len(contours):
        for c in contours:
            area = cv2.contourArea(c)
            if area > min_area:
                # cv2.drawContours(image, c, -1, (0, 0, 100), 1)
                xmin = np.min(c[:, :, 0])
                xmax = np.max(c[:, :, 0])
                ymin = np.min(c[:, :, 1])
                ymax = np.max(c[:, :, 1])
                big_area_list.append(np.array([xmin, xmax, ymin, ymax]))
        if len(big_area_list):
            find_flag = True
            big_area_array = np.array(big_area_list)
            xmin = np.min(big_area_array[:, 0])
            xmax = np.max(big_area_array[:, 1])
            ymin = np.min(big_area_array[:, 2])
            ymax = np.max(big_area_array[:, 3])
    if not find_flag:
        xmin = 0
        xmax = gray.shape[1]
        ymin = 0
        ymax = gray.shape[0]
    return xmin, xmax, ymin, ymax


def random_expand(height, width, init_coor, expand_ratio=None):
    # 将找到的大框随机扩展
    # c:expand_ratio=None, will use random ratio
    # testing: ratio=0.2
    min_expand_pixel = 10

    if expand_ratio is None:
        expand_list = np.linspace(0.1, 0.55, num=10)
        expand_len = len(expand_list) - 1
        w_expand_index = randint(0, expand_len)
        h_expand_index = randint(0, expand_len)
        w_ratio = expand_list[w_expand_index]
        h_ratio = expand_list[h_expand_index]
    else:
        w_ratio = expand_ratio
        h_ratio = expand_ratio

    xmin, xmax, ymin, ymax = init_coor
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    w_2 = center_x - xmin
    h_2 = center_y - ymin

    _w_expand = w_2 * w_ratio
    _h_expand = h_2 * h_ratio
    w_expand_index = w_ratio / 0.05
    h_expand_index = h_ratio / 0.05
    w_expand = _w_expand if _w_expand > min_expand_pixel * w_expand_index else min_expand_pixel * w_expand_index
    h_expand = _h_expand if _h_expand > min_expand_pixel * h_expand_index else min_expand_pixel * h_expand_index

    _w_2 = w_2 + w_expand
    _h_2 = h_2 + h_expand
    _xmin = int(center_x - _w_2 if center_x - _w_2 >= 0 else 0)
    _ymin = int(center_y - _h_2 if center_y - _h_2 >= 0 else 0)
    _xmax = int(center_x + _w_2 if center_x + _w_2 <= width else width)
    _ymax = int(center_y + _h_2 if center_y + _h_2 <= height else height)
    return _xmin, _xmax, _ymin, _ymax


if __name__ == '__main__':
    input_folder = '/home/xuxin/Desktop/simple_all'
    for name in os.listdir(input_folder):
        image_name = os.path.join(input_folder, name)
        image = cv2.imread(image_name)
        coor = find_roi_min_area(image)
        _xmin, _xmax, _ymin, _ymax = random_expand(image.shape[0], image.shape[1], coor)
        cv2.rectangle(image, (_xmin, _ymin), (_xmax, _ymax), (255, 0, 0), 1)
        cv2.imshow('image', image)
        if cv2.waitKey(0) == ord('q'):
            break
