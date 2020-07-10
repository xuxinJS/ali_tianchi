# coding=utf-8
import os
import cv2
import numpy as np
import multiprocessing


# 找出图片中黑色点的大致轮廓
def find_roi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # config
    kernel = np.ones((25, 25), np.uint8)
    min_area = 25
    max_area = 200
    contour_mask = np.zeros(gray.shape, dtype=np.uint8)
    final_mask = contour_mask.copy()
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
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            # if area >= max_area:靠近大的排除asdf
            print(area)
            cv2.drawContours(image, c, -1, (255, 0, 0), 2)

    # contours, hierarchy = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (255, 0, 0), 1)
    # cv2.drawContours(final_mask, contours, -1, 255, -1)
    cv2.imshow('image', image)
    # cv2.imshow('mask', final_mask)
    return final_mask


# 找出图片中黑色点的大致轮廓
def find_big_roi(input_image):
    # config
    kernel = np.ones((15, 15), np.uint8)
    dilate_kernel = np.ones((10, 10), np.uint8)
    min_area = 50

    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    gray_average = np.mean(gray)
    contour_mask = np.zeros(gray.shape, dtype=np.uint8)
    contour_mask_copy = contour_mask.copy()

    # 找到黑点
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=2)  # 闭运算，将黑点滤除
    edges = cv2.absdiff(gray, closed)  # 和原图相减得出黑点
    _thresh_value = np.mean(edges) * 2  # dynamic threshold
    thresh_value = _thresh_value if _thresh_value < 200 else 200
    ret, thresh = cv2.threshold(edges, thresh_value, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # _thresh_value = np.mean(gray) * 0.9  # dynamic threshold
    # thresh_value = _thresh_value if _thresh_value < 200 else 200
    # ret, thresh = cv2.threshold(gray.copy(), thresh_value, 255, cv2.THRESH_BINARY)
    # cv2.imshow('thresh', thresh)

    # 过滤黑点
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            # 过滤边角
            (x, y, _w, _h) = cv2.boundingRect(c)
            xmax = x + _w
            ymax = y + _h
            if x <= 0 or y <= 0 or xmax >= width or ymax >= height:
                # 过滤在边角
                continue

            if _w < _h:
                w = _h
                h = _w
            else:
                w = _w
                h = _h
            if w > 4 * h:
                # 过滤长 / 宽大于4的
                continue

            # 过滤白点  均值大于一定值  #todo 有很大部分过滤不掉

            gray_zeros = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(gray_zeros, [c], -1, 1, thickness=-1)
            roi = gray_zeros * gray.copy()
            roi_sum = np.sum(roi)
            roi[roi >= 1] = 1
            roi_pixel_num = np.sum(roi)
            roi_mean = roi_sum / roi_pixel_num
            if roi_mean > 0.95 * gray_average:
                continue
                # cv2.imshow('gray_zeros', gray_zeros)
                # cv2.waitKey(0)

            print(roi_mean, gray_average)
            valid_contours.append(c)

    # 可视化找到的黑点
    cv2.drawContours(contour_mask, valid_contours, -1, 255, -1)
    # contour_mask = cv2.dilate(contour_mask, dilate_kernel, iterations=1)
    contour_mask_bgr = cv2.merge([contour_mask, contour_mask_copy, contour_mask_copy])
    dst = cv2.addWeighted(input_image, 0.7, contour_mask_bgr, 0.3, 0)
    cv2.imshow('image', dst)
    cv2.imshow('contour_mask', contour_mask)
    print()


if __name__ == '__main__':
    # input_folder = '/home/dls1/simple_data/data_gen/0703_con'
    # g_channel_folder = '/home/dls1/simple_data/data_gen/0703_mag'
    # output_folder = '/home/dls1/simple_data/data_gen/0703_con_mag_cv'
    # cores = 12
    # pool = multiprocessing.Pool(processes=cores)
    #
    # for split in os.listdir(input_folder):
    #     folder_name = os.path.join(input_folder, split)
    #     for cls in os.listdir(folder_name):
    #         class_folder = os.path.join(folder_name, cls)
    #         save_folder = os.path.join(output_folder, split, cls)
    #         if not os.path.exists(save_folder):
    #             os.makedirs(save_folder)
    #         for file in os.listdir(class_folder):
    #             print(file)
    #             if g_channel_folder is not None:
    #                 g_channel_name = os.path.join(g_channel_folder, split, cls, file)
    #             else:
    #                 g_channel_name = None
    #             input_file_name = os.path.join(class_folder, file)
    #             full_save_name = os.path.join(save_folder, file)
    #             pool.apply_async(convert_data, args=(input_file_name, full_save_name, g_channel_name))
    #
    # pool.close()
    # pool.join()

    input_folder = '/home/dls1/simple_data/classification/test'
    for name in os.listdir(input_folder):
        image_name = os.path.join(input_folder, name)
        image = cv2.imread(image_name)
        find_big_roi(image)
        if cv2.waitKey(0) == ord('q'):
            break
