import cv2
import os
import numpy as np


class FindRoi:
    def __init__(self):
        self.min_roi = 7  # 最小黑点的面积
        self.min_big_roi = 20  # 大黑点的最小面积
        self.min_dilate_kernel = np.ones((10, 10), np.uint8)
        self.max_dilate_kernel = np.ones((40, 40), np.uint8)

    # 找出图片中黑色点的大致轮廓
    def find_roi(self, input_image, min_area=10, max_area=None):
        """
        找到太阳黑子的图中的黑点
        Args:
            input_image: BGR图
            min_area: 如果指定,找到黑点的圆的最小面积需大于此值  30
            max_area: 如果指定,找到黑点的圆的最大面积需小于此值  200

        Returns:
            找到的黑点的mask
        """
        kernel = np.ones((15, 15), np.uint8)
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        gray_average = np.mean(gray)
        contour_mask = np.zeros(gray.shape, dtype=np.uint8)

        # 找到黑点
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=2)  # 闭运算，将黑点滤除
        edges = cv2.absdiff(gray, closed)  # 和原图相减得出黑点
        _thresh_value = np.mean(edges) * 3  # dynamic threshold
        thresh_value = _thresh_value if _thresh_value < 200 else 200
        ret, thresh = cv2.threshold(edges, thresh_value, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 过滤黑点
        valid_contours = []
        find_roi_flag = False
        for c in contours:
            area = cv2.contourArea(c)
            area_valid_flag = False
            if max_area is None:
                if area >= min_area:
                    area_valid_flag = True
            else:
                if min_area <= area <= max_area:
                    area_valid_flag = True

            if area_valid_flag:
                # 过滤长条形黑边
                (x, y, _w, _h) = cv2.boundingRect(c)

                # 过滤在边角上的黑边
                xmax = x + _w
                ymax = y + _h
                if x <= 2 or y <= 2 or xmax >= width - 2 or ymax >= height - 2:
                    continue

                # 过滤长 / 宽大于4的黑边
                if _w < _h:
                    w = _h
                    h = _w
                else:
                    w = _w
                    h = _h
                if w > 4 * h:
                    continue

                # 过滤白点  均值大于一定值
                gray_zeros = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(gray_zeros, [c], -1, 1, thickness=-1)
                roi = gray_zeros * gray.copy()
                roi_sum = np.sum(roi)
                roi[roi >= 1] = 1
                roi_pixel_num = np.sum(roi)
                roi_mean = roi_sum / roi_pixel_num
                if roi_mean > 0.95 * gray_average:
                    continue
                find_roi_flag = True
                valid_contours.append(c)

        cv2.drawContours(contour_mask, valid_contours, -1, 255, -1)

        # 可视化找到的黑点
        # contour_mask_bgr = cv2.merge([contour_mask, contour_mask_copy, contour_mask_copy])
        # dst = cv2.addWeighted(input_image, 0.7, contour_mask_bgr, 0.3, 0)
        # cv2.imshow('image', dst)
        # cv2.imshow('contour_mask', contour_mask)
        return find_roi_flag, contour_mask

    def dilate_mask(self, mask, dilate_kernel):
        """
        将roi mask膨胀以覆盖地更全面
        Args:
            mask: valid：255 invalid：0
            dilate_kernel: 找到黑点的轮廓后向外膨胀的卷积核 np.ones((10, 10), np.uint8)

        Returns:
            dilated mask
        """
        dilated_mask = cv2.dilate(mask, dilate_kernel, iterations=1)
        dilated_mask[dilated_mask > 1] = 255
        return dilated_mask

    def valid_coor(self, mask, min_height=None, min_width=None):
        """
        找到mask有效区域最大外接矩形的坐标
        Args:
            mask: valid：255 invalid：0
        Returns:
            xmin, xmax, ymin, ymax
        """
        mask_height = mask.shape[0]
        mask_width = mask.shape[1]
        y_index, x_index = np.where(mask > 1)
        if len(y_index) > 1 and len(x_index) > 1:
            index_ymin = np.min(y_index)
            index_ymax = np.max(y_index)
            index_xmin = np.min(x_index)
            index_xmax = np.max(x_index)

            if min_height is not None:
                min_height = min_height if min_height > index_ymax - index_ymin else index_ymax - index_ymin
                if min_height > mask_height:
                    min_height = mask_height

                half_min_height = min_height / 2
                roi_height_center = (index_ymax + index_ymin) / 2
                _ymin = roi_height_center - half_min_height
                _ymax = roi_height_center + half_min_height
                ymin = int(_ymin if _ymin > 0 else 0)
                ymax = int(_ymax if _ymax < mask_height else mask_height)
            if min_width is not None:
                min_width = min_width if min_width > index_xmax - index_xmin else index_xmax - index_xmin
                if min_width > mask_width:
                    min_width = mask_width
                half_min_width = min_width / 2
                roi_width_center = (index_xmax + index_xmin) / 2
                _xmin = roi_width_center - half_min_width
                _xmax = roi_width_center + half_min_width
                xmin = int(_xmin if _xmin > 0 else 0)
                xmax = int(_xmax if _xmax < mask_width else mask_width)
        else:
            # 没有找到有效roi
            ymin = 0
            ymax = mask_height
            xmin = 0
            xmax = mask_width
        return xmin, xmax, ymin, ymax

    # 通道拼接
    def concat_data(self, gray, g_channel=None, r_channel=None):
        if g_channel is None and r_channel is None:
            bgr = cv2.merge([gray, gray, gray])
        elif g_channel is not None and r_channel is None:
            bgr = cv2.merge([gray, g_channel, gray])
        elif g_channel is None and r_channel is not None:
            bgr = cv2.merge([gray, gray, r_channel])
        else:
            bgr = cv2.merge([gray, g_channel, r_channel])
        return bgr


if __name__ == '__main__':
    roi = FindRoi()
    input_folder = '/home/xuxin/data/sun_classification/data_gen/cut/val/betax'
    for name in os.listdir(input_folder):
        image_name = os.path.join(input_folder, name)
        image = cv2.imread(image_name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, mask = roi.find_roi(image, roi.min_big_roi)
        dilate_mask = roi.dilate_mask(mask, roi.min_dilate_kernel)
        concat_image = roi.concat_data(gray, None, dilate_mask)

        # xmin, xmax, ymin, ymax = roi.valid_coor(dilate_mask, min_height=299, min_width=299)
        # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
        # cv2.imshow('image', image)
        # cv2.imshow('mask', mask)
        cv2.imshow('concat_image', concat_image)
        if cv2.waitKey(0) == ord('q'):
            break
