# 保存下感觉有疑问的图片

import os
import cv2
import numpy as np

target = 'betax'
continuum_base_dir = '/home/xuxin/data/sun/sun_classification/continuum/'
magnetogram_base_dir = '/home/xuxin/data/sun/sun_classification/magnetogram'
continuum = os.path.join(continuum_base_dir, target)
magnetogram = os.path.join(magnetogram_base_dir, target)

cv2.namedWindow('all', cv2.WINDOW_NORMAL)

for i in os.listdir(continuum):
    image_continuum_name = os.path.join(continuum, i)
    image_magnetogram_name = os.path.join(magnetogram, i)
    image_continuum = cv2.imread(image_continuum_name, 0)
    image_magnetogram = cv2.imread(image_magnetogram_name, 0)
    con_mag_stack = np.hstack([image_continuum, image_magnetogram])

    # sum
    image_continuum = image_continuum.astype(np.int)
    image_magnetogram = image_magnetogram.astype(np.int)
    sum = ((image_continuum + image_magnetogram) / 2).astype(np.uint8)

    # diff
    diff = image_continuum - image_magnetogram
    diff_max = np.max(diff)
    diff_min = np.min(diff)
    diff = diff.astype(np.float)
    diff = (diff - diff_min) / (diff_max - diff_min)
    diff = (diff * 255).astype(np.uint8)

    sum_diff_stack = np.hstack([sum, diff])
    all = np.vstack([con_mag_stack, sum_diff_stack])

    #puttext
    width = all.shape[1]
    height = all.shape[0]
    cv2.putText(all, 'continuum', (0, 25), cv2.FONT_HERSHEY_COMPLEX, 1, 127, 2)
    cv2.putText(all, 'magnetogram', (width // 2, 25), cv2.FONT_HERSHEY_COMPLEX, 1, 127, 2)
    cv2.putText(all, 'sum', (0, height // 2 + 25), cv2.FONT_HERSHEY_COMPLEX, 1, 127, 2)
    cv2.putText(all, 'diff', (width // 2, height // 2 + 25), cv2.FONT_HERSHEY_COMPLEX, 1, 127, 2)

    cv2.imshow('all', all)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
