import os
import cv2
import numpy as np
from os import path as op

folder = '/home/dls1/simple_data/data_gen/0630_con/test/beta'
thresValue = 100
kernel = np.ones((25, 25), np.uint8)
min_area = 20
for name in os.listdir(folder):
    full_name = op.join(folder, name)
    image = cv2.imread(full_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.dilate(gray, kernel, iterations=2)
    # cv2.imshow('dilate', gray2)
    gray2 = cv2.erode(gray2, kernel, iterations=2)
    # cv2.imshow('erode', gray2)
    edges = cv2.absdiff(gray, gray2)
    # cv2.imshow('edges', edges)
    x = cv2.Sobel(edges, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(edges, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    ret, ddst = cv2.threshold(dst, thresValue, 255, cv2.THRESH_BINARY)
    # cv2.imshow('ddst', ddst)
    contours, hierarchy = cv2.findContours(ddst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            print(area)
            cv2.drawContours(image, c, -1, (255, 0, 0), 1)
    cv2.imshow('image', image)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
    print()