import os
import cv2

continuum = '/home/dls1/simple_data/sun_classification/continuum/alpha'
magnetogram = '/home/dls1/simple_data/sun_classification/magnetogram/alpha'
for i in os.listdir(continuum):
    image_continuum_name = os.path.join(continuum, i)
    image_magnetogram_name = os.path.join(magnetogram, i)
    image_continuum = cv2.imread(image_continuum_name, 0)
    image_magnetogram = cv2.imread(image_magnetogram_name, 0)
    diff = abs(image_continuum + image_magnetogram)
    # ret, bin = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY)
    cv2.imshow('image_continuum', image_continuum)
    cv2.imshow('image_magnetogram', image_magnetogram)
    cv2.imshow('diff', diff)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
