import os
import cv2
import warnings

import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
from matplotlib.colors import LogNorm

warnings.filterwarnings(action='ignore')
figure = plt.figure()

folder = '/home/xuxin/Desktop/1'
for i in os.listdir(folder):
    print(i)
    # ---------------read fits file------------------
    file_name = os.path.join(folder, i)
    hdul = fits.open(file_name)
    hdul.verify("fix")  # slove error
    image_data = hdul[1].data
    hdul.close()

    # ---------------plot show----------------------
    plt.subplot(211)
    plt.imshow(image_data, cmap='gray')
    plt.subplot(212)
    plt.imshow(image_data, cmap='gray', norm=LogNorm())
    plt.colorbar()

    # ---------------cv2 nomalized show-------------
    print('Min:', np.min(image_data))
    print('Max:', np.max(image_data))
    print('Mean:', np.mean(image_data))
    print('Stdev:', np.std(image_data))
    imin = np.min(image_data)
    imax = np.max(image_data)
    cv_im = ((image_data - imin) / (imax - imin)).astype(np.float32)
    cv2.imshow('cv', cv_im)
    cv2.waitKey(0)

    plt.show()
