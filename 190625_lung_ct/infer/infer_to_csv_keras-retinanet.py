import keras

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import csv

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from glob import glob
from os import path

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
import SimpleITK as sitk


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def load_itk(filename):
    '''
    This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
    '''
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # indexes are z,y,x (notice the ordering)
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # z,y,x  Origin in voxel coordinates
    # voxel 体素
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # spacing of voxels to world coor(mm) z,y,x
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


def main():
    labels_to_names = {0: 'jj', 1: 'st', 2: 'dmyh_gh', 3: 'lbj'}
    labels_to_id = {0: 1, 1: 5, 2: 31, 3: 32}
    directory = '/home/xuxin/simple_data/tiaochi/test_dataset'
    mhd_directory = '/HHD/xuxin/data/train_data/tianchi/190625_CT/dataset/testA'


    flag_show = False
    flag_save_csv = True

    if flag_save_csv:
        csvfile = open("resnet145_45.csv", "w")
        writer = csv.writer(csvfile)
        writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'class', 'probability'])
        coor_convert = {}
        file_list = glob(mhd_directory + "/*.mhd")
        for file_name in file_list:
            file_id = path.basename(file_name).split('.')[0]
            _, origin, spacing = load_itk(file_name)  # z,y,x
            coor_convert[file_id] = [origin, spacing]
        # print(coor_convert)

    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())
    # adjust this to point to your downloaded/trained model
    # models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
    model_path = os.path.join('/HHD/xuxin/github_download/keras-retinanet/infer/res152_45.h5')

    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet152')

    # if the model is not converted to an inference model, use the line below
    # see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
    # model = models.convert_model(model)

    # print(model.summary())

    for i in os.listdir(directory):
        if flag_save_csv:
            mhd_id = i.split('_')[0]
            z_slice = (i.split('.')[0]).split('_')[1]
            origin = (coor_convert[mhd_id])[0]
            spacing = (coor_convert[mhd_id])[1]

        image_name = os.path.join(directory, i)
        # load image
        image = read_image_bgr(image_name)

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

        print("processing time: ", time.time() - start)

        # correct for image scale
        boxes /= scale

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5:
                break
            if flag_save_csv:
                box_center_x = (box[0] + box[2]) / 2
                box_center_y = (box[1] + box[3]) / 2
                voxel_x = box_center_x * spacing[2] + origin[2]
                voxel_y = box_center_y * spacing[1] + origin[1]
                voxel_z = int(z_slice) * spacing[0] + origin[0]
                voxel_label = labels_to_id[label]
                # print(box) # x1,y1,x2,y2
                # seriesuid,coordX,coordY,coordZ,class,probability
                print(mhd_id, voxel_x, voxel_y, voxel_z, voxel_label, score)
                writer.writerow([mhd_id, voxel_x, voxel_y, voxel_z, voxel_label, score])

            if flag_show:
                color = label_color(label)
                b = box.astype(int)
                draw_box(draw, b, color=color)
                caption = "{} {:.3f}".format(labels_to_names[label], score)
                draw_caption(draw, b, caption)

            if flag_show:
                plt.figure(figsize=(15, 15))
                plt.axis('off')
                plt.imshow(draw)
                plt.show()
    if flag_save_csv:
        csvfile.close()

if __name__ == '__main__':
    main()
