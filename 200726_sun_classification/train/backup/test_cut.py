import os
import sys
import cv2

from glob import glob
import numpy as np
import datetime as dt
from argparse import ArgumentParser
from sklearn import metrics
from tqdm import tqdm
import seaborn as sn
import tensorflow as tf

from os import path as op
from keras import backend as K
from matplotlib import pyplot as plt

from model_lib import *
sys.path.append('../data')
from data_cut import find_roi_min_area, random_expand

# control CUDA/tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument('-m', help='backbone_name', required=True, type=str)
    parser.add_argument('-pw', help='pretrained_weights path', required=True, type=str)
    parser.add_argument('-t', '--test_dir', help='folder of the testing data', required=True, type=str)
    parser.add_argument('-o', help='Path to save log', type=str, default='.')
    parser.add_argument('-gpu', default='0', type=str)
    return parser


if __name__ == '__main__':
    args = build_argparser().parse_args()
    backbone_name = args.m
    weight = args.pw
    test_dir = args.test_dir
    start_time = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = op.join(op.abspath(args.o), start_time)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    gpu = args.gpu
    class_ids = {'alpha': 0, 'beta': 1, 'betax': 2}
    class_num = len(class_ids)

    # label_file = args.label
    # label = open(label_file, 'r')
    # label_lines = label.readlines()
    # label.close()
    # label_dict = {}  # name:
    # for i in label_lines:
    #     data = i.strip().split(' ')
    #     label_dict[data[0]] = int(data[1])

    test_names = []
    test_labels = []  # if not test acc,will be empty
    pred_labels = []
    for name in os.listdir(test_dir):
        full_name = op.join(test_dir, name)
        if op.isdir(full_name):
            # test acc
            tmp_names = glob(op.join(full_name, '*'))
            tmp_indexs = [class_ids[name]] * len(tmp_names)
            test_names.extend(tmp_names)
            test_labels.extend(tmp_indexs)
        else:
            # only save result
            test_names.append(full_name)

    # keras config
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    keras_config = tf.ConfigProto()
    keras_config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    with tf.Session(config=keras_config) as sess:
        K.set_session(sess)
        if backbone_name == 'mobilenet':
            image_height = 224
            image_width = 224
            model, process_input = mobilenet(input_size=(image_height, image_width, 3),
                                             num_classes=class_num)
        elif backbone_name == 'resnet50':
            image_height = 224
            image_width = 224
            model, process_input = resnet50(input_size=(image_height, image_width, 3),
                                            num_classes=class_num)
        elif backbone_name == 'xception':
            image_height = 299
            image_width = 299
            model, process_input = xception(input_size=(image_height, image_width, 3),
                                            num_classes=class_num)
        elif backbone_name == 'inception_resnetv2':
            image_height = 299
            image_width = 299
            model, process_input = inresv2(input_size=(image_height, image_width, 3),
                                            num_classes=class_num)

        model.load_weights(weight)

        # image_generator = test_generator(test_names, (image_height, image_width), process_input)
        # if save_folder:
        #     error_folder = op.join(save_folder, 'error')
        #     good_folder = op.join(save_folder, 'good')
        #     evaluate_dir_exist_create(error_folder)
        #     evaluate_dir_exist_create(good_folder)
        #     occlusion_key = {0: 'no occlusion', 1: 'occlusion'}

        for name in tqdm(test_names):
            image = cv2.imread(name)
            # cut image
            coor = find_roi_min_area(image)
            xmin, xmax, ymin, ymax = random_expand(image.shape[0], image.shape[1], coor, 0.2)
            image_cut = image[ymin:ymax, xmin:xmax, :]

            infer_image = cv2.resize(image_cut, (image_width, image_height))
            infer_image = process_input(infer_image)
            infer_image = np.expand_dims(infer_image, axis=0)
            init_result = model.predict(infer_image)
            init_result = np.squeeze(init_result)
            result = np.argmax(init_result)
            pred_labels.append(result)
            # print(name, init_result)

    # write tianchi update result
    txt_name = "C_aurora_xin.txt"
    full_txt_name = op.join(output_dir, txt_name)
    with open(full_txt_name, 'w') as f:
        for index, image_name in enumerate(test_names):
            simple_name, _ = op.splitext(op.basename(image_name))
            pred_label = pred_labels[index] + 1
            f.write("%s %d\n" % (simple_name, pred_label))

    if len(test_labels):
        # write accuracy result txt
        acc_name = "acc.txt"
        full_acc_name = op.join(output_dir, acc_name)
        with open(full_acc_name, 'w') as f:
            f.write('full model name:%s\n' % weight)
            f.write('----------------confusion matrix----------------\n')
            cm = metrics.confusion_matrix(test_labels, pred_labels)
            f.write(str(cm))

            f.write("\n\n----------precisions, recall, f1_score----------\n")
            precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_labels, pred_labels)
            f.write("\tprecision\trecall\t\tf1_score\n")
            for i in range(class_num):
                f.write("%s\t%.5f\t\t%.5f\t\t%.5f\n" % (i, precisions[i], recall[i], f1_score[i]))

            f.write("average precision:%.5f\n" % np.mean(precisions))
            f.write("average recall:%.5f\n" % np.mean(recall))
            f.write("average f1_score:%.5f\n" % np.mean(f1_score))

        # save error result
        index_arr = np.arange(len(test_names))
        test_labels_arr = np.array(test_labels)
        pred_labels_arr = np.array(pred_labels)
        error_bool_index = (test_labels_arr != pred_labels_arr)
        index_err = index_arr[error_bool_index]
        error_folder = op.join(output_dir, 'error')
        if not os.path.exists(error_folder):
            os.makedirs(error_folder)
        error_folder_resize = op.join(output_dir, 'error_resize')
        if not os.path.exists(error_folder_resize):
            os.makedirs(error_folder_resize)
        for index in index_err:
            err_image_name = test_names[index]
            err_img_init = cv2.imread(err_image_name)

            err_img = err_img_init.copy()
            cv2.putText(err_img, "GT:%d Pred:%d" % (test_labels_arr[index], pred_labels_arr[index]), (0, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), thickness=1)
            save_image_name = op.join(error_folder, op.basename(err_image_name))
            cv2.imwrite(save_image_name, err_img)

            err_img_resize = cv2.resize(err_img_init, (image_width, image_height))
            cv2.putText(err_img_resize, "GT:%d Pred:%d" % (test_labels_arr[index], pred_labels_arr[index]), (0, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), thickness=1)
            save_image_name = op.join(error_folder_resize, op.basename(err_image_name))
            cv2.imwrite(save_image_name, err_img_resize)

        # plot show confusion metrics heamap
        # sn.heatmap(cm, annot=True, cmap=plt.cm.Oranges, fmt="d")
        # plt.ylabel('Ground Truth')
        # plt.xlabel('Prediction')
        # plt.title('confusion_matrix')
        # plt.show()
