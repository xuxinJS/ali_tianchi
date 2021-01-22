import os
import argparse
import cv2
import json
import numpy as np
import random
import multiprocessing
from shrink_image import shrink_polygon
import shutil
from tqdm import tqdm
import math
import time


def read_json_file(json_file_path):
    name_key_resp = dict()
    with open(json_file_path, "r") as f:
        json_file = json.loads(f.read())

        for item in json_file:
            name = item.get("name")

            if name_key_resp.get(name) is None:
                name_key_resp[name] = [item]
            else:
                name_key_resp[name].append(item)
    return name_key_resp


def datasets_split(ratio, datasets_list):
    train_r, val_r = ratio
    train_list = random.sample(datasets_list, int(train_r / sum(ratio) * len(datasets_list)))
    val_list = list(set(datasets_list) - set(train_list))
    return train_list, val_list


def wrap_perspective_bbox(A1, bbox):
    x1_, y1_, x2_, y2_ = bbox
    bbox = [(x1_, y1_), (x2_, y1_), (x2_, y2_), (x1_, y2_)]

    points = np.array(bbox, np.int32)
    points = np.array([points])
    wraped_points = cv2.transform(points, A1)
    x1, y1 = wraped_points[0][0][0], wraped_points[0][0][1]
    x2, y2 = wraped_points[0][1][0], wraped_points[0][1][1]
    x3, y3 = wraped_points[0][2][0], wraped_points[0][2][1]
    x4, y4 = wraped_points[0][3][0], wraped_points[0][3][1]

    x, y, w, h = cv2.boundingRect(np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)]))

    return x, y, x + w, y + h


def handle_image(opt, name_key_resp, image_name, datasets_name):
    input_dir, output_dir, transform_matrix_dir, input_label_path = opt.input_dir, opt.output_dir, opt.transform_matrix_dir, opt.input_label_path

    image_data = cv2.imread(os.path.join(input_dir, image_name))
    # image_data = cv2.resize(image_data, dsize=None, fx=0.1, fy=0.1)

    image_height, image_width = image_data.shape[:2]
    avg_number = np.sum(image_data) / (np.prod(image_data.size))

    ret, thresh = cv2.threshold(image_data, avg_number, 255, 0)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_RGB2GRAY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = -1
    max_index = -1
    for c_index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_index = c_index

    contour = cv2.approxPolyDP(contours[max_index], 30, True)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    src = np.array(box, np.float32)

    src = shrink_polygon(src, -0.98)

    left_top, right_top, right_bottom, left_bottom = src
    target_width = max([
        np.sqrt((right_top[0] - left_top[0]) ** 2 + (right_top[1] - left_top[1]) ** 2),
        np.sqrt((right_bottom[0] - left_bottom[0]) ** 2 + (right_bottom[1] - left_bottom[1]) ** 2),
    ])
    target_height = max([
        np.sqrt((left_bottom[0] - left_top[0]) ** 2 + (left_bottom[1] - left_top[1]) ** 2),
        np.sqrt((right_bottom[0] - right_top[0]) ** 2 + (right_bottom[1] - right_top[1]) ** 2),
    ])

    dst = np.array(
        [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
        np.float32)

    A1 = cv2.getPerspectiveTransform(src, dst)
    image = cv2.warpPerspective(image_data, A1, (int(target_width), int(target_height)), borderValue=125)
    transformed_image_height, transformed_image_width = image.shape[:2]

    # save transform matrix
    to_save_transform_matrix_dir = os.path.join(transform_matrix_dir, datasets_name)
    if not os.path.exists(to_save_transform_matrix_dir):
        os.makedirs(to_save_transform_matrix_dir)
    transform_matrix_path = os.path.join(to_save_transform_matrix_dir,
                                         os.path.splitext(image_name)[0] + ".npy")
    np.save(transform_matrix_path, A1)

    # save image file
    to_save_image_dir = os.path.join(output_dir, "images", datasets_name)
    if not os.path.exists(to_save_image_dir):
        os.makedirs(to_save_image_dir)
    dst_image_path = os.path.join(to_save_image_dir, image_name)
    cv2.imwrite(dst_image_path, image)

    # save label file
    to_save_label_dir = os.path.join(output_dir, "labels", datasets_name)
    if not os.path.exists(to_save_label_dir):
        os.makedirs(to_save_label_dir)
    dst_label_path = os.path.join(to_save_label_dir, os.path.splitext(image_name)[0] + ".txt")
    label_list = name_key_resp.get(image_name)
    with open(dst_label_path, "w") as f:
        for label_item in label_list:
            category, bbox = label_item.get("category"), label_item.get("bbox")

            # warp perspective bbox
            bbox = wrap_perspective_bbox(A1, bbox)

            x1, y1, x2, y2 = bbox
            x1_ = min(x1, x2)
            x2_ = max(x1, x2)
            y1_ = min(y1, y2)
            y2_ = max(y1, y2)
            x1_ = 0 if x1_ < 0 else x1_
            y1_ = 0 if y1_ < 0 else y1_
            x2_ = transformed_image_width - 1 if x2 > (transformed_image_width - 1) else x2_
            y2_ = transformed_image_height - 1 if y2 > (transformed_image_height - 1) else y2_

            # image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)
            # print(bbox,image.shape)
            # x_center = (x1 + x2) / 2
            # y_center = (y1 + y2) / 2
            # bbox_width = abs(x2 - x1)
            # bbox_height = abs(y2 - y1)
            #
            # # normalized xywh
            # x_center = x_center / transformed_image_width
            # y_center = y_center / transformed_image_height
            # bbox_width = bbox_width / transformed_image_width
            # bbox_height = bbox_height / transformed_image_height

            f.write(" ".join([str(item) for item in [category-1, x1_, y1_, x2_, y2_]]) + "\n")

    # cv2.imshow("win", cv2.resize(image, dsize=(1280, 720)))
    # cv2.waitKey(0)
    print("handled : {}".format(image_name))


def main(opt):
    input_dir, output_dir, transform_matrix_dir, input_label_path = opt.input_dir, opt.output_dir, opt.transform_matrix_dir, opt.input_label_path

    # read label file
    name_key_resp = read_json_file(input_label_path)

    total_list = os.listdir(input_dir)
    total_dataset_dict = {"raw": total_list}

    # create process pool
    pool = multiprocessing.Pool(processes=24)

    # generate transform_matrix
    for datasets_name, datasets_list in total_dataset_dict.items():
        for image_name in datasets_list:
            if image_name.endswith(".jpg"):
                # if image_name!="245_102_t20201128142608602_CAM2.jpg":
                #     continue
                # handle_image(opt, name_key_resp, image_name, datasets_name)
                pool.apply_async(handle_image, (opt, name_key_resp, image_name, datasets_name))

    pool.close()
    pool.join()


def bbox_max_height_and_width(image_path, label_path):
    area_dict = dict()
    with open(label_path, "r") as f:
        defect_lines = f.readlines()
        for defect_line in defect_lines:
            class_name, x1, y1, x2, y2 = [int(item) for item in defect_line.split()]
            bbox_width = abs(x2 - x1)
            bbox_height = abs(y2 - y1)
            area = bbox_height * bbox_width
            if area_dict.get(class_name) is None:
                area_dict[class_name] = area
            else:
                area_dict[class_name] = min(area_dict[class_name], area)

    return area_dict


def cut_image_by_fixed_size(dst_size, image_path, label_path, class_number, output_dir):
    save_image_threshold = {4: 40, 2: 55, 3: 28, 5: 9, 6: 130, 1: 102, 0: 100}

    image_name = os.path.basename(image_path)
    image_data = cv2.imread(image_path)
    h, w = image_data.shape[:2]
    label_data = np.zeros(shape=(h, w, class_number))
    output_images_dir = os.path.join(output_dir, "images", "cutted")
    output_labels_dir = os.path.join(output_dir, "labels", "cutted")

    dst_image_list = list()
    with open(label_path, "r") as f:
        defect_lines = f.readlines()
        for defect_line in defect_lines:
            class_name, x1, y1, x2, y2 = [int(item) for item in defect_line.split()]

            label_data[y1:y2, x1:x2, class_name] = 1
            dst_image_list.append([class_name, x1, y1, x2, y2])

    for i, dst_image_parm in enumerate(dst_image_list):
        class_name, x1, y1, x2, y2 = dst_image_parm
        if class_name == 4 or class_name == 1 or class_name == 2 or class_name == 3 or class_name == 0:
            continue
        start_x = -1
        start_y = -1
        end_x = w,
        end_y = h
        s_label_data = label_data[:, :, class_name]

        sum_val = 0
        count = 0
        is_continue = False
        while not (sum_val >= save_image_threshold[class_name]):
            start_x = random.randrange(x1 - dst_size + 1, x1)
            start_y = random.randrange(y1 - dst_size + 1, y1)
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = start_x + dst_size
            end_y = start_y + dst_size

            if end_x > (w - 1):
                end_x = w - 1
                start_x = end_x - dst_size
            if end_y > (h - 1):
                end_y = h - 1
                start_y = end_y - dst_size

            start_x, start_y, end_x, end_y = int(start_x), int(start_y), int(end_x), int(end_y)
            if start_x >= 0 and start_y >= 0 and end_x < w and end_y < h:
                sum_val = np.sum(s_label_data[start_y:end_y, start_x:end_x])
            count += 1
            if count > 2000:
                is_continue = True
                break
        if is_continue:
            print("*" * 20, "skip {}".format(class_name))
            continue

        f_output_labels_dir = os.path.join(output_labels_dir, str(class_name))
        if not os.path.exists(f_output_labels_dir):
            os.makedirs(f_output_labels_dir)
        f_output_images_dir = os.path.join(output_images_dir, str(class_name))
        if not os.path.exists(f_output_images_dir):
            os.makedirs(f_output_images_dir)

        dst_image_data = image_data[start_y:end_y, start_x:end_x, :]

        num = int(time.time() * 10000000)

        cv2.imwrite(os.path.join(f_output_images_dir, str(num) + "_" + image_name), dst_image_data)

        for class_index in range(class_number):
            temp_label_data = label_data[:, :, class_index]
            child_image_data = temp_label_data[start_y:end_y, start_x:end_x]
            temp_sum_val = np.sum(child_image_data)
            if temp_sum_val >= save_image_threshold[class_index]:
                # save current label and image
                rect_box_list = find_rect_box(child_image_data, save_image_threshold[class_index])
                for rect_box in rect_box_list:
                    x1, y1, x2, y2 = rect_box
                    with open(
                            os.path.join(f_output_labels_dir,
                                         str(num) + "_" + os.path.splitext(image_name)[0] + ".txt"),
                            "a+") as f:
                        x_center = ((x2 + x1) / 2) / dst_size
                        y_center = ((y2 + y1) / 2) / dst_size
                        bbox_width = (x2 - x1) / dst_size
                        bbox_height = (y2 - y1) / dst_size
                        f.write(" ".join(
                            [str(item) for item in [class_index, x_center, y_center, bbox_width, bbox_height]]) + "\n")

                    print("Save cutted image :{},class: {}".format(image_name, class_index))


def find_rect_box(label_data, save_image_threshold):
    box_list = list()
    label_data[label_data != 0] = 255
    label_data = np.array(label_data, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(label_data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c_index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area >= save_image_threshold:
            rect = cv2.boundingRect(contour)
            x, y, w, h = rect
            box_list.append([x, y, x + w, y + h])

    return box_list


def create_mini_image_datasets(opt):
    output_dir, class_number = opt.output_dir, opt.class_number

    # compute max box
    to_save_label_dir = os.path.join(output_dir, "labels", "raw")
    to_save_image_dir = os.path.join(output_dir, "images", "raw")

    # create process pool
    pool = multiprocessing.Pool(processes=24)
    # pool2 = multiprocessing.Pool(processes=24)
    # processing_list = list()
    #
    # for image_name in os.listdir(to_save_image_dir):
    #     image_path = os.path.join(to_save_image_dir, image_name)
    #     label_name = os.path.splitext(image_name)[0] + ".txt"
    #     label_path = os.path.join(to_save_label_dir, label_name)
    #     processing_list.append(pool.apply_async(bbox_max_height_and_width, (image_path, label_path)))
    #
    # pool.close()
    # pool.join()
    #
    # target_area_dict=dict()
    # for p in processing_list:
    #     area_dict = p.get()
    #     for cls_name,a in area_dict.items():
    #         if target_area_dict.get(cls_name) is None:
    #             target_area_dict[cls_name]=a
    #         else:
    #             target_area_dict[cls_name]=min(target_area_dict[cls_name],a)
    # print(target_area_dict)
    # {4: 40, 2: 55, 3: 28, 5: 9, 6: 130, 1: 102}

    max_size = 512
    print("Current select shape {}".format(max_size))

    for image_name in os.listdir(to_save_image_dir):
        # if image_name != "245_166_t20201128145539306_CAM1.jpg":
        #     continue
        image_path = os.path.join(to_save_image_dir, image_name)
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(to_save_label_dir, label_name)
        # cut_image_by_fixed_size(max_size, image_path, label_path, class_number, output_dir)
        pool.apply_async(cut_image_by_fixed_size, (max_size, image_path, label_path, class_number, output_dir))

    pool.close()
    pool.join()

    pass


def split_data(opt):
    output_dir, class_number = opt.output_dir, opt.class_number

    train_r, val_r = [int(item) for item in opt.train_val_ratio.split(",")]
    to_save_label_dir = os.path.join(output_dir, "labels", "cutted")
    to_save_image_dir = os.path.join(output_dir, "images", "cutted")
    # datasets split
    train_list, val_list = list(), list()

    for i in range(0, 6):
        t_image_path = os.path.join(to_save_image_dir, str(i))
        t_label_path = os.path.join(to_save_label_dir, str(i))
        total_label_list = [os.path.join(t_label_path, item) for item in os.listdir(t_label_path)]
        total_label_list = random.sample(total_label_list, 7000)

        t_train_list = random.sample(total_label_list, int(train_r / (train_r + val_r) * len(total_label_list)))

        t_val_list = list(set(total_label_list) - set(t_train_list))
        train_list += t_train_list
        val_list += t_val_list

    to_dst_train_label_dir = os.path.join(output_dir, "labels", "train")
    to_dst_train_image_dir = os.path.join(output_dir, "images", "train")

    to_dst_val_image_dir = os.path.join(output_dir, "images", "val")
    to_dst_val_label_dir = os.path.join(output_dir, "labels", "val")
    for path in [to_dst_train_image_dir,to_dst_train_label_dir,to_dst_val_image_dir,to_dst_val_label_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
    for train_item in train_list:
        image_path = os.path.join(os.path.dirname(train_item).replace("labels", "images"),
                                  os.path.splitext(os.path.basename(train_item))[0] + ".jpg")
        label_path = train_item
        shutil.copy(image_path,to_dst_train_image_dir)
        shutil.copy(label_path,to_dst_train_label_dir)
        print("Copyed {}".format(image_path))

    for val_item in val_list:
        image_path = os.path.join(os.path.dirname(val_item).replace("labels", "images"),
                                  os.path.splitext(os.path.basename(val_item))[0] + ".jpg")
        label_path = val_item
        shutil.copy(image_path, to_dst_val_image_dir)
        shutil.copy(label_path, to_dst_val_label_dir)
        print("Copyed {}".format(image_path))


    print(len(train_list), len(val_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        default='/data/datasets/ceramic_defect_detection/images/tile_round1_train_20201231/train_imgs/',
                        help='initial weights path')
    parser.add_argument('--output_dir', type=str, default='/data/datasets/ceramic_defect_detection/train/train_images/',
                        help='model.yaml path')
    parser.add_argument('--transform_matrix_dir', type=str,
                        default='/data/datasets/ceramic_defect_detection/train/transform_matrix/',
                        help='data.yaml path')
    parser.add_argument('--input_label_path', type=str,
                        default='/data/datasets/ceramic_defect_detection/images/tile_round1_train_20201231/train_annos.json',
                        help='hyperparameters path')

    parser.add_argument('--train_val_ratio', type=str,
                        default='8,2',
                        help='hyperparameters path')
    parser.add_argument('--class_number', type=int,
                        default=7,
                        help='hyperparameters path')

    opt = parser.parse_args()
    # main(opt)
    # create_mini_image_datasets(opt)
    split_data(opt)
