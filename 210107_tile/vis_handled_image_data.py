import os
import cv2
import argparse


def main(opt):
    image_data = cv2.imread(opt.image)
    w, h = image_data.shape[:2]
    with open(opt.label, "r") as f:
        data_lines = f.readlines()
        for data_line in data_lines:
            class_name, x_center, y_center, b_width, b_height = [float(item) for item in data_line.split()]

            ptLeftTop = (int(x_center*w-b_width*w/2), int(y_center*h-b_height*h/2))
            ptRightBottom = (int(x_center*w+b_width*w/2), int(y_center*h+b_height*h/2))
            print(ptLeftTop,ptRightBottom,x_center, y_center, b_width, b_height,class_name)
            point_color = (0, 255, 0)  # BGR
            thickness = 1
            lineType = 1
            image_data = cv2.rectangle(image_data, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
    cv2.imshow("win", cv2.resize(image_data, (720, 720)))
    cv2.waitKey(0)
    cv2.destroyWindow("win")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str,
                        default='/data/datasets/ceramic_defect_detection/train/train_images/images/cutted/4/2_245_166_t20201128145539306_CAM1.jpg',
                        help='initial weights path')
    parser.add_argument('--label', type=str,
                        default='/data/datasets/ceramic_defect_detection/train/train_images/labels/cutted/4/2_245_166_t20201128145539306_CAM1.txt',
                        help='model.yaml path')

    opt = parser.parse_args()
    main(opt)
