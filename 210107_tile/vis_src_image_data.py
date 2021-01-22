import os
import cv2
import argparse
import json


def main(opt):
    image_data = cv2.imread(opt.image)
    w, h = image_data.shape[:2]
    with open(opt.label, "r") as f:
        label_data = json.loads(f.read())
        for label_item in label_data:
            name, image_height, image_width, category, bbox = label_item.get("name"), label_item.get(
                "image_height"), label_item.get("image_width"), \
                                                              label_item.get("category"), label_item.get("bbox")
            if name == os.path.basename(opt.image):
                print(name, image_height, image_width, category, bbox)
                ptLeftTop = (int(bbox[0]), int(bbox[1]))
                ptRightBottom = (int(bbox[2]), int(bbox[3]))
                point_color = (0, 255, 0)  # BGR
                thickness = 4
                lineType = 4
                image_data = cv2.rectangle(image_data, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

    cv2.imshow("win", cv2.resize(image_data, (1280, 720)))
    cv2.waitKey(0)
    cv2.destroyWindow("win")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str,
                        default='/data/datasets/ceramic_defect_detection/images/tile_round1_train_20201231/train_imgs/245_166_t20201128145539306_CAM1.jpg',
                        help='initial weights path')
    parser.add_argument('--label', type=str,
                        default='/data/datasets/ceramic_defect_detection/images/tile_round1_train_20201231/train_annos.json',
                        help='model.yaml path')

    opt = parser.parse_args()
    main(opt)
