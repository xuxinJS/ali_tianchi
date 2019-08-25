import tensorflow as tf
import os
import argparse

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from object_detection.utils import dataset_util

DATASET_SPLIT_LIST = ['train', 'val']
# DATASET_SPLIT_LIST = ['trainval']
CATEGORIES = ['jj', 'fmdzg', 'fqz', 'st', 'dmyh_gh', 'lbj', 'xmzh']
# CATEGORIES = ['jj', 'st', 'dmyh_gh', 'lbj']
# CATEGORIES = ["00_dianjiedianrong", "01_erjiguan", "02_chazuo", "03_dianzu", "04_usb"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--voc_root', help='path of voc root directory(VOC2007)',
                        default='/home/dls1/simple_data/tiaochi/dataset_image/train_0810/3132/VOCdevkit2007/VOC2007',
                        type=str)
    parser.add_argument('-s', '--save_path', help='path of directory to save *.tfrecord',
                        default='/home/dls1/simple_data/tiaochi/dataset_image/train_0810/3132/',
                        type=str)
    args = parser.parse_args()
    return args


def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    depth = float(size.find('depth').text)
    assert depth == 3, 'depth must to be 3'
    image_format = b'jpeg'  # b'jpeg' or b'png'

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        classes_text.append(cls.encode('utf8'))
        if cls not in CATEGORIES:
            continue
        cls_id = CATEGORIES.index(cls) + 1
        classes.append(cls_id)
        xml_box = obj.find('bndbox')
        x_min = float(xml_box.find('xmin').text)
        x_max = float(xml_box.find('xmax').text)
        y_min = float(xml_box.find('ymin').text)
        y_max = float(xml_box.find('ymax').text)
        xmins.append(x_min / width)
        xmaxs.append(x_max / width)
        ymins.append(y_min / height)
        ymaxs.append(y_max / height)

    infos = (height, width, filename, image_format, xmins, xmaxs, ymins, ymaxs, classes_text, classes)
    return infos


def create_tf_example(xml_info, image_root):
    height, width, filename, image_format, xmins, xmaxs, ymins, ymaxs, classes_text, classes = xml_info
    image_full_name = os.path.join(image_root, filename)
    with tf.gfile.GFile(image_full_name, 'rb') as fid:
        encoded_image_data = fid.read()
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def voc_to_tfrecord(save_path, voc_root, split_type):
    xmls_root = os.path.join(voc_root, 'Annotations')
    xmls_file = os.listdir(xmls_root)
    images_root = os.path.join(voc_root, 'JPEGImages')
    split_txt = os.path.join(voc_root, 'ImageSets', 'Main', '%s.txt' % split_type)
    split_txt_list = []
    with open(split_txt, 'r') as file:
        for i in file.readlines():
            split_txt_list.append(i.split('\n')[0])
    save_file_name = os.path.join(save_path, '%s.tfrecord' % split_type)
    writer = tf.python_io.TFRecordWriter(save_file_name)
    for xml_file in xmls_file:
        xml_file_id = xml_file.split('.')[0]
        if xml_file_id in split_txt_list:
            xml_full_path = os.path.join(xmls_root, xml_file)
            xml_info = parse_annotation(xml_full_path)
            tf_example = create_tf_example(xml_info, images_root)
            writer.write(tf_example.SerializeToString())
    writer.close()


def main():
    args = parse_args()
    voc_root = args.voc_root
    save_path = args.save_path
    for split_type in DATASET_SPLIT_LIST:
        voc_to_tfrecord(save_path, voc_root, split_type)


if __name__ == '__main__':
    main()
