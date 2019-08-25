# coding=utf-8
import json
import cv2
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import xml.dom.minidom as minidom
import random
import tensorflow as tf
import os

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from object_detection.utils import dataset_util


def label_select(contest_type):
    if contest_type == "preliminary":
        label_mapping = {
            u'无疵点': 0,
            u'破洞': 1,
            u'水渍': 2,
            u'油渍': 2,
            u'污渍': 2,
            u'三丝': 3,
            u'结头': 4,
            u'花板跳': 5,
            u'百脚': 6,
            u'毛粒': 7,
            u'粗经': 8,
            u'松经': 9,
            u'断经': 10,
            u'吊经': 11,
            u'粗维': 12,
            u'纬缩': 13,
            u'浆斑': 14,
            u'整经结': 15,
            u'星跳': 16,
            u'跳花': 16,
            u'断氨纶': 17,
            u'稀密档': 18,
            u'浪纹档': 18,
            u'色差档': 18,
            u'磨痕': 19,
            u'轧痕': 19,
            u'修痕': 19,
            u'烧毛痕': 19,
            u'死皱': 20,
            u'云织': 20,
            u'双纬': 20,
            u'双经': 20,
            u'跳纱': 20,
            u'筘路': 20,
            u'纬纱不良': 20,
        }
    if contest_type == "rematch":
        label_mapping = {1: 'jj', 2: 'fmdzg', 3: 'fqz', 5: 'st', 31: 'dmyh_gh', 32: 'lbj', 33: 'xmzh'}
    return label_mapping


# opencv 显示中文
def cv_show_chinese(cv_bgr, text, left_up_point, font_size, font_path, color):
    cv2img = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    draw.text(left_up_point, text, color, font=font)
    image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return image


# 处理bbox：min向下取整，max向上取整，负值过滤，超边界过滤
def bbox_process(bbox, height, width):
    xmin = math.floor(float(bbox[0]))
    ymin = math.floor(float(bbox[1]))
    xmax = math.ceil(float(bbox[2]))
    ymax = math.ceil(float(bbox[3]))
    xmin = xmin if xmin > 0 else 0
    ymin = ymin if ymin > 0 else 0
    xmax = xmax if xmax < width else (width - 1)
    ymax = ymax if ymax < height else (height - 1)
    return [xmin, ymin, xmax, ymax]


# 将结果汇总，生成新的label
def summary_save_label(input_label, output_label, image_folder, show_flag=False, ture_type_path=None):
    json_file = open(input_label)
    json_dicts = json.load(json_file)
    json_file.close()
    # todo label计数  核对label是否正确
    file_name_set = set()
    new_label_list = []
    for i in json_dicts:
        file_name_set.add(i['name'])
    for i in file_name_set:
        tmp_label_dict = {}
        tmp_label_dict['name'] = i
        tmp_defect_list = []

        image_path = os.path.join(image_folder, i)
        image = cv2.imread(image_path)
        image_height = image.shape[0]
        image_width = image.shape[1]

        for j in json_dicts:
            if j['name'] == i:
                tmp_dict = {}
                tmp_dict['defect_name'] = j['defect_name']
                tmp_dict['bbox'] = bbox_process(j['bbox'], image_height, image_width)
                tmp_defect_list.append(tmp_dict)
        tmp_label_dict['defects'] = tmp_defect_list
        new_label_list.append(tmp_label_dict)
    # save new label
    with open(output_label, 'w') as fp:
        json.dump(new_label_list, fp, indent=4, separators=(',', ': '))


# 注意,使用处理过的label
def show_image_with_label(image_folder, new_label, ture_type_path):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    json_file = open(new_label)
    json_dicts = json.load(json_file)
    json_file.close()

    label_to_id = label_select("preliminary")

    for i in json_dicts:
        file_name = os.path.join(image_folder, i['name'])
        image = cv2.imread(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        defects = i['defects']
        for j in defects:
            defect_name = j['defect_name']
            bbox = j['bbox']
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            image = cv_show_chinese(image, defect_name, (xmin, ymin), 20, ture_type_path, (0, 0, 255))
            label_id = str(label_to_id[defect_name])
            cv2.putText(image, label_id, (xmax, ymax), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

        cv2.imshow('image', image)
        key = cv2.waitKey(0) & 0xff
        if key == ord('q'):
            break


def add_xml_doc(anno_path, size, pt1, pt2, label, depth, image_type="jpg"):
    # size-height,width,depth
    # pt1 (xmin,ymin)
    # pt1 (xmax,ymax)
    if not os.path.exists(anno_path):
        dom = minidom.getDOMImplementation().createDocument(None, 'annotation', None)
        root = dom.documentElement

        # add folder
        element = dom.createElement('folder')
        element.appendChild(dom.createTextNode('VOC2007'))
        root.appendChild(element)

        # add filename
        element = dom.createElement('filename')
        file_name = os.path.basename(anno_path).replace("xml", image_type)
        element.appendChild(dom.createTextNode(file_name))
        root.appendChild(element)

        # add source
        element = dom.createElement('source')
        child = dom.createElement("database")
        child.appendChild(dom.createTextNode("The VOC2007 Database"))
        element.appendChild(child)
        child = dom.createElement("annotation")
        child.appendChild(dom.createTextNode("PASCAL VOC2007"))
        element.appendChild(child)
        child = dom.createElement("image")
        child.appendChild(dom.createTextNode("flickr"))
        element.appendChild(child)
        child = dom.createElement("flickrid")
        child.appendChild(dom.createTextNode("326445091"))
        element.appendChild(child)
        root.appendChild(element)

        element = dom.createElement('owner')
        child = dom.createElement("flickrid")
        child.appendChild(dom.createTextNode("TIANCHI"))
        element.appendChild(child)
        child = dom.createElement("name")
        child.appendChild(dom.createTextNode("?"))
        element.appendChild(child)
        root.appendChild(element)

        element = dom.createElement('size')
        child = dom.createElement("width")
        child.appendChild(dom.createTextNode(str(size[1])))
        element.appendChild(child)
        child = dom.createElement("height")
        child.appendChild(dom.createTextNode(str(size[0])))
        element.appendChild(child)
        child = dom.createElement("depth")
        child.appendChild(dom.createTextNode(str(depth)))
        element.appendChild(child)
        root.appendChild(element)

        element = dom.createElement('segmented')
        element.appendChild(dom.createTextNode("0"))
        root.appendChild(element)

        if label:
            element = dom.createElement('object')
            child = dom.createElement("name")
            child.appendChild(dom.createTextNode(label))
            element.appendChild(child)
            child = dom.createElement("pose")
            child.appendChild(dom.createTextNode("Unspecified"))
            element.appendChild(child)
            child = dom.createElement("truncated")
            child.appendChild(dom.createTextNode("0"))
            element.appendChild(child)
            child = dom.createElement("difficult")
            child.appendChild(dom.createTextNode("0"))
            element.appendChild(child)

            child = dom.createElement("bndbox")
            grand_child = dom.createElement("xmin")
            grand_child.appendChild(dom.createTextNode(str(pt1[0])))
            child.appendChild(grand_child)
            grand_child = dom.createElement("ymin")
            grand_child.appendChild(dom.createTextNode(str(pt1[1])))
            child.appendChild(grand_child)
            grand_child = dom.createElement("xmax")
            grand_child.appendChild(dom.createTextNode(str(pt2[0])))
            child.appendChild(grand_child)
            grand_child = dom.createElement("ymax")
            grand_child.appendChild(dom.createTextNode(str(pt2[1])))
            child.appendChild(grand_child)
            element.appendChild(child)
            root.appendChild(element)
    else:
        if label:
            dom = minidom.parse(anno_path)
            root = dom.documentElement
            names = root.getElementsByTagName('annotation')

            element = dom.createElement('object')
            child = dom.createElement("name")
            child.appendChild(dom.createTextNode(label))
            element.appendChild(child)
            child = dom.createElement("pose")
            child.appendChild(dom.createTextNode("Unspecified"))
            element.appendChild(child)
            child = dom.createElement("truncated")
            child.appendChild(dom.createTextNode("0"))
            element.appendChild(child)
            child = dom.createElement("difficult")
            child.appendChild(dom.createTextNode("0"))
            element.appendChild(child)

            child = dom.createElement("bndbox")
            grand_child = dom.createElement("xmin")
            grand_child.appendChild(dom.createTextNode(str(pt1[0])))
            child.appendChild(grand_child)
            grand_child = dom.createElement("ymin")
            grand_child.appendChild(dom.createTextNode(str(pt1[1])))
            child.appendChild(grand_child)
            grand_child = dom.createElement("xmax")
            grand_child.appendChild(dom.createTextNode(str(pt2[0])))
            child.appendChild(grand_child)
            grand_child = dom.createElement("ymax")
            grand_child.appendChild(dom.createTextNode(str(pt2[1])))
            child.appendChild(grand_child)
            element.appendChild(child)
            root.appendChild(element)

    with open(anno_path, 'w+', encoding='utf-8') as f:
        dom.writexml(f, addindent='\t', newl='\n', encoding='utf-8')


def splite_data(images_path, save_dir, rate=(0.7, 0.1, 0.2)):
    image_list = os.listdir(images_path)
    image_set = set(image_list)

    training_list = random.sample(image_list, int(len(image_list) * rate[0]))
    training_set = set(training_list)

    remain_set = image_set - training_set
    val_list = random.sample(list(remain_set), int(len(image_list) * rate[1]))
    val_set = set(val_list)

    test_list = list(remain_set - val_set)
    if len(test_list) > 1:
        with open(os.path.join(save_dir, "test.txt"), "w+") as f:
            for item in test_list:
                f.write(item.split(".")[0] + "\n")
    else:
        val_set = remain_set
    with open(os.path.join(save_dir, "val.txt"), "w+") as f:
        for item in val_list:
            f.write(item.split(".")[0] + "\n")
    with open(os.path.join(save_dir, "train.txt"), "w+") as f:
        for item in training_list:
            f.write(item.split(".")[0] + "\n")
    with open(os.path.join(save_dir, "trainval.txt"), "w+") as f:
        for item in (training_list + val_list):
            f.write(item.split(".")[0] + "\n")


def save_voc_anno(image_folder, new_label, out_folder):
    # image_folder:包含有缺陷和无缺陷的照片
    # new_label:汇总过的结果
    # out_folder：创建完成之后手动将image_folder的照片移动到JPEGImages
    JPEGImages = os.path.join(out_folder, "VOC2007", "JPEGImages")
    Annatations = os.path.join(out_folder, "VOC2007", "Annotations")
    Main = os.path.join(out_folder, "VOC2007", "ImageSets", "Main")
    for save_path in [JPEGImages, Annatations, Main]:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    # label
    json_file = open(new_label)
    json_dicts = json.load(json_file)
    json_file.close()

    # image
    image_list = os.listdir(image_folder)
    all_image_set = set(image_list)
    defect_image_set = set()

    for i in json_dicts:
        relative_file_name = i['name']
        full_file_name = os.path.join(image_folder, relative_file_name)
        full_anno_name = os.path.join(Annatations, relative_file_name.split('.')[0] + '.xml')
        print(full_anno_name)
        defect_image_set.add(relative_file_name)
        image = cv2.imread(full_file_name)

        for j in i['defects']:
            defect_name = j['defect_name']
            bbox = j['bbox']
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            add_xml_doc(full_anno_name, image.shape, (xmin, ymin), (xmax, ymax), defect_name, 3)

    no_defect_image_set = all_image_set - defect_image_set
    for i in no_defect_image_set:
        full_file_name = os.path.join(image_folder, i)
        full_anno_name = os.path.join(Annatations, i.split('.')[0] + '.xml')
        image = cv2.imread(full_file_name)
        add_xml_doc(full_anno_name, image.shape, None, None, None, 3)

    splite_data(JPEGImages, Main)
    # print('defect_image_set', len(defect_image_set))
    # print('no_defect_image_set', len(no_defect_image_set))


##############create tfrecord ################
def parse_annotation(annotation_path, cls_dict):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    depth = int(size.find('depth').text)
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
        # if cls not in CATEGORIES:
        #     continue
        cls_id = cls_dict[cls] + 1
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


def voc_to_tfrecord(voc_root, save_folder, cls_dict, split_type):
    xmls_root = os.path.join(voc_root, 'Annotations')
    xmls_file = os.listdir(xmls_root)
    images_root = os.path.join(voc_root, 'JPEGImages')
    split_txt = os.path.join(voc_root, 'ImageSets', 'Main', '%s.txt' % split_type)
    split_txt_list = []
    with open(split_txt, 'r') as file:
        for i in file.readlines():
            split_txt_list.append(i.split('\n')[0])
    save_file_name = os.path.join(save_folder, '%s.tfrecord' % split_type)
    writer = tf.python_io.TFRecordWriter(save_file_name)
    for xml_file in xmls_file:
        xml_file_id = xml_file.split('.')[0]
        if xml_file_id in split_txt_list:
            xml_full_path = os.path.join(xmls_root, xml_file)
            xml_info = parse_annotation(xml_full_path, cls_dict)
            tf_example = create_tf_example(xml_info, images_root)
            writer.write(tf_example.SerializeToString())
    writer.close()


def create_tf_record(voc_root, save_folder):
    # voc_root:path of voc root directory(VOC2007)
    # save_folder:path of directory to save *.tfrecord
    DATASET_SPLIT_LIST = ['train', 'val', 'test']
    # DATASET_SPLIT_LIST = ['val']
    CATEGORIES_DICT = label_select('preliminary')
    for split_type in DATASET_SPLIT_LIST:
        voc_to_tfrecord(voc_root, save_folder, CATEGORIES_DICT, split_type)


if __name__ == '__main__':
    image_folder = '/T3/data/train_data/public/tianchi/190825_cloth/guangdong1_round1_train1_20190818/all_images'
    json_file_path = '/T3/data/train_data/public/tianchi/190825_cloth/guangdong1_round1_train1_20190818/Annotations/anno_train.json'
    new_json_file_path = '/T3/data/train_data/public/tianchi/190825_cloth/guangdong1_round1_train1_20190818/Annotations/new.json'
    true_type_font_path = '/home/dls1/Desktop/hwxh.ttf'
    voc_out_folder = '/T3/data/train_data/public/tianchi/190825_cloth/Dataset/190818'
    voc_root = '/T3/data/train_data/public/tianchi/190825_cloth/Dataset/190818/VOC2007'
    # summary_save_label(json_file_path, new_json_file_path, image_folder)
    # show_image_with_label(image_folder, new_json_file_path, true_type_font_path)
    # save_voc_anno(image_folder, new_json_file_path, voc_out_folder)
    # create_tf_record(voc_root, voc_out_folder)
    ids = [label_select('preliminary')]

    with open('/home/dls1/xx.json', 'w') as fp:
        json.dump(ids, fp, indent=4, separators=(',', ': '))