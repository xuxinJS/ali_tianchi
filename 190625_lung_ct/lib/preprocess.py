import SimpleITK as sitk
import numpy as np
import xml.dom.minidom as minidom
import os
import random

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


def image_process_range(image, image_min, image_max):
    bool_idx_image_min = (image <= image_min)
    image[bool_idx_image_min] = image_min
    bool_idx_image_max = (image >= image_max)
    image[bool_idx_image_max] = image_max
    return image


def label_select(contest_type):
    # 1-结节 2-肺密度增高影 3-肺气肿或肺大泡 5-索条 31-动脉硬化或钙化 32-淋巴结钙化 33-胸膜增厚
    if contest_type == "preliminary":
        label_mapping = {1: 'jj', 5: 'st', 31: 'dmyh_gh', 32: 'lbj'}
        labels_to_id = {0: 1, 1: 5, 2: 31, 3: 32}
    if contest_type == "rematch":
        label_mapping = {1: 'jj', 2: 'fmdzg', 3: 'fqz', 5: 'st', 31: 'dmyh_gh', 32: 'lbj', 33: 'xmzh'}
        labels_to_id = {0: 1, 1: 2, 2: 3, 3: 5, 4: 31, 5: 32, 6: 33}
    return label_mapping, labels_to_id


def add_xml_doc(path, size, pt1, pt2, label, depth, image_type="jpg"):  # size-width,height,depth
    if not os.path.exists(path):
        dom = minidom.getDOMImplementation().createDocument(None, 'annotation', None)
        root = dom.documentElement

        # add folder
        element = dom.createElement('folder')
        element.appendChild(dom.createTextNode('VOC2007'))
        root.appendChild(element)

        # add filename
        element = dom.createElement('filename')
        file_name = os.path.basename(path).replace("xml", image_type)
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
            dom = minidom.parse(path)
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

    with open(path, 'w+', encoding='utf-8') as f:
        dom.writexml(f, addindent='\t', newl='\n', encoding='utf-8')


def splite_data(images_path, save_dir, rate=(0.8, 0.2, 0)):
    image_list = os.listdir(images_path)
    training_list = random.sample(image_list, int(len(image_list) * rate[0]))
    image_set = set(image_list)
    training_set = set(training_list)
    remain_set = image_set - training_set
    val_list = random.sample(list(remain_set), int(len(image_list) * rate[1]))
    val_set = set(val_list)
    test_list = list(remain_set - val_set)

    with open(os.path.join(save_dir, "test.txt"), "w+") as f:
        for item in test_list:
            f.write(item.split(".")[0] + "\n")
    with open(os.path.join(save_dir, "val.txt"), "w+") as f:
        for item in val_list:
            f.write(item.split(".")[0] + "\n")
    with open(os.path.join(save_dir, "train.txt"), "w+") as f:
        for item in training_list:
            f.write(item.split(".")[0] + "\n")
    with open(os.path.join(save_dir, "trainval.txt"), "w+") as f:
        for item in (training_list + val_list):
            f.write(item.split(".")[0] + "\n")
