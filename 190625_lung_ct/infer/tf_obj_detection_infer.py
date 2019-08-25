import numpy as np
import os
import tensorflow as tf
from PIL import Image
import argparse
import cv2
import sys
import pandas as pd
from time import time

sys.path.append('../lib')
from preprocess import label_select
from csvTools import writeCSV

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir',
                        help='path of image dir to inference,file name format:seriesuid_zslice.xxx',
                        default='/home/dls1/simple_data/tiaochi/dataset_image/train_split/test',
                        type=str)
    parser.add_argument('-g', '--graph_path', help='frozen graph path',
                        default='/home/xuxin/simple_data/tiaochi/dataset_image/train_tfrecord/models/190805_ssd_resnet50_v1_fpn/export/frozen_inference_graph.pb',
                        type=str)
    parser.add_argument('-v', '--voxel', help='voxel origin and spacing',
                        default='/home/dls1/work/tianchi/code/read_data/all_mhd_info.csv', type=str)
    parser.add_argument('-r', '--result', help='path to save result', default=None, type=str)
    parser.add_argument("-lc", "--label_choose", help="choose preliminary and rematch label mapping",
                        type=str, default="rematch", choices=["preliminary", 'rematch'])
    parser.add_argument('-t', '--thresh', help='score threshold', default=0.3, type=float)
    parser.add_argument('-s', '--show', help='show result on image', action="store_true")

    args = parser.parse_args()
    return args


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates
                # to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


# fast speed
def run_inference(detection_graph, images_path, threshold, voxel, image_height, image_width, labels_to_id, show_flag,
                  save_flag):

    images = np.ndarray(shape=(1, image_height, image_width, 3))

    with detection_graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            for i in os.listdir(images_path):
                image_path = os.path.join(images_path, i)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images[0] = image
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: images})
                output_dict['file_name'] = i
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                image_result, break_flag = result_process(output_dict, threshold, voxel, image_height, image_width,
                                                          labels_to_id, show_flag, image_path)
                if break_flag:
                    break
                if save_flag:
                    all_csv_results.extend(image_result)


def result_process(result, threshold, voxel, image_height, image_width, labels_to_id, show_flag, image_path):
    csv_results = []
    break_flag = False

    file_name = result['file_name']
    scores = result['detection_scores']
    classes = result['detection_classes']
    boxes = result['detection_boxes']
    thresh_idx = scores >= threshold
    valid_scores = scores[thresh_idx]
    valid_classes = classes[thresh_idx]
    float_boxes = boxes[thresh_idx]
    float_boxes[:, 0] *= image_height
    float_boxes[:, 2] *= image_height
    float_boxes[:, 1] *= image_width
    float_boxes[:, 3] *= image_width
    # print(file_name, valid_scores, valid_classes, float_boxes)

    mhd_id = file_name.split('_')[0]
    z_slice = file_name.split('.')[0].split('_')[1]
    voxel_origin_x = voxel.loc[int(mhd_id)]['origin_x']
    voxel_origin_y = voxel.loc[int(mhd_id)]['origin_y']
    voxel_origin_z = voxel.loc[int(mhd_id)]['origin_z']
    voxel_spacing_x = voxel.loc[int(mhd_id)]['spacing_x']
    voxel_spacing_y = voxel.loc[int(mhd_id)]['spacing_y']
    voxel_spacing_z = voxel.loc[int(mhd_id)]['spacing_z']

    for i in range(float_boxes.shape[0]):
        box_center_x = (float_boxes[i][1] + float_boxes[i][3]) / 2
        box_center_y = (float_boxes[i][0] + float_boxes[i][2]) / 2
        voxel_x = box_center_x * voxel_spacing_x + voxel_origin_x
        voxel_y = box_center_y * voxel_spacing_y + voxel_origin_y
        voxel_z = int(z_slice) * voxel_spacing_z + voxel_origin_z
        tf_class = valid_classes[i] - 1
        voxel_label = labels_to_id[tf_class]
        score = valid_scores[i]
        # seriesuid,coordX,coordY,coordZ,class,probability
        print(mhd_id, voxel_x, voxel_y, voxel_z, voxel_label, score)
        csv_results.append([mhd_id, voxel_x, voxel_y, voxel_z, voxel_label, score])

    if show_flag:
        valid_boxes = np.int0(float_boxes)
        image = cv2.imread(image_path)
        for i in range(valid_boxes.shape[0]):
            ymin = valid_boxes[i][0]
            xmin = valid_boxes[i][1]
            ymax = valid_boxes[i][2]
            xmax = valid_boxes[i][3]
            show_class = valid_classes[i] - 1
            show_prob = valid_scores[i]
            show_str = "%d %.2f" % (show_class, show_prob)
            cv2.putText(image, show_str, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        cv2.imshow("image", image)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break_flag = True

    return csv_results, break_flag


def main():
    args = parse_args()
    images_path = args.image_dir
    graph_path = args.graph_path
    threshold = args.thresh
    voxel_origin_spacing = args.voxel
    voxel = pd.read_csv(voxel_origin_spacing, sep=",", index_col=0)
    result_path = args.result
    label_choose = args.label_choose
    if result_path:
        save_flag = True
        global all_csv_results
        all_csv_results = []
        header = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'class', 'probability']
        all_csv_results.append(header)
    else:
        save_flag = False
    image_height = 512
    image_width = 512

    label_mapping, labels_to_id = label_select(label_choose)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    run_inference(detection_graph, images_path, threshold, voxel, image_height, image_width, labels_to_id, args.show,
                  save_flag)

    if save_flag:
        writeCSV(result_path, all_csv_results)


if __name__ == '__main__':
    main()
