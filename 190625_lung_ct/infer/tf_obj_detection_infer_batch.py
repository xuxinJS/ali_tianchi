import numpy as np
import os
import tensorflow as tf
from PIL import Image
import argparse
import cv2
import sys
from time import time


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir',
                        help='path of image dir to inference,file name format:seriesuid_zslice.xxx',
                        default='image',
                        type=str)
    parser.add_argument('-g', '--graph_path', help='frozen graph path',
                        default='frozen_inference_graph.pb',
                        type=str)
    args = parser.parse_args()
    return args


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(images_path, graph):
    batch_size = 10
    images = np.ndarray(shape=(batch_size, 512, 512, 3))
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
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
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
            start = time()
            for index, i in enumerate(os.listdir(images_path)):
                if index % batch_size == batch_size - 1:
                    image_path = os.path.join(images_path, i)
                    image = cv2.imread(image_path)
                    images[index] = image
                    output_dict = sess.run(tensor_dict,
                                           feed_dict={image_tensor: images})
                    # all outputs are float32 numpy arrays, so convert types as appropriate
                    print(output_dict['num_detections'])
                    output_dict['num_detections'] = int(output_dict['num_detections'][0])
                    output_dict['detection_classes'] = output_dict[
                        'detection_classes'][0].astype(np.uint8)
                    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                    output_dict['detection_scores'] = output_dict['detection_scores'][0]
                    if 'detection_masks' in output_dict:
                        output_dict['detection_masks'] = output_dict['detection_masks'][0]
                        print(result)
                        # break
                else:
                    image_path = os.path.join(images_path, i)
                    image = cv2.imread(image_path)
                    images[index % batch_size] = image

            print('infer time:%f ms' %((time()-start)*1000))
    return output_dict




def main():
    args = parse_args()
    images_path = args.image_dir
    graph_path = args.graph_path
    image_height = 512
    image_width = 512

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    result = run_inference_for_single_image(images_path, detection_graph)




if __name__ == '__main__':
    main()