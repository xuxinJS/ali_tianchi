#!/usr/bin/env bash
##make dataset
#cd ../read_data
#python raw_to_voc2007_rgb.py \
#-i /T3/data/train_data/public/tianchi/190625_CT/dataset/split/train \
#-l /T3/data/train_data/public/tianchi/190625_CT/dataset/split/train.csv \
#-o /home/dls1/simple_data/tiaochi/dataset_image/train_0821/1235 \
#-imin -1000 \
#-imax -300 \
#-lc rematch

# make tfrecord
#python voc_to_tfrecord.py \
#-i /home/dls1/simple_data/tiaochi/dataset_image/train_0821/1235/VOCdevkit2007/VOC2007 \
#-s /home/dls1/simple_data/tiaochi/dataset_image/train_0821/1235

#train
cd /T3/github_download/tensorflow/models/research
PIPELINE_CONFIG_PATH='/home/dls1/simple_data/tiaochi/log/tf/190821/1235_3/pipeline15.config'
MODEL_DIR='/home/dls1/simple_data/tiaochi/log/tf/190821/1235_3'
python /T3/github_download/tensorflow/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --alsologtostderr

#sleep 5s
##export
#INPUT_TYPE=image_tensor
#PIPELINE_CONFIG_PATH='/home/dls1/simple_data/tiaochi/log/tf/190821/1235/pipeline15.config'
#TRAINED_CKPT_PREFIX=/home/dls1/simple_data/tiaochi/log/tf/190821/1235/model.ckpt-12831
#EXPORT_DIR=/home/dls1/simple_data/tiaochi/log/tf/190821/1235/export
#python /T3/github_download/tensorflow/models/research/object_detection/export_inference_graph.py \
#    --input_type=${INPUT_TYPE} \
#    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
#    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
#    --output_directory=${EXPORT_DIR}
#sleep 5s
##tf eval
#python /T3/github_download/tensorflow/models/research/object_detection/model_main.py \
#    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
#    --model_dir=${MODEL_DIR} \
#    --checkpoint_dir=${MODEL_DIR} \
#    --run_once=True > /home/dls1/simple_data/tiaochi/log/tf/190811/coco_ssd_resnet50_15.txt
#
#infer
#cd /home/dls1/work/tianchi/code/infer
#python tf_obj_detection_infer.py \
#-i /home/dls1/simple_data/tiaochi/dataset_image/train_0821/test/1235 \
#-g /home/dls1/simple_data/tiaochi/log/tf/190821/1235/export/frozen_inference_graph.pb \
#-v /home/dls1/work/tianchi/code/read_data/all_mhd_info.csv \
#-lc rematch \
#-r /home/dls1/work/tianchi/code/infer/ssd_resnet50_1235.csv

