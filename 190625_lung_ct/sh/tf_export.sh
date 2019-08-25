#!/usr/bin/env bash
# From tensorflow/models/research/
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=/home/xuxin/simple_data/tiaochi/dataset_image/train_tfrecord/models/190805_ssd_resnet50_v1_fpn512_512/pipeline.config
TRAINED_CKPT_PREFIX=/home/xuxin/simple_data/tiaochi/dataset_image/train_tfrecord/models/190805_ssd_resnet50_v1_fpn512_512/model.ckpt-120000
EXPORT_DIR=/home/xuxin/simple_data/tiaochi/dataset_image/train_tfrecord/models/190805_ssd_resnet50_v1_fpn512_512/export
python /HHD/xuxin/github_download/tensorflow/models/research/object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
