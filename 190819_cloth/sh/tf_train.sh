#!/usr/bin/env bash
#From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH='/T3/data/train_data/public/tianchi/190825_cloth/log/ssd_mobilenet_v1_fpn.config'
MODEL_DIR='/T3/data/train_data/public/tianchi/190825_cloth/log/190825'
python /T3/github_download/tensorflow/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --alsologtostderr