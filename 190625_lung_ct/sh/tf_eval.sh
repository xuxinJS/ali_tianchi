#!/usr/bin/env bash
# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH='/home/dls1/simple_data/tiaochi/log/tf/190821/1235/pipeline15.config'
MODEL_DIR='/home/dls1/simple_data/tiaochi/log/tf/190821/1235/'
python /T3/github_download/tensorflow/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --checkpoint_dir=${MODEL_DIR} \
    --run_once=True
