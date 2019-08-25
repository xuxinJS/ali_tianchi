#!/usr/bin/env bash
python tf_obj_detection_infer.py \
-i /home/dls1/simple_data/tiaochi/dataset_image/test_n750_n250 \
-g /home/dls1/simple_data/tiaochi/log/tf/190811/15/export/frozen_inference_graph.pb \
-v /home/dls1/work/tianchi/code/read_data/all_mhd_info.csv \
-lc rematch \
-r /home/dls1/simple_data/tiaochi/log/tf/190811/ssd_resnet50_15.csv
