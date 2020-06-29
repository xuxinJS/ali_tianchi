#!/bin/bash
TEST="/home/dls1/simple_data/data_0627/continuum/test"
DST="/home/dls1/simple_data/test_log"
MODEL="xception"
WEIGHT="/home/dls1/Desktop/xception_ep23_vloss0.0805.h5"
OUTPUT='/home/dls1/simple_data/test_log'
BATCH=2
GPU="0"

python3 test.py -t $TEST -m $MODEL -pw $WEIGHT  \
-o $OUTPUT -gpu $GPU


