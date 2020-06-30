#!/bin/bash
TEST="/home/dls1/simple_data/data_0627/continuum/test"
MODEL="xception"
WEIGHT="/home/dls1/Desktop/xception_ep23_vloss0.0805.h5"
OUTPUT='/home/dls1/simple_data/test_log'
GPU="0"

python test.py -t $TEST -m $MODEL -pw $WEIGHT -o $OUTPUT -gpu $GPU


