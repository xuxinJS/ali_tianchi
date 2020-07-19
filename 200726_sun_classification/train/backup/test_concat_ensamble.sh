#!/bin/bash
TEST="/home/dls1/simple_data/classification/test_cut"
OUTPUT="/home/dls1/Desktop/20200716_log"
MODEL1="xception"
WEIGHT1="/home/dls1/Desktop/xception_ep20_loss0.0699.h5"
MODEL2="inception_resnetv2"
WEIGHT2="/home/dls1/Desktop/inception_resnetv2_ep20_loss0.0782.h5"
GPU="0"

python test_concat_ensamble.py -t $TEST -m1 $MODEL1 -pw1 $WEIGHT1 \
-m2 $MODEL2 -pw2 $WEIGHT2 -o $OUTPUT -gpu $GPU

