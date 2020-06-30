#!/bin/bash
TRAIN="/home/xuxin/Desktop/t_s/c"
VAL="/home/xuxin/Desktop/t_s/m"
DST="/home/xuxin/data/sun_classification/log"
MODEL="xception"
EPOCH=40
BATCH=10
LR=1e-3
EPOCH_DROP=10
AUG=0.75
GPU="0"

python train.py -t $TRAIN -v $VAL -m $MODEL -dst $DST -gpu $GPU \
-aug $AUG -e $EPOCH -b $BATCH -pn $BATCH -lr $LR -ed $EPOCH_DROP


