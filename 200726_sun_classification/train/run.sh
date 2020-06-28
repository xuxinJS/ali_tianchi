#!/bin/bash
TRAIN="/home/xuxin/Desktop/t_s/c"
VAL="/home/xuxin/Desktop/t_s/m"
DST="/home/xuxin/data/sun_classification/log/0628"
MODEL="xception"
EPOCH=30
BATCH=12
LR=1e-3
EPOCH_DROP=10
AUG=0.75

python3 train.py -t $TRAIN -v $VAL -m $MODEL -dst $DST  \
-aug $AUG -e $EPOCH -b $BATCH -lr $LR -ed $EPOCH_DROP


