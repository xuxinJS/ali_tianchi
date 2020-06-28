#!/bin/bash
TRAIN="/home/xuxin/Desktop/t_s/c"
VAL="/home/xuxin/Desktop/t_s/m"
DST="/home/xuxin/data/sun_classification/log"
MODEL="mobilenet"
EPOCH=10
BATCH=2
LR=1e-3
EPOCH_DROP=10
AUG=0.75

python3 train.py -t $TRAIN -v $VAL -m $MODEL -dst $DST  \
-aug $AUG -e $EPOCH -b $BATCH -pn $BATCH -lr $LR -ed $EPOCH_DROP


