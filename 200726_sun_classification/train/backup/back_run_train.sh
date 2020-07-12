#!/bin/bash
TRAIN="/home/xuxin/Desktop/continuum/train"
DST="/home/xuxin/Desktop/log"
MODEL="mobilenet"
EPOCH=10
BATCH=3
LR=1e-3
EPOCH_DROP=10
AUG=0.75
CUT=0.75
GPU="0"
#-v $VAL
python train.py -t $TRAIN  -m $MODEL -dst $DST -gpu $GPU -aug $AUG -cut $CUT \
-e $EPOCH -b $BATCH -pn $BATCH -lr $LR -ed $EPOCH_DROP


