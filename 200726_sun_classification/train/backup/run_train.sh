#!/bin/bash
TRAIN="/home/dls1/simple_data/data_gen/0702_con/train"
VAL="/home/dls1/simple_data/data_gen/0702_con/val"
DST="/home/dls1/simple_data/train_log"
MODEL="xception"
EPOCH=30
BATCH=10
LR=1e-3
EPOCH_DROP=10
AUG=0.75
GPU="0"

python train.py -t $TRAIN -v $VAL -m $MODEL -dst $DST -gpu $GPU \
-aug $AUG -e $EPOCH -b $BATCH -pn $BATCH -lr $LR -ed $EPOCH_DROP


