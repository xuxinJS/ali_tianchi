#!/bin/bash
TRAIN="/home/xuxin/data/sun/gen/continuum/train"
VAL="/home/xuxin/data/sun/gen/continuum/val"
DST="/home/xuxin/model/0627"
LOG="/home/xuxin/model/0627/log"
MODEL="xception"

python3 training.py -t $TRAIN -v $VAL -m $MODEL -e 1 -b 8 -dst $DST -log $LOG


