#!/bin/bash
TRAIN="/home/dls1/simple_data/gen_0627/continuum/train"
VAL="/home/dls1/simple_data/gen_0627/continuum/val"
DST="/home/dls1/simple_data/gen_0627/con_log"
LOG="/home/dls1/simple_data/gen_0627/con_log"
MODEL="xception"
EPOCH=10
BATCH=12


python3 training.py -t $TRAIN -v $VAL -m $MODEL -e $EPOCH -b $BATCH -dst $DST -log $LOG


