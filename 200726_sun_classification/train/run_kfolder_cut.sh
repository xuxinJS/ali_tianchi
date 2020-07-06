#!/bin/bash
# train
TRAIN="/home/xuxin/Desktop/continuum/train"
TEST="/home/xuxin/Desktop/continuum/val"
DST="/home/xuxin/Desktop/log"
MODEL="xception"
EPOCH=30
BATCH=10
LR=1e-3
EPOCH_DROP=10
AUG=0.75
CUT=0.75
GPU="0"

TIME=$(date "+%m%d_%H%M")
DST_TIME=$DST/$TIME
python train_kfolder_cut.py -t $TRAIN  \
-m $MODEL -dst $DST_TIME -gpu $GPU -aug $AUG -cut $CUT \
-e $EPOCH -b $BATCH -pn $BATCH -lr $LR -ed $EPOCH_DROP

# test
OUTPUT=$DST_TIME/test_log
WEIGHT_DIR=$DST_TIME/model
index=0
for i in `ls ${WEIGHT_DIR}|sort -r`
do
  MODEL_NAME=${WEIGHT_DIR}/$i
  python test_cut.py -t $TEST -m $MODEL -pw $MODEL_NAME -o $OUTPUT -gpu $GPU
  index=`expr $index + 1`
  if [ $index = 2 ]
  then
    break
  fi
done
