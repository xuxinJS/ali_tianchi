#!/bin/bash
# train
TRAIN="/home/xuxin/data/sun_classification/data_gen/cut/train"
VAL="/home/xuxin/data/sun_classification/data_gen/cut/val"
TEST="/home/xuxin/data/sun_classification/data_gen/cut/test"
DST="/home/xuxin/data/sun_classification/train_log"
MODEL="xception"
EPOCH=30
BATCH=10
LR=1e-3
EPOCH_DROP=10
AUG=0.75
GPU="0"

TIME=$(date "+%Y%m%d_%H%M")
DST_TIME=$DST/$TIME
python train.py -t $TRAIN -v $VAL \
-m $MODEL -dst $DST_TIME -gpu $GPU \
-aug $AUG -e $EPOCH -b $BATCH -pn $BATCH -lr $LR -ed $EPOCH_DROP

# test
OUTPUT=$DST_TIME/test_log
WEIGHT_DIR=$DST_TIME/model
index=0
for i in `ls ${WEIGHT_DIR}|sort -r`
do
  MODEL_NAME=${WEIGHT_DIR}/$i
  python test.py -t $TEST -m $MODEL -pw $MODEL_NAME -o $OUTPUT -gpu $GPU
  index=`expr $index + 1`
  if [ $index = 2 ]
  then
    break
  fi
done
