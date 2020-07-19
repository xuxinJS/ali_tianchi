#!/bin/bash
# train
TRAIN="/home/dls1/simple_data/data_gen/clear_concat/con_cv"
VAL="/home/dls1/simple_data/data_gen/train_val_test/cut_concat/val"
TEST="/home/dls1/simple_data/classification/test_concat/con_cv"
DST="/home/dls1/simple_data/train_log"
MODEL="xception"
EPOCH=25
BATCH=12
LR=1e-3
EPOCH_DROP=50
AUG=0.75
GPU="0"

TIME=$(date "+%Y%m%d_%H%M")
DST_TIME=$DST/$TIME
# -v $VAL
python train.py -t $TRAIN \
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
