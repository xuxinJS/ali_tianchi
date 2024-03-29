#!/bin/bash
# train
TRAIN="/raid/myshare/xuxin/sun/data/gen/all_balance_cut"
VAL="/raid/myshare/xuxin/sun/data/gen/0703_con_cut/val"
TEST="/raid/myshare/xuxin/sun/data/init/test_cut"
DST="/raid/myshare/xuxin/sun/log/con_cut"
MODEL="xception"
EPOCH=20
BATCH=12
LR=1e-3
EPOCH_DROP=30
AUG=0.8
GPU="0"

TIME=$(date "+%Y%m%d_%H%M")
DST_TIME=$DST/$TIME
#-v $VAL
python train_concat.py -t $TRAIN \
-m $MODEL -dst $DST_TIME -gpu $GPU \
-aug $AUG -e $EPOCH -b $BATCH -pn $BATCH -lr $LR -ed $EPOCH_DROP

# test
OUTPUT=$DST_TIME/test_log
WEIGHT_DIR=$DST_TIME/model
index=0
for i in `ls ${WEIGHT_DIR}|sort -r`
do
  MODEL_NAME=${WEIGHT_DIR}/$i
  python test_concat.py -t $TEST -m $MODEL -pw $MODEL_NAME -o $OUTPUT -gpu $GPU
  index=`expr $index + 1`
  if [ $index = 2 ]
  then
    break
  fi
done
