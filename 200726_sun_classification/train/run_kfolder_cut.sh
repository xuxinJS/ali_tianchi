#!/bin/bash
# train
TRAIN="/home/dls1/simple_data/data_gen/0705_con/train"
TEST="/home/dls1/simple_data/data_gen/0705_con/test"
DST="/home/dls1/simple_data/train_log"
MODEL="inception_resnetv2"
EPOCH=40
BATCH=10
LR=1e-3
EPOCH_DROP=15
AUG=0.75
CUT=1
GPU="0"

TIME=$(date "+%Y%m%d_%H%M")
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
