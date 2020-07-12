#!/bin/bash
WEIGHT_DIR="/home/xuxin/Desktop/train_log/20200705_1116/model"

index=0
for i in `ls ${WEIGHT_DIR}|sort -r`
do
  index=`expr $index + 1`
  echo $index
  echo ${WEIGHT_DIR}/$i
  if [ $index = 2 ]
  then
    break
  fi
done

