#!/bin/bash
SRC="/T3/data_gen/fab68/demo_data_split/train"
DST="/T3/data_gen/fab68/model"
LOG="/T3/data_gen/fab68/log"



if [ ! -d "$SRC" ]
then
    echo "Your SRC argument is NOT a folder path."
    exit
fi

if [ ! -e "$DST" ]
then
    rm -rf $DST
    mkdir -p $DST
else
    mkdir -p $DST
fi

:<< BLOCK
for i in `ls $SRC`
do
    python3 training.py -src $SRC"/"$i -dst $DST -m $MODEL -e 20 -b 32

done
BLOCK

#MODEL="efficient_b1"
#python3 training.py -src $SRC -dst $DST -m $MODEL -e 30 -b 10 -flip
MODEL="efficient_b0"
python3 training.py -src $SRC -dst $DST -m $MODEL -e 10 -b 10 -flip -log $LOG


