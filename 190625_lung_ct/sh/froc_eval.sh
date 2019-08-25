#!/usr/bin/env bash
python noduleCADEvaluationLUNA16.py  \
/T3/data/train_data/public/tianchi/190625_CT/dataset/split/test.csv \
/home/dls1/work/tianchi/code/infer/ssd_resnet50_1235.csv \
ssd_resnet50_15.txt \
out
