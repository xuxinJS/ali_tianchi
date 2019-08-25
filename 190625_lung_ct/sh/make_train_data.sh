#!/usr/bin/env bash
# From the tensorflow/models/research/ directory
# create voc2007
#python raw_to_voc2007_rgb.py \
#-i /T3/data/train_data/public/tianchi/190625_CT/dataset/split/test \
#-l /T3/data/train_data/public/tianchi/190625_CT/dataset/split/test.csv \
#-o /home/dls1/simple_data/tiaochi/dataset_image/train_0821/test/tfrecord \
#-imin -1000 \
#-imax -300 \
#-lc rematch

#python raw_to_voc2007_rgb.py \
#-i /T3/data/train_data/public/tianchi/190625_CT/dataset/split/train \
#-l /T3/data/train_data/public/tianchi/190625_CT/dataset/split/train.csv \
#-o /home/dls1/simple_data/tiaochi/dataset_image/train_0821/313233 \
#-imin -100 \
#-imax 300 \
#-lc rematch
#
#
## make tfrecord
#python voc_to_tfrecord.py \
#-i /home/dls1/simple_data/tiaochi/dataset_image/train_0821/test/tfrecord/VOCdevkit2007/VOC2007 \
#-s /home/dls1/simple_data/tiaochi/dataset_image/train_0821/test/tfrecord
#
#python voc_to_tfrecord.py \
#-i /home/admin/jupyter/xx/data/train/313233/VOCdevkit2007/VOC2007 \
#-s /home/admin/jupyter/xx/data/tf/313233


