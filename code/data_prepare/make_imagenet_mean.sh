#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/home/jingsi/ml2_final_project/new_fer2013/datasets
DATA=/home/jingsi/ml2_final_project/new_fer2013/datasets
TOOLS=/home/jingsi/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/fer2013_train_lmdb_resize224 \
  $DATA/fer2013_mean_224.binaryproto

echo "Done."
