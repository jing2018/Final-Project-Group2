#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

EXAMPLE=/home/jingsi/ml2_final_project/new_fer2013/datasets
DATA=/home/jingsi/ml2_final_project/new_fer2013/datasets
TOOLS=/home/jingsi/caffe/build/tools

TRAIN_DATA_ROOT=/home/jingsi/ml2_final_project/new_fer2013/datasets/train/
VAL_DATA_ROOT=/home/jingsi/ml2_final_project/new_fer2013/datasets/val/
TEST_DATA_ROOT=/home/jingsi/ml2_final_project/new_fer2013/datasets/test/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=false
if $RESIZE; then
  RESIZE_HEIGHT=224
  RESIZE_WIDTH=224
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

if [ ! -d "$TEST_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $TEST_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --gray=true \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $EXAMPLE/fer2013_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --gray=true \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    $EXAMPLE/fer2013_val_lmdb

echo "Creating test lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --gray=true \
    $TEST_DATA_ROOT \
    $DATA/test.txt \
    $EXAMPLE/fer2013_test_lmdb


echo "Done."
