#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/home/zyh/PycharmProjects/baidu_dog/caffe
DATA=/home/zyh/PycharmProjects/baidu_dog/caffe
TOOLS=/home/zyh/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/train_crop_lmdb \
  $DATA/mean_crop.binaryproto

echo "Done."
