#!/usr/bin/env sh
set -e

/home/zyh/caffe-face-caffe-face/build/tools/caffe train --solver=/home/zyh/PycharmProjects/baidu_dog/caffe/Resnet50/ResNet50-solver.prototxt   --gpu 0 2>&1 | tee /home/zyh/PycharmProjects/baidu_dog/caffe/Resnet50/train.log$@
