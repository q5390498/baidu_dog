#!/usr/bin/env sh
set -e

/home/zyh/caffe/build/tools/caffe train --solver=/home/zyh/PycharmProjects/baidu_dog/caffe/xception/solver.prototxt   --gpu 0 2>&1 | tee /home/zyh/PycharmProjects/baidu_dog/caffe/xception/train.log$@
