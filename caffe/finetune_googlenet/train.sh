#!/usr/bin/env sh
set -e

/home/zyh/caffe//build/tools/caffe train --solver=/home/zyh/PycharmProjects/baidu_dog/caffe/finetune_googlenet/solver.prototxt  --gpu 0 2>&1 | tee /home/zyh/PycharmProjects/baidu_dog/caffe/finetune_googlenet/train.log$@
