#!/usr/bin/env sh
set -e

/home/zyh/caffe//build/tools/caffe train --solver=/home/zyh/PycharmProjects/baidu_dog/caffe/finetune_from_bvlc/solver.prototxt --weights /home/zyh/PycharmProjects/baidu_dog/bvlc_googlenet.caffemodel --gpu 0 2>&1 | tee /home/zyh/PycharmProjects/baidu_dog/caffe/finetune_from_bvlc/train.log$@
