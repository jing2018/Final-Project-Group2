#!/bin/bash
now=$(date '+%d%m%Y_%H%M%S');
~/caffe/build/tools/caffe train -solver Resnet_pretrain_solver.prototxt -gpu 0 2>&1 | tee logs/log-$now.log
