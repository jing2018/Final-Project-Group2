#!/bin/bash
now=$(date '+%d%m%Y_%H%M%S');
~/caffe/build/tools/caffe train -solver AlexNet_256_size_solver.prototxt  -gpu 2 2>&1 | tee Alexnet_log/Alex_256_log-$now.log
~