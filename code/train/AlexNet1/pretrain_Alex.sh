#!/bin/bash
now=$(date '+%d%m%Y_%H%M%S');
~/caffe/build/tools/caffe train -solver AlexNet_solver.prototxt  -gpu 0 2>&1 | tee Alexnet_log/Alex_log-$now.log
~