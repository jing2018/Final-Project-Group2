#!/bin/bash
now=$(date '+%d%m%Y_%H%M%S');
~/caffe/build/tools/caffe train -solver AlexNet_gamma0.95_solver.prototxt  -gpu 3 2>&1 | tee Alexnet_log/Alex_gamma0.95_log-$now.log
~