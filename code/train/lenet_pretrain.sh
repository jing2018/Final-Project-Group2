#!/bin/bash
now=$(date '+%d%m%Y_%H%M%S');
~/caffe/build/tools/caffe train -solver lenet_solver.prototxt  -gpu 0 2>&1 | tee lenetlogs/log-lenet$now.log
