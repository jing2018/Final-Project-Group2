#!/bin/bash
now=$(date '+%d%m%Y_%H%M%S');
~/caffe/build/tools/caffe train -solver Resnet_pretrain_solver.prototxt -snapshot ../models/resnet/Resnet_pretrain_solver_iter_150000.solverstate -gpu 0 2>&1 | tee logs/log-$now.log
