#!/bin/bash

~/caffe/tools/extra/parse_log.sh Alex_256_log-24042018_002923.log
gnuplot 1_train_val_loss.gnuplot 
gnuplot 2_test_acc.gnuplot 
gnuplot 3_learningrate.gnuplot
