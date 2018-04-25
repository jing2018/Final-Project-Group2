#!/bin/bash

~/caffe/tools/extra/parse_log.sh Alex_450000_log-24042018_002909.log
gnuplot 1_train_val_loss.gnuplot 
gnuplot 2_test_acc.gnuplot 
gnuplot 3_learningrate.gnuplot
