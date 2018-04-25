#!/bin/bash

~/caffe/tools/extra/parse_log.sh log-lenet.log
gnuplot 1_train_val_loss.gnuplot 
gnuplot 2_test_acc.gnuplot 
gnuplot 3_learningrate.gnuplot
