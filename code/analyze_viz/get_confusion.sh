#!/bin/bash

python confusion.py --proto lenet_deploy.prototxt --model /home/jingsi/ml2_final_project/new_fer2013/models/lenet/lenet_solver_iter_10000.caffemodel --lmdb fer2013_val_lmdb --mean fer2013_mean.binaryproto

