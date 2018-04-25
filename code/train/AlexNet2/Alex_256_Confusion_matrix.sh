#!/bin/bash

python ../new_fer2013/datasets/confusion_matrix.py --proto datasets/AlexNet_deploy.prototxt --model models/AlexNet/AlexNet_256_size_solver_iter_50000.caffemodel --mean datasets/fer2013_mean.binaryproto --lmdb datasets/fer2013_val_lmdb