#!/bin/bash

ls train/0 | sed "s:^:0/:" | sed "s:$: 0:" >> train.txt
ls train/1 | sed "s:^:1/:" | sed "s:$: 1:" >> train.txt
ls train/2 | sed "s:^:2/:" | sed "s:$: 2:" >> train.txt
ls train/3 | sed "s:^:3/:" | sed "s:$: 3:" >> train.txt
ls train/4 | sed "s:^:4/:" | sed "s:$: 4:" >> train.txt
ls train/5 | sed "s:^:5/:" | sed "s:$: 5:" >> train.txt
ls train/6 | sed "s:^:6/:" | sed "s:$: 6:" >> train.txt


ls val/0 | sed "s:^:0/:" | sed "s:$: 0:" >> val.txt
ls val/1 | sed "s:^:1/:" | sed "s:$: 1:" >> val.txt
ls val/2 | sed "s:^:2/:" | sed "s:$: 2:" >> val.txt
ls val/3 | sed "s:^:3/:" | sed "s:$: 3:" >> val.txt
ls val/4 | sed "s:^:4/:" | sed "s:$: 4:" >> val.txt
ls val/5 | sed "s:^:5/:" | sed "s:$: 5:" >> val.txt
ls val/6 | sed "s:^:6/:" | sed "s:$: 6:" >> val.txt


ls test/0 | sed "s:^:0/:" | sed "s:$: 0:" >> test.txt
ls test/1 | sed "s:^:1/:" | sed "s:$: 1:" >> test.txt
ls test/2 | sed "s:^:2/:" | sed "s:$: 2:" >> test.txt
ls test/3 | sed "s:^:3/:" | sed "s:$: 3:" >> test.txt
ls test/4 | sed "s:^:4/:" | sed "s:$: 4:" >> test.txt
ls test/5 | sed "s:^:5/:" | sed "s:$: 5:" >> test.txt
ls test/6 | sed "s:^:6/:" | sed "s:$: 6:" >> test.txt

