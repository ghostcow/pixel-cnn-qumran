#!/bin/bash

# qumran dataset version 2 training script
# usage: ./train.sh <ORIENTATION>
# where ORIENTATION is the selected angle and/or reflection from set {0,1,2,3,4,5,6,7}.
python train.py -g 1 --gpu_mem_frac=0.4 -o data/checkpoints/qv2_$1 --gen_interval 50 --nr_resnet 5 --nr_filters 40 --nr_logistic_mix 5 --rotation $1 -i data/letters_data -x 501 | tee qv2_$1.log
