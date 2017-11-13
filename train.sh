#!/bin/bash

# qumran dataset version 2 training script
# usage: ./train.sh <ORIENTATION>
# where ORIENTATION is the selected letter orientation from set {0,1,2,3,4,5,6,7}.
python train.py --data_dir data/qumran_dataset_v2 \
				--nr_gpu 1 \
				--save_dir data/checkpoints/qv2_$1 \
				--gen_interval 50 \
				--nr_resnet 5 \
				--nr_filters 40 \
				--nr_logistic_mix 5 \
				--rotation $1 \
				--max_epochs 501 | tee logs/qv2_$1.log
