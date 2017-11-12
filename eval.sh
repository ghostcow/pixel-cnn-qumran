#!/bin/bash

# qumran dataset version 2 evaluation script
# usage: ./eval.sh <ORIENTATION>
# where ORIENTATION is the selected letter orientation from set {0,1,2,3,4,5,6,7}.
python eval.py --data_dir data/qumran_test_letters \
				--save_dir data/checkpoints/qv2_$1 \
				--data_set letters \
				--rotation $1 \
				--load_params \
				--just_gen \
				--nr_gpu 1 \
				--init_batch_size 1 \
				--gpu_mem_frac=0.4 \
				--batch_size 1 \
				--nr_resnet 5 \
				--nr_filters 40 \
				--nr_logistic_mix 5 \
				--single_angle \
				--num_psnr_trials=1 | tee logs/eval_$1.log
