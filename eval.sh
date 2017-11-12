#!/bin/bash

# qumran dataset version 2 evaluation script
# script completes the test set letters using adaptive orientation.
# completion of each orientation is saved in data dir as "letter_completion_orientation_$i.pkl"
# where i in 0,..,7
for i in `seq 0 7`; do
	python eval.py --data_dir data/qumran_test_letters \
					--save_dir data/checkpoints/qv2_$i \
					--rotation $i \
					--data_set letters \
					--load_params \
					--just_gen \
					--nr_gpu 1 \
					--init_batch_size 1 \
					--batch_size 5 \
					--gpu_mem_frac=0.4 \
					--nr_resnet 5 \
					--nr_filters 40 \
					--nr_logistic_mix 5 \
					--single_angle \
					--num_psnr_trials=1 | tee logs/eval_$i.log
done
