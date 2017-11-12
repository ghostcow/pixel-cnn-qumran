#!/bin/bash
python eval.py --data_dir data/letters_data --save_dir data/checkpoints/qv2_$1 --data_set letters \
	--rotation $1 --load_params --just_gen \
	--nr_gpu 1 --init_batch_size 1 \
	--batch_size 1 --nr_resnet 5 --nr_filters 40 \
	--nr_logistic_mix 5 --single_angle --num_psnr_trials=1 | tee eval_$1.log
