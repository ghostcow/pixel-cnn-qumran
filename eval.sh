#!/bin/bash
python eval.py -i data/letters_data -o data/checkpoints/qv2_$1 --rotation $1 --load_params --just_gen --single_angle -g 1 --gpu_mem_frac=0.4 --batch_size 19 --nr_resnet 5 --nr_filters 40 --nr_logistic_mix 5 | tee qum_test_$1.log
