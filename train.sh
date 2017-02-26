#!/bin/bash
cd ~/pixel-cnn
ROOT="/home/lioruzan/letters_to_classify_v1"
CHKPTS="$ROOT/checkpoints"

OUT="$CHKPTS/rand"
mkdir -p $OUT
python train.py -g 4 -o data/letters_data -i data/letters_data -d letters -x 501 -c && \
mv $ROOT/letters_sample*.png $ROOT/test_bpd_letters.npz $ROOT/params_letters.ckpt.data-00000-of-00001 $ROOT/params_letters.ckpt.index $ROOT/params_letters.ckpt.meta $ROOT/checkpoint $OUT

OUT="$CHKPTS/0"
mkdir -p $OUT
python train.py -g 4 -o data/letters_data -i data/letters_data -d letters -x 501 -f 0 && \
mv $ROOT/letters_sample*.png $ROOT/test_bpd_letters.npz $ROOT/params_letters.ckpt.data-00000-of-00001 $ROOT/params_letters.ckpt.index $ROOT/params_letters.ckpt.meta $ROOT/checkpoint $OUT

OUT="$CHKPTS/1"
mkdir -p $OUT
python train.py -g 4 -o data/letters_data -i data/letters_data -d letters -x 501 -f 1 && \
mv $ROOT/letters_sample*.png $ROOT/test_bpd_letters.npz $ROOT/params_letters.ckpt.data-00000-of-00001 $ROOT/params_letters.ckpt.index $ROOT/params_letters.ckpt.meta $ROOT/checkpoint $OUT

OUT="$CHKPTS/2"
mkdir -p $OUT
python train.py -g 4 -o data/letters_data -i data/letters_data -d letters -x 501 -f 2 && \
mv $ROOT/letters_sample*.png $ROOT/test_bpd_letters.npz $ROOT/params_letters.ckpt.data-00000-of-00001 $ROOT/params_letters.ckpt.index $ROOT/params_letters.ckpt.meta $ROOT/checkpoint $OUT

OUT="$CHKPTS/3"
mkdir -p $OUT
python train.py -g 4 -o data/letters_data -i data/letters_data -d letters -x 501 -f 3 && \
mv $ROOT/letters_sample*.png $ROOT/test_bpd_letters.npz $ROOT/params_letters.ckpt.data-00000-of-00001 $ROOT/params_letters.ckpt.index $ROOT/params_letters.ckpt.meta $ROOT/checkpoint $OUT
