#!/bin/bash
cd ~/pixel-cnn

python train.py -g 4 -o data/letters_data/checkpoints/0 -i data/letters_data -d letters -x 501 -f 0
