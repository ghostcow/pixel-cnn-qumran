#!/bin/bash
cd ~/pixel-cnn

python train.py -g 4 -o data/letters_data/checkpoints/0 -i data/letters_data -d letters -x 501 -f 0 -r && \
python train.py -g 4 -o data/letters_data/checkpoints/1 -i data/letters_data -d letters -x 501 -f 1 && \
python train.py -g 4 -o data/letters_data/checkpoints/2 -i data/letters_data -d letters -x 501 -f 2 && \
python train.py -g 4 -o data/letters_data/checkpoints/3 -i data/letters_data -d letters -x 501 -f 3 && \
python train.py -g 4 -o data/letters_data/checkpoints/rand -i data/letters_data -d letters -x 501 -c
