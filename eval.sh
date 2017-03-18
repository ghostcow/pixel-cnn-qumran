#!/bin/bash
# python eval.py -o data/letters_data/checkpoints/0 -f 0 -r -g 4 -j | tee eval00.log && \
# python eval.py -o data/letters_data/checkpoints/1 -f 1 -r -g 4 -j | tee eval11.log && \
# python eval.py -o data/letters_data/checkpoints/2 -f 2 -r -g 4 -j | tee eval22.log && \
# python eval.py -o data/letters_data/checkpoints/3 -f 3 -r -g 4 -j | tee eval33.log
# python eval.py -o data/letters_data/checkpoints/rand -f 0 -r -c -g 4 -j | tee evalr0.log && \
# python eval.py -o data/letters_data/checkpoints/rand -f 1 -r -c -g 4 -j | tee evalr1.log && \
# python eval.py -o data/letters_data/checkpoints/rand -f 2 -r -c -g 4 -j | tee evalr2.log && \
# python eval.py -o data/letters_data/checkpoints/rand -f 3 -r -c -g 4 -j | tee evalr3.log
# python eval.py -i /home/lioruzan/tehilim_data/test_data -o data/letters_data/checkpoints/0 -f 0 -r -t -g 4 -j -u | tee gen00.log
python eval.py -o data/letters_data/checkpoints/1 -f 0 -r -g 4 -j -u | tee eval10.log && \
python eval.py -o data/letters_data/checkpoints/1 -f 2 -r -g 4 -j -u | tee eval12.log && \
python eval.py -o data/letters_data/checkpoints/1 -f 3 -r -g 4 -j -u | tee eval13.log && \
python eval.py -o data/letters_data/checkpoints/2 -f 0 -r -g 4 -j -u | tee eval20.log && \
python eval.py -o data/letters_data/checkpoints/2 -f 1 -r -g 4 -j -u | tee eval21.log && \
python eval.py -o data/letters_data/checkpoints/2 -f 3 -r -g 4 -j -u | tee eval23.log && \
python eval.py -o data/letters_data/checkpoints/3 -f 0 -r -g 4 -j -u | tee eval30.log && \
python eval.py -o data/letters_data/checkpoints/3 -f 1 -r -g 4 -j -u | tee eval31.log && \
python eval.py -o data/letters_data/checkpoints/3 -f 2 -r -g 4 -j -u | tee eval32.log
