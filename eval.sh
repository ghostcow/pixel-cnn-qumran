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
# python eval.py -o data/letters_data/checkpoints/1 -f 0 -r -g 4 -u | tee eval10.log && \
# python eval.py -o data/letters_data/checkpoints/1 -f 2 -r -g 4 -u | tee eval12.log && \
# python eval.py -o data/letters_data/checkpoints/1 -f 3 -r -g 4 -u | tee eval13.log && \
# python eval.py -o data/letters_data/checkpoints/2 -f 0 -r -g 4 -u | tee eval20.log && \
# python eval.py -o data/letters_data/checkpoints/2 -f 1 -r -g 4 -u | tee eval21.log && \
# python eval.py -o data/letters_data/checkpoints/2 -f 3 -r -g 4 -u | tee eval23.log && \
# python eval.py -o data/letters_data/checkpoints/3 -f 0 -r -g 4 -u | tee eval30.log && \
# python eval.py -o data/letters_data/checkpoints/3 -f 1 -r -g 4 -u | tee eval31.log && \
# python eval.py -o data/letters_data/checkpoints/3 -f 2 -r -g 4 -u | tee eval32.log
# python eval.py -o data/letters_data/checkpoints/1 -f 1 -r -g 4 -u -b 5 | tee adapt1.log && \
# python eval.py -o data/letters_data/checkpoints/2 -f 2 -r -g 4 -u -b 1 | tee adapt2.log && \
# python eval.py -o data/letters_data/checkpoints/3 -f 3 -r -g 4 -u -b 3 | tee adapt3.log && \
# python eval.py -o data/letters_data/checkpoints/0 -f 0 -r -g 4 -u -b 8 | tee adapt0.log

# python eval.py -i /home/lioruzan/tehilim_data/test_data -o data/letters_data/checkpoints/0 -f 0 -r -j -b 5 | tee test_gen00.log && \
# python eval.py -i /home/lioruzan/tehilim_data/test_data -o data/letters_data/checkpoints/0 -f 0 -r -j -u -b 5 | tee test_gen_adapt0.log
# python eval.py -i /home/lioruzan/tehilim_data/test_data -o data/letters_data/checkpoints/1 -f 1 -r -j -u -b 5 | tee test_gen_adapt1.log && \
# python eval.py -i /home/lioruzan/tehilim_data/test_data -o data/letters_data/checkpoints/2 -f 2 -r -j -u -b 5 | tee test_gen_adapt2.log && \
# python eval.py -i /home/lioruzan/tehilim_data/test_data -o data/letters_data/checkpoints/3 -f 3 -r -j -u -b 5 | tee test_gen_adapt3.log
python eval.py -i data/letters_data -o data/letters_data/checkpoints/rand -r -c -f 0 -v 2>&1 | tee adaptr.log && \
python eval.py -i /home/lioruzan/tehilim_data/test_data -o data/letters_data/checkpoints/rand -r -c -f 0 -j -u 2>&1 | tee test_adaptr0.log && \
python eval.py -i /home/lioruzan/tehilim_data/test_data -o data/letters_data/checkpoints/rand -r -c -f 1 -j -u 2>&1 | tee test_adaptr1.log && \
python eval.py -i /home/lioruzan/tehilim_data/test_data -o data/letters_data/checkpoints/rand -r -c -f 2 -j -u 2>&1 | tee test_adaptr2.log && \
python eval.py -i /home/lioruzan/tehilim_data/test_data -o data/letters_data/checkpoints/rand -r -c -f 3 -j -u 2>&1 | tee test_adaptr3.log
