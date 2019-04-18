#!/usr/bin/env bash
cd ../
python3 run.py 8 bilstm_crf 10 --filter-size 4 --num-filter-maps 500 --dropout 0.2 --lr 0.003 --gpu