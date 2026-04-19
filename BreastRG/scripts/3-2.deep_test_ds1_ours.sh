#!/bin/bash

dataset="DS1"
annotation="data/iu_xray/annotation.json"
base_dir="./data/iu_xray/images"
delta_file="/home/csexrjiang/20260418_BreastRG/checkpoint/RG_model.pth"


version="ds1"
savepath="./save/$dataset/$version"

python -u train.py \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --test_batch_size 8 \
    --savedmodel_path ${savepath} \
    --max_length 800 \
    --min_new_tokens 50 \
    --max_new_tokens 800 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices 1 \

    2>&1 |tee -a ${savepath}/log.txt