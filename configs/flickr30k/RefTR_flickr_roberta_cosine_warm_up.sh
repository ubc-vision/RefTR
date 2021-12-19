#!/usr/bin/env bash
# coco pretrained provided by r50_dconvDETR_C5_pretrained_coco_Q100 setting
set -x

file_name=${0##*/}
file_name_without_ext=${file_name%.*}
EXP_DIR=exps/flickr/${file_name_without_ext}
PY_ARGS=${@:1}
dataset=flickr30k
backbone=resnet50
bert_model=roberta-base
# mkdir -p ./logs/${dataset}

conda activate pytorch
which python

python3.8 -u main_vg.py \
    --pretrained_model "./data/MODEL_ZOO/detr-r50-e632da11.pth"\
    --num_feature_levels 1\
    --num_queries_per_phrase 1\
    --dataset ${dataset}\
    --bert_model ${bert_model}\
    --backbone ${backbone}\
    --dec_layers 6\
    --img_size 640\
    --max_img_size 640\
    --epochs 65 \
    --lr_schedule CosineWarmupLR\
    --lr_drop_epochs 40 55\
    --warm_up_epoch 10\
    --aux_loss\
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
# >  ./logs/${dataset}/${file_name_without_ext}.log 2>&1 &
    # --resume 