#!/usr/bin/env bash
# coco pretrained provided by r50_dconvDETR_C5_pretrained_coco_Q100 setting
set -x

EXP_DIR=exps/RefTR_flickr
PY_ARGS=${@:1}

conda activate pytorch
which python

python3.8 -u main_vg.py \
    --pretrained_model "./data/MODEL_ZOO/detr-r50-e632da11.pth"\
    --num_feature_levels 1\
    --dataset flickr30k\
    --dec_layers 6\
    --img_size 640\
    --max_img_size 640\
    --batch_size 16\
    --epochs 60\
    --warm_up_epoch 5\
    --lr_schedule CosineWarmupLR\
    --aux_loss\
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
    
    # --num_queries_per_phrase 1\
    # --resume 