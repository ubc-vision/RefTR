#!/usr/bin/env bash
# coco pretrained provided by r50_dconvDETR_C5_pretrained_coco_Q100 setting
set -x

EXP_DIR=exps/RefTR_refcocog_unc
PY_ARGS=${@:1}

conda activate pytorch
which python

python3.8 -u main_vg.py \
    --pretrained_model "./data/MODEL_ZOO/detr-r50-e632da11.pth"\
    --num_feature_levels 1\
    --num_queries_per_phrase 1\
    --dataset refcocog_umd\
    --train_split train\
    --test_split val test\
    --dec_layers 6\
    --aux_loss \
    --img_size 640\
    --max_img_size 640\
    --epochs 90\
    --lr_drop 60\
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
    
    # --resume 