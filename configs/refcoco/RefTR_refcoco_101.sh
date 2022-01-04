#!/usr/bin/env bash
# coco pretrained provided by r50_dconvDETR_C5_pretrained_coco_Q100 setting
set -x

EXP_DIR=exps/RefTR_refcoco_unc_101
PY_ARGS=${@:1}

conda activate pytorch
which python

python3.8 -u main_vg.py \
    --pretrained_model "./data/MODEL_ZOO/detr-r101-2c7b67e5.pth"\
    --num_feature_levels 1\
    --num_queries_per_phrase 1\
    --dataset refcoco_unc\
    --train_split train\
    --test_split val testA testB\
    --dec_layers 6\
    --backbone resnet101\
    --aux_loss \
    --img_size 640\
    --max_img_size 640\
    --epochs 90\
    --lr_drop 60\
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
    
    # --resume 