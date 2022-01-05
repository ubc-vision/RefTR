#!/usr/bin/env bash
# coco pretrained provided by r50_dconvDETR_C5_pretrained_coco_Q100 setting
set -x

PY_ARGS=${@:1}

conda activate pytorch
which python

EXP_DIR=exps/refcoco/r101_det
python3.8 -u main_vg.py \
    --pretrained_model "./data/MODEL_ZOO/detr-r101-2c7b67e5.pth"\
    --num_feature_levels 1\
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


EXP_DIR=exps/refcoco/r101
python3.8 -u main_vg.py \
    --pretrained_model "./SAVED_MODEL/refcoco_101_det/RefTR_refcoco_101/checkpoint_best.pth"\
    --num_feature_levels 1\
    --masks\
    --lr 1e-5\
    --lr_mask_branch_proj 10\
    --dataset refcoco_unc\
    --train_split train\
    --test_split val testA testB\
    --dec_layers 6\
    --backbone resnet101\
    --aux_loss \
    --img_size 640\
    --max_img_size 640\
    --epochs 40\
    --lr_drop 30\
    --output_dir ${EXP_DIR} \