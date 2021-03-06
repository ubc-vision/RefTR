#!/usr/bin/env bash
# coco pretrained provided by r50_dconvDETR_C5_pretrained_coco_Q100 setting
set -x

###############################################################################################
# EXP_DIR=exps/refcocog/RefTR_SEG
# PY_ARGS=${@:1}

# conda activate pytorch
# which python

# python3.8 -u main_vg.py \
#     --pretrained_model "./SAVED_MODEL/refcoco_50_det/RefTR_refcocog_l6/checkpoint_best.pth"\
#     --num_feature_levels 1\
#     --num_queries_per_phrase 1\
#     --masks\
#     --lr 1e-5\
#     --lr_mask_branch_proj 10\
#     --dataset refcocog_umd\
#     --train_split train\
#     --test_split val test\
#     --dec_layers 6\
#     --aux_loss \
#     --img_size 640\
#     --max_img_size 640\
#     --epochs 40\
#     --lr_drop 30\
#     --output_dir ${EXP_DIR} \
#     ${PY_ARGS}
    
#     # --resume 

###############################################################################################
EXP_DIR=exps/refcocog/RefTR_SEG_PT
PY_ARGS=${@:1}

conda activate pytorch
which python

python3.8 -u main_vg.py \
    --pretrained_model "./SAVED_MODEL/refcoco_50_det_pretrained/RefTR_refcocog_pt/checkpoint_best.pth"\
    --num_feature_levels 1\
    --num_queries_per_phrase 1\
    --masks\
    --lr 1e-5\
    --lr_mask_branch_proj 10\
    --dataset refcocog_umd\
    --train_split train\
    --test_split test\
    --dec_layers 6\
    --aux_loss \
    --img_size 640\
    --max_img_size 640\
    --epochs 40\
    --lr_drop 30\
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
    
    # --resume 