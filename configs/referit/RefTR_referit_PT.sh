#!/usr/bin/env bash
# coco pretrained provided by r50_dconvDETR_C5_pretrained_coco_Q100 setting
set -x

EXP_DIR=exps/referit/RefTR_PT
PY_ARGS=${@:1}

conda activate pytorch
which python

python3.8 -u main_vg.py \
    --resume './SAVED_MODEL/PT/RefTR_VG_PT_08.pth'\
    --resume_model_only\
    --num_feature_levels 1\
    --num_queries_per_phrase 1\
    --dec_layers 6\
    --aux_loss \
    --dataset referit\
    --img_size 640\
    --max_img_size 640\
    --ckpt_cycle 90\
    --epochs 90\
    --lr_drop 60\
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
    
#     # --resume 
# python3.8 -u main_vg.py \
#     --resume ${EXP_DIR}/checkpoint0069.pth\
#     --num_feature_levels 1\
#     --num_queries_per_phrase 1\
#     --dec_layers 3\
#     --dataset referit\
#     --img_size 640\
#     --max_img_size 640\
#     --epochs 90\
#     --lr_drop 60\
#     --output_dir ${EXP_DIR} \
#     ${PY_ARGS}