#!/usr/bin/env bash
# coco pretrained provided by r50_dconvDETR_C5_pretrained_coco_Q100 setting
set -x

EXP_DIR=exps/flickr/RefTR_pt
PY_ARGS=${@:1}

conda activate pytorch
which python

python3.8 -u main_vg.py \
    --resume "./SAVED_MODEL/PT/RefTR_VG_PT_08.pth"\
    --resume_model_only\
    --num_feature_levels 1\
    --num_queries_per_phrase 1\
    --dataset flickr30k\
    --dec_layers 6\
    --img_size 640\
    --max_img_size 640\
    --epochs 40\
    --lr_drop 30\
    --aux_loss\
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
    
    # --resume 