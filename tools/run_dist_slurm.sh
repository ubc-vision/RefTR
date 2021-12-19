#!/usr/bin/env bash
# --------------------------------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# --------------------------------------------------------------------------------------------------------------------------
# Modified from https://github.com/open-mmlab/mmdetection/blob/3b53fe15d87860c6941f3dda63c0f27422da6266/tools/slurm_train.sh
# --------------------------------------------------------------------------------------------------------------------------

set -x
PARTITION=edith
JOB_NAME=$1
GPUS=$2
RUN_COMMAND=${@:3}
RUN_TIME=${RUN_TIME:-"240:00:00"}
dt=$(date '+%Y_%m%d_%H')
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-2}
# CPUS_PER_TASK=2
SRUN_ARGS=${SRUN_ARGS:-""}

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --time=${RUN_TIME}\
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    ${RUN_COMMAND}\
>  ./logs/${JOB_NAME}_${dt}.log 2>&1 &

# Removing this args because the
# SRUN_ARGS="--nodelist=edith1" MASTER_PORT=29501 GPUS_PER_NODE=4  ./tools/run_dist_slurm.sh edith RefTR 4 configs/r50_deformable_vg_detr_single_scale_pretrained.sh