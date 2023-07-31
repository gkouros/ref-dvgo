#!/bin/bash

DATASET=$1
EXP=$2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate refdvgo

export PATH="/usr/local/cuda/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib64:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES/CUDA/}

DATA_DIR=/esat/topaz/gkouros/datasets/nerf/$DATASET
DIR="$(pwd)/$( dirname -- "$0"; )/.."
cd ${DIR}
mkdir -p logs/$DATASET/$EXP

python run.py --config logs/$DATASET/$EXP/config.py --render_only --render_test --dump_images --eval_ssim --expname $EXP --overwrite="$ARGS"

conda deactivate
