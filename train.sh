#!/bin/bash

DATASET=$1
EXP=$2
ARGS=${@:3} # all subsequent args are assumed args for the python script
echo "ARGS: $ARGS"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate refdvgo

export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/bin/:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES/CUDA/}

echo PATH=$PATH
echo LDPATH=$LD_LIBRARY_PATH

DATA_DIR=/esat/topaz/gkouros/datasets/nerf/$DATASET
DIR="$(pwd)/$( dirname -- "$0"; )/.."
cd ${DIR}
mkdir -p logs/$DATASET/$EXP

MAX_JOBS=2 python run.py --config configs/${DATASET}.py --render_test --eval_ssim --eval_lpips_vgg --expname $EXP --overwrite="$ARGS"

conda deactivate
