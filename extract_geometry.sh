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

DATA_DIR=/esat/topaz/gkouros/datasets/nerf/$DATASET
DIR="$(pwd)/$( dirname -- "$0"; )/.."
cd ${DIR}
mkdir -p logs/$DATASET/$EXP

ARGS=""
# export bbox and cams only
python3 run.py --config configs/${DATASET}.py --render_test --expname $EXP --export_bbox_and_cams_only "bbox_and_cams.npz"
# export coarse geometry from trained model
python3 run.py --config configs/${DATASET}.py --render_test --expname $EXP --export_coarse_only "coarse_geometry.npz"

conda deactivate
