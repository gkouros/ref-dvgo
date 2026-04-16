#!/bin/bash

DATASET=$1
EXP=$2
ARGS=${@:3} # all subsequent args are assumed args for the python script
echo "ARGS: $ARGS"

source ~/miniforge3/etc/profile.d/conda.sh

conda activate refdvgo
# export TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6+PTX"
# export CUDA_HOME="$CONDA_PREFIX"
# export PATH="$CUDA_HOME/bin:$CONDA_PREFIX/bin:$PATH"
# export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH"

# export CC=/esat/kenaz/gkouros/miniforge3/envs/refdvgo/bin/x86_64-conda-linux-gnu-gcc
# export CXX=/esat/kenaz/gkouros/miniforge3/envs/refdvgo/bin/x86_64-conda-linux-gnu-g++
# export TORCH_CUDA_ARCH_LIST="8.6"
# export NVCC_FLAGS="-allow-unsupported-compiler"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export CPATH="$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH"

echo PATH=$PATH
echo LDPATH=$LD_LIBRARY_PATH

DIR="$(pwd)/$( dirname -- "$0"; )"
cd ${DIR}
mkdir -p logs/$DATASET/$EXP

MAX_JOBS=2 python run.py --config configs/${DATASET}.py --dump_images --render_test --eval_ssim --eval_lpips_vgg --expname $EXP --overwrite="$ARGS"

conda deactivate
