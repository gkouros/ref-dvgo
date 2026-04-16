#!/bin/bash -l

source ~/miniforge3/etc/profile.d/conda.sh

mamba create -n refdvgo python=3.9.15 -y && \
    mamba activate refdvgo && \
    mamba install 'gxx==11.2.0' -c conda-forge -y && \
    mamba install 'pytorch<2.0.0' torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y && \
    pip3 install -r requirements.txt && \
    mamba install -y "mkl=2024.0.0" && \
    pip install -U openmim && mim install mmcv && \
    mamba install -c conda-forge binutils && \
    mamba install pytorch-scatter -c pyg && \
    mamba install 'gxx==11.2.0' -c conda-forge -y && \
    pip install torch_efficient_distloss