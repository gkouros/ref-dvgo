#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n refdvgo python=3.9 -y && \
    conda activate temp && \
    conda install 'pytorch<2.0.0' torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y && \
    conda install "cuda-nvcc==11.7" "cuda-toolkit==11.7" -c nvidia -c conda-forge
    conda install 'gxx==11.2.0' -c conda-forge -y && \
    conda install cudatoolkit cudatoolkit-dev  -c conda-forge
    conda install pytorch-scatter -c pyg -y && \
    pip3 install -r requirements.txt