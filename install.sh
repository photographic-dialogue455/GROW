#!/usr/bin/env bash
# GROW environment installer. Tested: Python 3.10 · A800-80GB · CUDA 12.6 · torch 2.7.0
set -e

# run from the repo root (so requirements.txt / pip install -e . resolve)
cd "$(dirname "$0")"

# system deps: ffmpeg/sox/libsndfile (audio I/O)
#   Debian/Ubuntu: sudo apt-get install -y ffmpeg sox libsndfile1

conda create -n grow python=3.10 -y
# make `conda activate` available in this non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate grow

pip install torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e .
