#!/usr/bin/env bash
set -euo pipefail

IMAGE="${MBA_IMAGE:-mba:torch2.2-cu121}"
WORKSPACE="${MBA_WORKSPACE:-$PWD}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

mkdir -p "$HF_CACHE"

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -e HF_HOME=/workspace/.cache/huggingface \
  -e TOKENIZERS_PARALLELISM=false \
  -e ACCELERATE_USE_DEEPSPEED=false \
  -e DISABLE_DEEPSPEED=1 \
  -e CUDA_MODULE_LOADING=LAZY \
  -e PYTHONPATH=/workspace:/workspace/FActScore \
  -e NLTK_DATA=/opt/nltk_data \
  -v "$WORKSPACE:/workspace" \
  -v "$HF_CACHE:/workspace/.cache/huggingface" \
  -w /workspace \
  "$IMAGE" bash
