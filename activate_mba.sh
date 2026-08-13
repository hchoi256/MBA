#!/usr/bin/env bash

source /workspace/mba/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONPATH=/workspace:/workspace/FActScore
export HF_HOME=/workspace/.cache/huggingface
export NLTK_DATA=/workspace/.nltk_data
export TOKENIZERS_PARALLELISM=false
export ACCELERATE_USE_DEEPSPEED=false
export DISABLE_DEEPSPEED=1
export CUDA_MODULE_LOADING=LAZY
export CUDA_HOME=/usr/local/cuda
