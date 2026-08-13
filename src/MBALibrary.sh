# Separate this env from SFT/GRPO because FlashAttention2 training
# requires a different torch/CUDA stack than vLLM-based market DB generation.
conda create -n MBALibrary python=3.10 -y
source activate MBALibrary

export PYTHONNOUSERSITE=1

python -m pip install -U pip setuptools wheel
python -m pip install "vllm==0.22.1"
python -m pip install \
  transformers \
  accelerate \
  sentence-transformers \
  faiss-cpu \
  pandas \
  numpy==1.26.4 \
  requests \
  tqdm \
  pillow \
  openpyxl \
  psutil

python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("torch file:", torch.__file__)

import vllm
print("vllm:", vllm.__version__)

import faiss
print("faiss ok")

import sentence_transformers
print("sentence_transformers ok")

import openpyxl
print("openpyxl ok")
PY

# torch: 2.11.0
# vllm: 0.22.1
# faiss ok
# sentence_transformers ok
# openpyxl ok