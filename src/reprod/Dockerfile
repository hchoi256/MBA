# Reconstructed clean image matching the known-good mba runtime.
# The original image digest was not captured; the observed core stack was:
# Python 3.10.13, PyTorch 2.2.0, CUDA 12.1, cuDNN 8.9, Ubuntu 22.04.
ARG BASE_IMAGE=pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
ARG MAX_JOBS=8

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    ACCELERATE_USE_DEEPSPEED=false \
    DISABLE_DEEPSPEED=1 \
    CUDA_MODULE_LOADING=LAZY \
    TORCH_CUDA_ARCH_LIST=8.6 \
    HF_HOME=/workspace/.cache/huggingface \
    NLTK_DATA=/opt/nltk_data \
    PYTHONPATH=/workspace:/workspace/FActScore

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        sqlite3 \
        libsqlite3-dev \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements-runtime.txt /tmp/requirements-runtime.txt

# Keep the base image's exact PyTorch 2.2.0 CUDA 12.1 build.
RUN python - <<'PY'
import torch
assert torch.__version__.split('+')[0] == '2.2.0', torch.__version__
assert torch.version.cuda == '12.1', torch.version.cuda
print('Base PyTorch OK:', torch.__version__, 'CUDA', torch.version.cuda)
PY

RUN python -m pip install --no-cache-dir -r /tmp/requirements-runtime.txt

# flash-attn was 2.5.9.post1 in the known-good container. Install only after
# PyTorch is fixed so the extension is compiled/linked against the correct ABI.
RUN MAX_JOBS=${MAX_JOBS} python -m pip install --no-cache-dir --no-build-isolation \
        flash-attn==2.5.9.post1

RUN mkdir -p /opt/nltk_data && python - <<'PY'
import nltk
for name in ('punkt', 'punkt_tab'):
    nltk.download(name, download_dir='/opt/nltk_data', quiet=True, raise_on_error=True)
print('NLTK data installed')
PY

COPY verify_environment.py /usr/local/bin/verify_mba_environment.py
RUN chmod +x /usr/local/bin/verify_mba_environment.py

CMD ["bash"]
