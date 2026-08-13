#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ENV="${CAPTION_SOURCE_ENV:-/workspace/mba}"
PYTHON_SCRIPT="${CAPTION_SCRIPT:-${WORKSPACE}/generate_captions.py}"

# Only NumPy 1.26.4 is installed here.
# The original /workspace/mba environment is never modified.
CAPTION_DEPS="${CAPTION_DEPS_DIR:-${WORKSPACE}/.caption-pydeps}"

log() {
    echo "[caption] $*"
}

fail() {
    echo "[caption] ERROR: $*" >&2
    exit 1
}

[[ -x "${SOURCE_ENV}/bin/python" ]] || \
    fail "Python environment not found: ${SOURCE_ENV}"

[[ -f "${PYTHON_SCRIPT}" ]] || \
    fail "Python script not found: ${PYTHON_SCRIPT}"

# Recreate only the isolated NumPy overlay.
if [[ "${1:-}" == "--rebuild-env" ]]; then
    log "Removing isolated caption dependencies."
    rm -rf "${CAPTION_DEPS}"
    shift
fi

mkdir -p "${CAPTION_DEPS}"

# Check whether the isolated NumPy installation exists and is valid.
NEED_NUMPY_INSTALL=0

if ! PYTHONPATH="${CAPTION_DEPS}" \
     PYTHONNOUSERSITE=1 \
     "${SOURCE_ENV}/bin/python" - <<'PY' >/dev/null 2>&1
import numpy
assert numpy.__version__ == "1.26.4", numpy.__version__
PY
then
    NEED_NUMPY_INSTALL=1
fi

if [[ "${NEED_NUMPY_INSTALL}" -eq 1 ]]; then
    log "Installing NumPy 1.26.4 into: ${CAPTION_DEPS}"

    rm -rf "${CAPTION_DEPS}"
    mkdir -p "${CAPTION_DEPS}"

    "${SOURCE_ENV}/bin/python" -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --no-deps \
        --target "${CAPTION_DEPS}" \
        "numpy==1.26.4"
fi

log "Checking caption runtime."

PYTHONPATH="${CAPTION_DEPS}" \
PYTHONNOUSERSITE=1 \
"${SOURCE_ENV}/bin/python" - <<'PY'
import sys
import numpy
import torch
import transformers
from PIL import Image
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)

print(f"[caption] Python:       {sys.executable}")
print(f"[caption] NumPy:        {numpy.__version__}")
print(f"[caption] PyTorch:      {torch.__version__}")
print(f"[caption] Transformers: {transformers.__version__}")
print(f"[caption] CUDA:         {torch.cuda.is_available()}")

if numpy.__version__ != "1.26.4":
    raise RuntimeError(
        f"Expected NumPy 1.26.4, but loaded {numpy.__version__}"
    )
PY

log "Starting caption generation."
log "Script: ${PYTHON_SCRIPT}"

export PYTHONPATH="${CAPTION_DEPS}"
export PYTHONNOUSERSITE=1

exec "${SOURCE_ENV}/bin/python" "${PYTHON_SCRIPT}" "$@"
