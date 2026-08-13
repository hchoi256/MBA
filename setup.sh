cat >/tmp/setup_mba_env.sh <<'BASH'
#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo; echo "[ERROR] line=${LINENO} command=${BASH_COMMAND}"; exit 1' ERR

WORKSPACE="/workspace"
REPRO_DIR="${WORKSPACE}/src/reprod"
REQ_FILE="${REPRO_DIR}/requirements-runtime.txt"
VENV="${WORKSPACE}/mba"
NLTK_DIR="${WORKSPACE}/.nltk_data"
HF_DIR="${WORKSPACE}/.cache/huggingface"
LOG="${WORKSPACE}/mba_environment_install.log"

exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo " mba environment installation"
echo " Workspace : ${WORKSPACE}"
echo " Venv      : ${VENV}"
echo " Log       : ${LOG}"
echo "============================================================"

cd "${WORKSPACE}"

if [[ ! -f "${REQ_FILE}" ]]; then
    echo "[ERROR] Requirements file not found:"
    echo "        ${REQ_FILE}"
    echo
    echo "Current reproduction folders:"
    find "${WORKSPACE}" -maxdepth 2 -name requirements-runtime.txt -print || true
    exit 1
fi

for file in \
    "${WORKSPACE}/2_sft.py" \
    "${WORKSPACE}/3_grpo.py" \
    "${WORKSPACE}/4_eval.py" \
    "${WORKSPACE}/wiki_fact_scorer.py"
do
    if [[ ! -f "${file}" ]]; then
        echo "[ERROR] Missing pipeline file: ${file}"
        exit 1
    fi
done

if [[ ! -d "${WORKSPACE}/FActScore" ]]; then
    echo "[ERROR] Missing directory: ${WORKSPACE}/FActScore"
    exit 1
fi

echo
echo "[1/10] Checking base Python, CUDA and GPU environment..."

python - <<'PY'
import sys
import torch

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__.split("+")[0] == "2.2.0", torch.__version__
assert torch.version.cuda == "12.1", torch.version.cuda
assert torch.cuda.is_available()
assert torch.cuda.device_count() in (4, 8), torch.cuda.device_count()
PY

if ! command -v nvcc >/dev/null 2>&1; then
    echo "[ERROR] nvcc is not installed."
    exit 1
fi

nvcc --version
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
    --format=csv,noheader

echo
echo "[2/10] Checking compiler tools..."

MISSING_TOOLS=()
for cmd in gcc g++ make git; do
    command -v "${cmd}" >/dev/null 2>&1 || MISSING_TOOLS+=("${cmd}")
done

if (( ${#MISSING_TOOLS[@]} > 0 )); then
    echo "Missing tools: ${MISSING_TOOLS[*]}"

    if [[ "$(id -u)" -eq 0 ]]; then
        apt-get update
        apt-get install -y --no-install-recommends \
            git \
            build-essential \
            ca-certificates \
            libsqlite3-dev \
            libgl1 \
            libglib2.0-0
        rm -rf /var/lib/apt/lists/*
    elif command -v sudo >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y --no-install-recommends \
            git \
            build-essential \
            ca-certificates \
            libsqlite3-dev \
            libgl1 \
            libglib2.0-0
        sudo rm -rf /var/lib/apt/lists/*
    else
        echo "[ERROR] Compiler tools are missing and root/sudo is unavailable."
        exit 1
    fi
fi

gcc --version | head -n 1
g++ --version | head -n 1
git --version

echo
echo "[3/10] Creating a clean Python virtual environment..."

if [[ -d "${VENV}" ]]; then
    echo "Removing previous virtual environment: ${VENV}"
    rm -rf "${VENV}"
fi

python -m venv "${VENV}"
source "${VENV}/bin/activate"

export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1

echo "Python executable: $(which python)"
python --version

echo
echo "[4/10] Installing packaging tools..."

python -m pip install \
    "pip==23.3.1" \
    "setuptools==81.0.0" \
    "wheel==0.41.2"

echo
echo "[5/10] Installing PyTorch 2.2.0 CUDA 12.1 into the clean venv..."

python -m pip install \
    "torch==2.2.0" \
    "torchvision==0.17.0" \
    "torchaudio==2.2.0" \
    --index-url https://download.pytorch.org/whl/cu121

python - <<'PY'
import torch

assert torch.__version__.split("+")[0] == "2.2.0", torch.__version__
assert torch.version.cuda == "12.1", torch.version.cuda
assert torch.cuda.is_available()

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
PY

echo
echo "[6/10] Installing the observed mba runtime packages..."

python -m pip install -r "${REQ_FILE}"

echo
echo "[7/10] Installing FlashAttention 2.5.9.post1..."

export CUDA_HOME="/usr/local/cuda"
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS="${MAX_JOBS:-8}"

python -m pip install \
    --no-build-isolation \
    "flash-attn==2.5.9.post1"

echo
echo "[8/10] Installing NLTK resources and checking spaCy..."

mkdir -p "${NLTK_DIR}" "${HF_DIR}"

NLTK_DATA="${NLTK_DIR}" python - <<PY
import nltk

target = "${NLTK_DIR}"

for resource in ("punkt", "punkt_tab"):
    print("Downloading:", resource)
    ok = nltk.download(resource, download_dir=target, quiet=False)
    if not ok:
        raise RuntimeError(f"Failed to download NLTK resource: {resource}")

print("NLTK resources installed in:", target)
PY

python - <<'PY'
import spacy

nlp = spacy.load("en_core_web_sm")
print("spaCy model:", nlp.meta.get("name"), nlp.meta.get("version"))
PY

echo
echo "[9/10] Creating the persistent activation script..."

cat >"${WORKSPACE}/activate_mba.sh" <<'ACTIVATE'
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
ACTIVATE

chmod +x "${WORKSPACE}/activate_mba.sh"

SOURCE_LINE='source /workspace/activate_mba.sh'
grep -qxF "${SOURCE_LINE}" "${HOME}/.bashrc" 2>/dev/null || \
    echo "${SOURCE_LINE}" >>"${HOME}/.bashrc"

source "${WORKSPACE}/activate_mba.sh"

echo
echo "[10/10] Verifying the complete runtime..."

python - <<'PY'
from __future__ import annotations

import importlib
import importlib.metadata
import py_compile
import sqlite3
import sys
from pathlib import Path

expected = {
    "torch": "2.2.0+cu121",
    "torchvision": "0.17.0+cu121",
    "torchaudio": "2.2.0+cu121",
    "transformers": "4.45.2",
    "tokenizers": "0.20.1",
    "huggingface-hub": "0.36.2",
    "accelerate": "1.13.0",
    "peft": "0.19.1",
    "safetensors": "0.8.0",
    "bitsandbytes": "0.43.3",
    "flash-attn": "2.5.9.post1",
    "timm": "0.9.12",
    "numpy": "1.26.4",
    "Pillow": "12.2.0",
    "scipy": "1.15.3",
    "scikit-learn": "1.7.2",
    "sentence-transformers": "5.5.1",
    "faiss-cpu": "1.7.4",
    "nltk": "3.9.4",
    "spacy": "3.8.14",
    "rank-bm25": "0.2.2",
    "openai": "2.41.1",
    "einops": "0.8.2",
    "sentencepiece": "0.2.1",
    "protobuf": "6.33.6",
}

errors = []

print("Python:", sys.executable)
print("Python version:", sys.version)

for package, wanted in expected.items():
    try:
        actual = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        actual = None

    marker = "OK" if actual == wanted else "ERROR"
    print(f"{marker:5s} {package:24s} expected={wanted:14s} actual={actual}")

    if actual != wanted:
        errors.append(f"{package}: expected {wanted}, found {actual}")

import torch

print()
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.__version__.split("+")[0] != "2.2.0":
    errors.append(f"Wrong torch version: {torch.__version__}")
if torch.version.cuda != "12.1":
    errors.append(f"Wrong Torch CUDA version: {torch.version.cuda}")
if not torch.cuda.is_available():
    errors.append("CUDA is unavailable")
if torch.cuda.device_count() not in (4, 8):
    errors.append(f"Expected 4 or 8 GPUs, found {torch.cuda.device_count()}")

for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(
        f"GPU{index}: {props.name}, "
        f"{props.total_memory / 2**30:.1f} GiB, "
        f"compute={props.major}.{props.minor}"
    )

print()
print("Extension imports:")

for module_name in (
    "flash_attn",
    "bitsandbytes",
    "transformers",
    "tokenizers",
    "peft",
    "accelerate",
    "timm",
    "faiss",
    "sentence_transformers",
    "spacy",
    "nltk",
):
    try:
        module = importlib.import_module(module_name)
        print("OK   ", module_name, getattr(module, "__file__", ""))
    except Exception as exc:
        errors.append(f"Import failed: {module_name}: {exc!r}")

print()
print("SQLite FTS5:")

conn = sqlite3.connect(":memory:")
options = {row[0] for row in conn.execute("pragma compile_options")}
print("SQLite version:", sqlite3.sqlite_version)
print("ENABLE_FTS5:", "ENABLE_FTS5" in options)

if "ENABLE_FTS5" not in options:
    errors.append("SQLite lacks FTS5 support")

print()
print("Pipeline compilation:")

for filename in ("2_sft.py", "3_grpo.py", "4_eval.py"):
    path = Path("/workspace") / filename
    try:
        py_compile.compile(str(path), doraise=True)
        print("OK   ", path)
    except Exception as exc:
        errors.append(f"Compilation failed: {path}: {exc!r}")

print()
print("Custom FActScore imports:")

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/FActScore")

for module_name in (
    "factscore.factscorer",
    "factscore.atomic_facts",
    "factscore.retrieval",
    "factscore.llama_lm",
    "wiki_fact_scorer",
):
    try:
        module = importlib.import_module(module_name)
        print("OK   ", module_name, getattr(module, "__file__", ""))
    except Exception as exc:
        errors.append(f"FActScore import failed: {module_name}: {exc!r}")

print()
if errors:
    print("ENVIRONMENT VERIFICATION FAILED")
    for error in errors:
        print(" -", error)
    raise SystemExit(1)

print("CORE ENVIRONMENT VERIFICATION PASSED")
PY

echo
echo "Running pip dependency check..."
python -m pip check

echo
echo "Checking important workspace assets..."

ASSETS=(
    "/workspace/FActScore/data/enwiki-20260101-fts.db"
    "/workspace/mistral-7b-instruct/config.json"
    "/workspace/data/mba/expert_data"
    "/workspace/data/ADEChallengeData2016/images/validation"
    "/workspace/data/mba/MBALibrary/config.json"
    "/workspace/data/mba/MBALibrary/market_evidence.faiss"
    "/workspace/data/mba/MBALibrary/market_evidence_meta.jsonl"
)

MISSING_ASSETS=0

for asset in "${ASSETS[@]}"; do
    if [[ -e "${asset}" ]]; then
        echo "OK      ${asset}"
    else
        echo "MISSING ${asset}"
        MISSING_ASSETS=$((MISSING_ASSETS + 1))
    fi
done

echo
echo "============================================================"
echo " mba core environment installation completed successfully"
echo "============================================================"
echo
echo "Activate in future shells with:"
echo
echo "  source /workspace/activate_mba.sh"
echo
echo "Installation log:"
echo
echo "  ${LOG}"
echo

if (( MISSING_ASSETS > 0 )); then
    echo "WARNING: ${MISSING_ASSETS} model/data asset(s) are missing."
    echo "The Python environment is installed, but those assets must be"
    echo "copied or mounted before the affected pipeline stage can run."
else
    echo "All required model/data asset paths were found."
fi

echo
echo "SFT command:"
echo "  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 /workspace/2_sft.py"
echo
echo "GRPO command:"
echo "  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u /workspace/3_grpo.py"
echo
echo "Eval command:"
echo "  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u /workspace/4_eval.py --model_type base"
BASH

chmod +x /tmp/setup_mba_env.sh
bash /tmp/setup_mba_env.sh