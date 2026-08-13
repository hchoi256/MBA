#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE="/workspace"
FACTSCORE_DIR="${WORKSPACE}/FActScore"
DATA_DIR="${FACTSCORE_DIR}/data"
MODEL_DIR="${FACTSCORE_DIR}/models"
LLAMA_DIR="${WORKSPACE}/llama-7B"

DB_ID="1Qu4JHWjpUKhGPaAW5UHhS5RJ545CVy4I"
DB_PATH="${DATA_DIR}/enwiki-20230401.db"
DB_PART="${DB_PATH}.part"
MIN_DB_BYTES=$((10 * 1024 * 1024 * 1024))

if [[ -f "${WORKSPACE}/activate_mba.sh" ]]; then
  # shellcheck disable=SC1091
  source "${WORKSPACE}/activate_mba.sh"
fi

PYTHON="$(command -v python)"
HF_CLI="$(command -v hf || true)"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

cleanup() {
  if [[ -n "${FACTSCORE_TMP:-}" && -d "${FACTSCORE_TMP}" ]]; then
    rm -rf "${FACTSCORE_TMP}"
  fi
}
trap cleanup EXIT

db_is_valid() {
  "${PYTHON}" - "${DB_PATH}" "${MIN_DB_BYTES}" <<'PY'
import os
import sys

path = sys.argv[1]
minimum_size = int(sys.argv[2])

if not os.path.isfile(path):
    raise SystemExit(1)

if os.path.getsize(path) < minimum_size:
    raise SystemExit(1)

with open(path, "rb") as f:
    if f.read(16) != b"SQLite format 3\x00":
        raise SystemExit(1)
PY
}

command -v unzip >/dev/null 2>&1 || \
  fail "'unzip' is not installed. Run sudo apt-get install -y unzip first."

"${PYTHON}" -c 'import gdown' >/dev/null 2>&1 || \
  fail "'gdown' is not installed. Run python -m pip install gdown in the mba virtual environment."

[[ -n "${HF_CLI}" ]] || fail "The 'hf' command could not be found."
[[ -d "${FACTSCORE_DIR}/factscore" ]] || \
  fail "FActScore source not found: ${FACTSCORE_DIR}/factscore"

mkdir -p \
  "${DATA_DIR}" \
  "${MODEL_DIR}" \
  "${LLAMA_DIR}"

log "Check and download the original LLaMA-7B model"

"${HF_CLI}" download huggyllama/llama-7b \
  --include "*.safetensors" \
            "*.safetensors.index.json" \
            "config.json" \
            "generation_config.json" \
            "tokenizer.json" \
            "tokenizer.model" \
            "tokenizer_config.json" \
            "special_tokens_map.json" \
            "added_tokens.json" \
            "README.md" \
            "LICENSE" \
            ".gitattributes" \
  --local-dir "${LLAMA_DIR}"

[[ -s "${LLAMA_DIR}/config.json" ]] || \
  fail "LLaMA config.json not found: ${LLAMA_DIR}/config.json"

if ! compgen -G "${LLAMA_DIR}/*.safetensors" >/dev/null; then
  fail "LLaMA safetensors weights not found: ${LLAMA_DIR}"
fi

log "Check Wikipedia DB"

if [[ -f "${DB_PATH}" ]] && ! db_is_valid; then
  echo "Removing invalid or incomplete DB: ${DB_PATH}"
  ls -lh "${DB_PATH}" || true
  rm -f "${DB_PATH}"
fi

if ! db_is_valid; then
  log "Download enwiki-20230401.db"

  if [[ -f "${DB_PART}" ]]; then
    if head -c 256 "${DB_PART}" 2>/dev/null | \
       grep -Eiq '<!doctype html|<html'; then
      rm -f "${DB_PART}"
    fi
  fi

  "${PYTHON}" -m gdown \
    --continue \
    "${DB_ID}" \
    -O "${DB_PART}"

  [[ -f "${DB_PART}" ]] || \
    fail "DB temporary file was not created: ${DB_PART}"

  mv -f "${DB_PART}" "${DB_PATH}"

  if ! db_is_valid; then
    ls -lh "${DB_PATH}" 2>/dev/null || true
    file "${DB_PATH}" 2>/dev/null || true
    fail "The DB download result is not a valid SQLite DB."
  fi
else
  echo "Using existing valid DB: ${DB_PATH}"
fi

log "Run the official factscore.download_data"

FACTSCORE_TMP="$(mktemp -d)"
cd "${FACTSCORE_TMP}"

"${PYTHON}" -m factscore.download_data \
  --llama_7B_HF_path "${LLAMA_DIR}" \
  --data_dir "${DATA_DIR}" \
  --model_dir "${MODEL_DIR}"

log "Organize downloaded files into their final paths"

if [[ -d "${FACTSCORE_TMP}/demos" ]]; then
  rm -rf "${DATA_DIR}/demos"
  mv "${FACTSCORE_TMP}/demos" "${DATA_DIR}/demos"
fi

if [[ -d "${FACTSCORE_TMP}/data/labeled" ]]; then
  rm -rf "${DATA_DIR}/labeled"
  mv "${FACTSCORE_TMP}/data/labeled" "${DATA_DIR}/labeled"
fi

if [[ -d "${FACTSCORE_TMP}/data/unlabeled" ]]; then
  rm -rf "${DATA_DIR}/unlabeled"
  mv "${FACTSCORE_TMP}/data/unlabeled" "${DATA_DIR}/unlabeled"
fi

if [[ -f "${FACTSCORE_TMP}/roberta_stopwords.txt" ]]; then
  mv -f \
    "${FACTSCORE_TMP}/roberta_stopwords.txt" \
    "${DATA_DIR}/roberta_stopwords.txt"
fi

log "Verify final files"

REQUIRED_PATHS=(
  "${DATA_DIR}/demos"
  "${DATA_DIR}/labeled"
  "${DATA_DIR}/unlabeled"
  "${DATA_DIR}/roberta_stopwords.txt"
  "${DB_PATH}"
  "${MODEL_DIR}/inst-llama-7B/config.json"
)

for path in "${REQUIRED_PATHS[@]}"; do
  [[ -e "${path}" ]] || fail "Required file or directory not found: ${path}"
  echo "OK  ${path}"
done

db_is_valid || fail "Final DB validation failed."

"${PYTHON}" - "${DB_PATH}" <<'PY'
import sqlite3
import sys

path = sys.argv[1]
conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
row = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1"
).fetchone()
conn.close()

if row is None:
    raise SystemExit("The SQLite DB contains no tables.")

print("SQLite DB verification completed. First table:", row[0])
PY

echo
du -sh \
  "${LLAMA_DIR}" \
  "${DATA_DIR}/demos" \
  "${DATA_DIR}/labeled" \
  "${DATA_DIR}/unlabeled" \
  "${DB_PATH}" \
  "${MODEL_DIR}/inst-llama-7B" 2>/dev/null || true

echo
echo "FActScore download completed successfully."