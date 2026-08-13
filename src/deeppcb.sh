#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT="/workspace/data/raw/deeppcb"
ARCHIVE_PATH="${RAW_ROOT}/deeppcb_master.zip"
EXTRACT_ROOT="${RAW_ROOT}/DeepPCB-master"
OUT_BASE="/workspace/data/mba/source_images/technical_visual_artifact/deeppcb"

mkdir -p "${RAW_ROOT}" "${OUT_BASE}"

if [ ! -d "${EXTRACT_ROOT}" ]; then
  echo "[1/4] Download DeepPCB"
  wget -O "${ARCHIVE_PATH}" "https://github.com/tangsanli5201/DeepPCB/archive/refs/heads/master.zip"

  echo "[2/4] Extract DeepPCB"
  unzip -q "${ARCHIVE_PATH}" -d "${RAW_ROOT}"

  echo "[3/4] Remove archive"
  rm -f "${ARCHIVE_PATH}"
else
  echo "[Skip] DeepPCB already extracted: ${EXTRACT_ROOT}"
  rm -f "${ARCHIVE_PATH}" || true
fi

echo "[4/4] Preprocess and sample 100 DeepPCB images"
python -u /workspace/0_preprocess_deeppcb.py \
  --deeppcb_root "${EXTRACT_ROOT}" \
  --num_samples 100 \
  --output_image_dir "${OUT_BASE}/images" \
  --output_template_dir "${OUT_BASE}/templates" \
  --output_annotation_dir "${OUT_BASE}/annotations" \
  --metadata_jsonl "${OUT_BASE}/deeppcb_mba_metadata.jsonl" \
  --manifest_csv "${OUT_BASE}/deeppcb_mba_manifest.csv" \
  --overwrite

echo "DONE: DeepPCB = $(find "${OUT_BASE}/images" -type f | wc -l)"
