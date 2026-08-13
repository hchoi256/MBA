#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT="/workspace/data/raw/visa"
ARCHIVE_PATH="${RAW_ROOT}/VisA_20220922.tar"
OUT_BASE="/workspace/data/mba/source_images/visual_condition_anomaly/visa"

mkdir -p "${RAW_ROOT}" "${OUT_BASE}"

echo "[1/4] Download VisA dataset if needed"
if [ ! -f "${RAW_ROOT}/.extracted_visa" ]; then
  if [ ! -f "${ARCHIVE_PATH}" ]; then
    wget -O "${ARCHIVE_PATH}" "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
  else
    echo "[Skip] archive already exists: ${ARCHIVE_PATH}"
  fi

  echo "[2/4] Extract VisA"
  tar -xf "${ARCHIVE_PATH}" -C "${RAW_ROOT}"
  touch "${RAW_ROOT}/.extracted_visa"

  echo "[3/4] Remove archive"
  rm -f "${ARCHIVE_PATH}"
else
  echo "[Skip] VisA already extracted: ${RAW_ROOT}"
  rm -f "${ARCHIVE_PATH}" || true
fi

echo "[4/4] Preprocess: category/status-balanced 350 VisA images"
python -u /workspace/0_preprocess_visa.py \
  --visa_root "${RAW_ROOT}" \
  --num_samples 350 \
  --prefer_anomaly_ratio 0.85 \
  --dedup_hamming 4 \
  --output_image_dir "${OUT_BASE}/images" \
  --output_mask_dir "${OUT_BASE}/masks" \
  --metadata_jsonl "${OUT_BASE}/visa_mba_metadata.jsonl" \
  --manifest_csv "${OUT_BASE}/visa_mba_manifest.csv" \
  --overwrite

echo "DONE: VisA = $(find "${OUT_BASE}/images" -type f | wc -l)"
