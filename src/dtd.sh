#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT="/workspace/data/raw/dtd"
ARCHIVE_PATH="${RAW_ROOT}/dtd-r1.0.1.tar.gz"
DTD_DIR="${RAW_ROOT}/dtd"
OUT_BASE="/workspace/data/mba/source_images/aesthetic_texture_material/dtd"

mkdir -p "${RAW_ROOT}" "${OUT_BASE}"

echo "[1/4] Download DTD dataset if needed"
if [ ! -f "${RAW_ROOT}/.extracted_dtd" ]; then
  if [ ! -f "${ARCHIVE_PATH}" ]; then
    wget -O "${ARCHIVE_PATH}" "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz" \
    || wget -O "${ARCHIVE_PATH}" "http://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
  else
    echo "[Skip] archive already exists: ${ARCHIVE_PATH}"
  fi
  echo "[2/4] Extract DTD"
  tar -xzf "${ARCHIVE_PATH}" -C "${RAW_ROOT}"
  touch "${RAW_ROOT}/.extracted_dtd"
  echo "[3/4] Remove archive"
  rm -f "${ARCHIVE_PATH}"
else
  echo "[Skip] DTD already extracted: ${DTD_DIR}"
  rm -f "${ARCHIVE_PATH}" || true
fi

echo "[4/4] Preprocess and sample 350 DTD texture/material images"
python -u /workspace/0_preprocess_dtd.py \
  --dtd_root "${DTD_DIR}" \
  --num_samples 350 \
  --output_image_dir "${OUT_BASE}/images" \
  --output_annotation_dir "${OUT_BASE}/annotations" \
  --metadata_jsonl "${OUT_BASE}/dtd_mba_metadata.jsonl" \
  --manifest_csv "${OUT_BASE}/dtd_mba_manifest.csv" \
  --overwrite

echo "DONE: DTD = $(find "${OUT_BASE}/images" -type f | wc -l)"
