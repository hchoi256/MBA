#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT="/workspace/data/raw/rico"
ARCHIVE_PATH="${RAW_ROOT}/unique_uis.tar.gz"
OUT_BASE="/workspace/data/aistar/source_images/composition_layout/rico"

mkdir -p "${RAW_ROOT}" "${OUT_BASE}"

if [ ! -f "${RAW_ROOT}/.extracted_unique_uis" ]; then
  echo "[1/5] Download RICO unique UI screenshots and hierarchies"
  wget -O "${ARCHIVE_PATH}" "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz"

  echo "[2/5] Extract RICO"
  tar -xzf "${ARCHIVE_PATH}" -C "${RAW_ROOT}"
  touch "${RAW_ROOT}/.extracted_unique_uis"

  echo "[3/5] Remove archive"
  rm -f "${ARCHIVE_PATH}"
else
  echo "[Skip] RICO already extracted: ${RAW_ROOT}"
  rm -f "${ARCHIVE_PATH}" || true
fi

echo "[4/5] Download ui_details.csv if missing"
if [ ! -f "${RAW_ROOT}/ui_details.csv" ]; then
  wget -O "${RAW_ROOT}/ui_details.csv" "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/ui_details.csv" || true
fi

echo "[5/5] Preprocess and sample 350 RICO screenshots"
python -u /workspace/0_preprocess_rico.py \
  --rico_root "${RAW_ROOT}" \
  --ui_details_csv "${RAW_ROOT}/ui_details.csv" \
  --num_samples 350 \
  --output_image_dir "${OUT_BASE}/images" \
  --output_hierarchy_dir "${OUT_BASE}/hierarchies" \
  --metadata_jsonl "${OUT_BASE}/rico_aistar_metadata.jsonl" \
  --manifest_csv "${OUT_BASE}/rico_aistar_manifest.csv" \
  --overwrite

echo "DONE: RICO = $(find "${OUT_BASE}/images" -type f | wc -l)"
