#!/usr/bin/env bash
set -euo pipefail

ADE_ROOT="/workspace/data/ADEChallengeData2016"
OUT_BASE="/workspace/data/mba/source_images/general/ade20k"

python -u /workspace/0_preprocess_ade.py \
  --ade_root "${ADE_ROOT}" \
  --num_samples 500 \
  --output_image_dir "${OUT_BASE}/images" \
  --output_annotation_dir "${OUT_BASE}/annotations" \
  --metadata_jsonl "${OUT_BASE}/ade20k_mba_metadata.jsonl" \
  --manifest_csv "${OUT_BASE}/ade20k_mba_manifest.csv" \
  --overwrite

echo "DONE: ADE20K = $(find "${OUT_BASE}/images" -type f | wc -l)"
