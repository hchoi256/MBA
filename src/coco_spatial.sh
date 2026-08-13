#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT="/workspace/data/raw/coco"
OUT_BASE="/workspace/data/mba/source_images/spatial_human_flow/coco_person_flow"

mkdir -p "${RAW_ROOT}" "${OUT_BASE}"
cd "${RAW_ROOT}"

if [ ! -d "${RAW_ROOT}/val2017" ]; then
  echo "[1/5] Download COCO val2017 images"
  wget -O val2017.zip "http://images.cocodataset.org/zips/val2017.zip"
  echo "[2/5] Extract COCO val2017"
  unzip -q val2017.zip -d "${RAW_ROOT}"
  echo "[3/5] Remove image archive"
  rm -f val2017.zip
else
  echo "[Skip] COCO val2017 already extracted"
  rm -f val2017.zip || true
fi

if [ ! -f "${RAW_ROOT}/annotations/instances_val2017.json" ]; then
  echo "[4/5] Download COCO annotations"
  wget -O annotations_trainval2017.zip "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
  unzip -q annotations_trainval2017.zip -d "${RAW_ROOT}"
  rm -f annotations_trainval2017.zip
else
  echo "[Skip] COCO annotations already exist"
  rm -f annotations_trainval2017.zip || true
fi

echo "[5/5] Preprocess and sample 350 COCO spatial/human-flow images"
python -u /workspace/0_preprocess_coco_spatial.py \
  --image_dir "${RAW_ROOT}/val2017" \
  --annotation_json "${RAW_ROOT}/annotations/instances_val2017.json" \
  --num_samples 350 \
  --output_image_dir "${OUT_BASE}/images" \
  --output_annotation_dir "${OUT_BASE}/annotations" \
  --metadata_jsonl "${OUT_BASE}/coco_person_flow_mba_metadata.jsonl" \
  --manifest_csv "${OUT_BASE}/coco_person_flow_mba_manifest.csv" \
  --overwrite

echo "DONE: COCO person-focused subset = $(find "${OUT_BASE}/images" -type f | wc -l)"
