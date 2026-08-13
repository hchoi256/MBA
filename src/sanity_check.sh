echo "=== expert JSON ==="
find /workspace/data/mba/expert_data \
  -maxdepth 1 -type f -name '*.json' | wc -l
# 1999

echo "=== combined expert JSONL ==="
wc -l /workspace/data/mba/expert_data/all_expert_trajectories.jsonl
# 1999

echo "=== manifest ==="
wc -l /workspace/data/mba/source_manifest_2k.jsonl
# 1999

echo "=== DDG ==="
wc -l /workspace/data/mba/ddg_marketdb.jsonl
# 5997

echo "=== images ==="
find /workspace/data/mba/source_images \
  -type f \( \
    -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o \
    -iname '*.webp' -o -iname '*.bmp' -o -iname '*.tif' -o \
    -iname '*.tiff' \
  \) | wc -l
# 1999

echo "=== captions ==="
find /workspace/data/mba/source_images \
  -name captions.jsonl -print0 |
xargs -0 cat |
grep -cve '^[[:space:]]*$'
# 1999