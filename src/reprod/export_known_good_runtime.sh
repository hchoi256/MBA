#!/usr/bin/env bash
# Run this INSIDE the old, known-good container.
# It exports custom source files and the deterministic test manifest, but omits
# the 70+ GiB FActScore databases and model weights.
set -euo pipefail

OUTPUT="${1:-/workspace/mba_known_good_overlay_$(date +%Y%m%d_%H%M%S).tar.gz}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/workspace"

# Copy small top-level pipeline/custom files.
for file in /workspace/2_sft.py /workspace/3_grpo.py /workspace/4_eval.py /workspace/wiki_fact_scorer.py; do
  if [[ -f "$file" ]]; then
    mkdir -p "$TMP$(dirname "$file")"
    cp -a "$file" "$TMP$file"
  fi
done

# Copy the FActScore source tree without its Git metadata, caches, or huge DBs.
if [[ -d /workspace/FActScore ]]; then
  tar -C / \
    --exclude='workspace/FActScore/.git' \
    --exclude='workspace/FActScore/data/*.db' \
    --exclude='workspace/FActScore/**/__pycache__' \
    -cf - workspace/FActScore | tar -C "$TMP" -xf -
fi

if [[ -f /workspace/checkpoints/grpo/grpo_test_prompts.json ]]; then
  mkdir -p "$TMP/workspace/checkpoints/grpo"
  cp -a /workspace/checkpoints/grpo/grpo_test_prompts.json \
    "$TMP/workspace/checkpoints/grpo/grpo_test_prompts.json"
fi

(
  cd "$TMP/workspace"
  find . -type f -print0 | sort -z | xargs -0 sha256sum > KNOWN_GOOD_OVERLAY.sha256
)

tar -C "$TMP" -czf "$OUTPUT" workspace
printf 'Created: %s\n' "$OUTPUT"
