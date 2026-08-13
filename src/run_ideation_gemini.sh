#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/workspace/data/mba"
SRC_DIR="${BASE_DIR}/source_images"
SOURCE_MANIFEST="${BASE_DIR}/source_manifest_2k.jsonl"
DDG_EVIDENCE="${BASE_DIR}/ddg_marketdb.jsonl"
OUT_DIR="${BASE_DIR}/expert_data"
RAW_MANIFEST_TMP="${BASE_DIR}/.source_manifest_2k_raw.tmp.jsonl"
CANON_MANIFEST_TMP="${BASE_DIR}/.source_manifest_2k.canon.tmp.jsonl"
CANON_DDG_TMP="${BASE_DIR}/.ddg_marketdb.canon.tmp.jsonl"
GEN_SCRIPT_PATH="${GEN_SCRIPT_PATH:-/workspace/1_generate_expert_trajectories_gemini.py}"
export GEN_SCRIPT_PATH

if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "ERROR: GEMINI_API_KEY is not set."
  echo "Run: export GEMINI_API_KEY='YOUR_KEY'"
  exit 1
fi

export PYTHONWARNINGS="ignore::ResourceWarning"

echo "[VERSION] MBA_GEMINI_3_5_FLASH_SLOT_LEVEL_RESUME_NO_FALLBACK_V2"
echo "[POLICY] Preserve valid OpenAI/Gemini slots; remove legacy fallback slots; retry only missing slots."
echo "[ACTIVE] SOURCE_MANIFEST=${SOURCE_MANIFEST}"
echo "[ACTIVE] DDG_EVIDENCE=${DDG_EVIDENCE}"
echo "[ACTIVE] OUT_DIR=${OUT_DIR}"
echo "[ACTIVE] GEN_SCRIPT_PATH=${GEN_SCRIPT_PATH}"
echo "[CONFIG] GEMINI_MODEL=${GEMINI_MODEL:-gemini-3.5-flash}"
echo "[CONFIG] GEMINI_THINKING_LEVEL=${GEMINI_THINKING_LEVEL:-minimal}"
echo "[CONFIG] GENERATE_IMAGE_WORKERS=${GENERATE_IMAGE_WORKERS:-1}"
echo "[CONFIG] GEMINI_MAX_RETRIES=${GEMINI_MAX_RETRIES:-2}"
echo "[CONFIG] IDEA_MAX_OUTPUT_TOKENS=${IDEA_MAX_OUTPUT_TOKENS:-1200}"
echo "[CONFIG] GEMINI_MAX_ESTIMATED_COST_USD=${GEMINI_MAX_ESTIMATED_COST_USD:-0}"

echo "[1/4] Install/verify dependencies"
python - <<'PY'
import importlib.util
import subprocess
import sys

packages = {
    "google.genai": "google-genai>=2.0.0",
    "PIL": "pillow",
    "tqdm": "tqdm",
}
missing = [pkg for module, pkg in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", *missing])
else:
    print("all required packages already installed")
PY

mkdir -p "${BASE_DIR}" "${OUT_DIR}"

if [ ! -s "${DDG_EVIDENCE}" ]; then
  echo "ERROR: ${DDG_EVIDENCE} is missing or empty. Run DDG first."
  exit 1
fi

echo "[2/4] Canonicalize source manifest and existing DDG evidence"
cat \
  "${SRC_DIR}/general/ade20k/ade20k_mba_metadata.jsonl" \
  "${SRC_DIR}/composition_layout/rico/rico_mba_metadata.jsonl" \
  "${SRC_DIR}/spatial_human_flow/coco_person_flow/coco_person_flow_mba_metadata.jsonl" \
  "${SRC_DIR}/visual_condition_anomaly/visa/visa_mba_metadata.jsonl" \
  "${SRC_DIR}/aesthetic_texture_material/dtd/dtd_mba_metadata.jsonl" \
  "${SRC_DIR}/technical_visual_artifact/deeppcb/deeppcb_mba_metadata.jsonl" \
  > "${RAW_MANIFEST_TMP}"

python - <<'PY'
import importlib.util
import json
import os
import re
from pathlib import Path

base = Path("/workspace/data/mba")
raw_manifest = base / ".source_manifest_2k_raw.tmp.jsonl"
active_manifest = base / "source_manifest_2k.jsonl"
active_ddg = base / "ddg_marketdb.jsonl"
tmp_manifest = base / ".source_manifest_2k.canon.tmp.jsonl"
tmp_ddg = base / ".ddg_marketdb.canon.tmp.jsonl"
gen_script = os.environ["GEN_SCRIPT_PATH"]
lenses = ["cost_resource_efficiency", "intelligent_automation", "customer_operations_growth"]

def clean(x):
    return "" if x is None else str(x).strip()

def sid_of(row):
    return clean(row.get("sample_id"))

def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def structurally_complete(row):
    if not sid_of(row) or clean(row.get("business_lens")) not in lenses:
        return False
    if len(row.get("core_retrieval_queries") or []) != 5:
        return False
    cards = row.get("web_evidence_cards") or []
    if len(cards) != 5:
        return False
    for card in cards:
        if not isinstance(card, dict) or not clean(card.get("url")):
            return False
        if not (
            clean(card.get("text"))
            or clean(card.get("salient_market_text"))
            or clean(card.get("snippet"))
            or clean(card.get("title"))
        ):
            return False
    return True

manifest_rows = read_jsonl(raw_manifest)
ddg_rows = read_jsonl(active_ddg)
pairs = {
    (sid_of(row), clean(row.get("business_lens")))
    for row in ddg_rows
    if structurally_complete(row)
}
complete_ids = {
    sid_of(row)
    for row in manifest_rows
    if sid_of(row) and all((sid_of(row), lens) in pairs for lens in lenses)
}
complete_ddg = [row for row in ddg_rows if sid_of(row) in complete_ids]

spec = importlib.util.spec_from_file_location("gemini_validator", gen_script)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
validator = module.validate_precomputed_evidence

current_ddg = list(complete_ddg)
removed_bad = []
for _ in range(199):
    write_jsonl(tmp_ddg, current_ddg)
    try:
        validator(str(tmp_ddg))
        break
    except Exception as exc:
        match = re.search(r"line\s+(\d+):\s+bad evidence term", str(exc))
        if not match:
            raise
        line_no = int(match.group(1))
        sample_id = sid_of(current_ddg[line_no - 1])
        current_ddg = [row for row in current_ddg if sid_of(row) != sample_id]
        removed_bad.append(sample_id)
else:
    raise RuntimeError("too many bad-evidence cleanup iterations")

lens_by_id = {}
for row in current_ddg:
    lens_by_id.setdefault(sid_of(row), set()).add(clean(row.get("business_lens")))
final_ids = {
    sample_id
    for sample_id, found_lenses in lens_by_id.items()
    if all(lens in found_lenses for lens in lenses)
}
final_manifest = [row for row in manifest_rows if sid_of(row) in final_ids]
final_ddg = [row for row in current_ddg if sid_of(row) in final_ids]
write_jsonl(tmp_manifest, final_manifest)
write_jsonl(tmp_ddg, final_ddg)
tmp_manifest.replace(active_manifest)
tmp_ddg.replace(active_ddg)

print("bad_evidence_removed_sample_ids:", removed_bad)
print("FINAL source_manifest images:", len(final_manifest))
print("FINAL ddg rows:", len(final_ddg))
print("FINAL expected ideas:", len(final_manifest) * 15)
if not final_manifest:
    raise SystemExit("No final images; aborting")
if len(final_ddg) != len(final_manifest) * 3:
    raise SystemExit("Final DDG row count mismatch")
PY

echo "[3/4] Validate canonical evidence"
python - <<'PY'
import importlib.util
import os
spec = importlib.util.spec_from_file_location("gemini_validator", os.environ["GEN_SCRIPT_PATH"])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
evidence = module.validate_precomputed_evidence("/workspace/data/mba/ddg_marketdb.jsonl")
print("[VALIDATE] passed evidence_rows=", len(evidence))
PY

if [ "${SKIP_GENERATE:-0}" = "1" ]; then
  echo "[SKIP_GENERATE=1] Stop before generation"
  wc -l "${SOURCE_MANIFEST}" "${DDG_EVIDENCE}"
  exit 0
fi

echo "[4/4] Resume missing idea slots with Gemini"
python -u "${GEN_SCRIPT_PATH}" \
  --source_manifest_jsonl "${SOURCE_MANIFEST}" \
  --output_dir "${OUT_DIR}" \
  --precomputed_web_evidence_jsonl "${DDG_EVIDENCE}" \
  --require_precomputed_web_evidence \
  --strict_web_evidence \
  --min_web_evidence_cards 5 \
  --skip_existing \
  --image_workers "${GENERATE_IMAGE_WORKERS:-1}" \
  --max_image_side 768 \
  --idea_max_output_tokens "${IDEA_MAX_OUTPUT_TOKENS:-1200}" \
  --gemini_model "${GEMINI_MODEL:-gemini-3.5-flash}" \
  --thinking_level "${GEMINI_THINKING_LEVEL:-minimal}" \
  --max_retries "${GEMINI_MAX_RETRIES:-2}" \
  --max_estimated_cost_usd "${GEMINI_MAX_ESTIMATED_COST_USD:-0}"

echo "[DONE]"
echo "Complete output: ${OUT_DIR}/*.json"
echo "Partial resume state: ${OUT_DIR}/_partial/"
echo "Failure/usage logs: ${OUT_DIR}/_state/"
