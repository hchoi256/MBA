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
GEN_SCRIPT_PATH="${GEN_SCRIPT_PATH:-/workspace/1_generate_expert_trajectories_openai.py}"
export GEN_SCRIPT_PATH

echo "[VERSION] MBA_MARKETDB_GENERATE_ONLY_PER_IDEA_NO_FALLBACK_RESUME_SAFE"
echo "[POLICY] DDG is skipped. Existing ddg_marketdb.jsonl is canonicalized, then per-idea 1_generate runs. Failed idea generations are NOT replaced by fallback ideas; incomplete samples are retried on resume."
echo "[ACTIVE] SOURCE_MANIFEST=${SOURCE_MANIFEST}"
echo "[ACTIVE] DDG_EVIDENCE=${DDG_EVIDENCE}"
echo "[ACTIVE] OUT_DIR=${OUT_DIR}"
echo "[ACTIVE] GEN_SCRIPT_PATH=${GEN_SCRIPT_PATH}"

export PYTHONWARNINGS="ignore::RuntimeWarning:duckduckgo_search.*,ignore::ResourceWarning"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='sk-...'"
  exit 1
fi

echo "[CONFIG] GENERATE_IMAGE_WORKERS=${GENERATE_IMAGE_WORKERS:-1}"
echo "[CONFIG] EVIDENCE_CONTEXT_CHARS=${EVIDENCE_CONTEXT_CHARS:-360}"
echo "[CONFIG] IDEA_MAX_OUTPUT_TOKENS=${IDEA_MAX_OUTPUT_TOKENS:-900}"
echo "[CONFIG] OPENAI_REQUEST_TIMEOUT=${OPENAI_REQUEST_TIMEOUT:-180}"
echo "[CONFIG] OPENAI_MAX_RETRIES=${OPENAI_MAX_RETRIES:-2}"

echo "[1/4] Install/verify dependencies only if missing"
python - <<'PY'
import importlib.util, subprocess, sys
pkgs = {
    "openai": "openai",
    "PIL": "pillow",
    "tqdm": "tqdm",
    "httpx": "httpx",
    "bs4": "beautifulsoup4",
    "lxml": "lxml",
    "trafilatura": "trafilatura",
    "ddgs": "ddgs",
    "duckduckgo_search": "duckduckgo-search",
}
missing = []
for mod, pkg in pkgs.items():
    if importlib.util.find_spec(mod) is None:
        missing.append(pkg)
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U"] + missing)
else:
    print("all required packages already installed")
PY

mkdir -p "${BASE_DIR}" "${OUT_DIR}"

if [ ! -s "${DDG_EVIDENCE}" ]; then
  echo "ERROR: ${DDG_EVIDENCE} is missing or empty. Run DDG first."
  exit 1
fi

echo "[2/4] Canonicalize source_manifest_2k.jsonl and ddg_marketdb.jsonl from existing DDG only"
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

def clean(x): return "" if x is None else str(x).strip()
def sid_of(r): return clean(r.get("sample_id"))

def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def row_structurally_complete(r):
    if sid_of(r) == "" or clean(r.get("business_lens")) not in lenses:
        return False
    if len(r.get("core_retrieval_queries") or []) != 5:
        return False
    cards = r.get("web_evidence_cards") or []
    if len(cards) != 5:
        return False
    for c in cards:
        if not isinstance(c, dict):
            return False
        if not clean(c.get("url")):
            return False
        if not (clean(c.get("text")) or clean(c.get("salient_market_text")) or clean(c.get("snippet")) or clean(c.get("title"))):
            return False
    return True

manifest_rows = read_jsonl(raw_manifest)
ddg_rows = read_jsonl(active_ddg)

pairs = set()
for r in ddg_rows:
    if row_structurally_complete(r):
        pairs.add((sid_of(r), clean(r.get("business_lens"))))

complete_sids = {
    sid_of(r) for r in manifest_rows
    if sid_of(r) and all((sid_of(r), lens) in pairs for lens in lenses)
}
complete_manifest = [r for r in manifest_rows if sid_of(r) in complete_sids]
complete_ddg = [r for r in ddg_rows if sid_of(r) in complete_sids]

spec = importlib.util.spec_from_file_location("gen_validator_for_cleaning", gen_script)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
validator = mod.validate_precomputed_evidence

current_ddg = list(complete_ddg)
removed_bad = []
for step in range(1, 200):
    write_jsonl(tmp_ddg, current_ddg)
    try:
        validator(str(tmp_ddg))
        print(f"[CLEAN] 1_generate validator passed after {step-1} bad-evidence removals")
        break
    except Exception as e:
        msg = str(e)
        m = re.search(r"line\s+(\d+):\s+bad evidence term", msg)
        if not m:
            raise
        line_no = int(m.group(1))
        sid = sid_of(current_ddg[line_no - 1])
        before = len(current_ddg)
        current_ddg = [r for r in current_ddg if sid_of(r) != sid]
        removed_bad.append(sid)
        print(f"[CLEAN] removed sample_id={sid} rows_removed={before-len(current_ddg)} reason={msg}")
else:
    raise RuntimeError("too many bad-evidence cleanup iterations")

lens_by_sid = {}
for r in current_ddg:
    lens_by_sid.setdefault(sid_of(r), set()).add(clean(r.get("business_lens")))
final_sids = {sid for sid, ls in lens_by_sid.items() if all(l in ls for l in lenses)}

final_manifest = [r for r in manifest_rows if sid_of(r) in final_sids]
final_ddg = [r for r in current_ddg if sid_of(r) in final_sids]

write_jsonl(tmp_manifest, final_manifest)
write_jsonl(tmp_ddg, final_ddg)
tmp_manifest.replace(active_manifest)
tmp_ddg.replace(active_ddg)

print("raw_manifest_images:", len(manifest_rows))
print("input_ddg_rows:", len(ddg_rows))
print("bad_evidence_removed_sample_ids:", removed_bad)
print("FINAL source_manifest_2k images:", len(final_manifest))
print("FINAL ddg_marketdb rows:", len(final_ddg))
print("FINAL expected_ddg_rows:", len(final_manifest) * 3)
print("FINAL expected_ideas:", len(final_manifest) * 15)

if len(final_manifest) == 0:
    raise SystemExit("No final images; aborting.")
if len(final_ddg) != len(final_manifest) * 3:
    raise SystemExit("Final DDG row count mismatch.")
PY

echo "[3/4] Validate canonical files for 1_generate"
python - <<'PY'
import importlib.util
import os
spec = importlib.util.spec_from_file_location("gen_validator", os.environ["GEN_SCRIPT_PATH"])
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
e = mod.validate_precomputed_evidence("/workspace/data/mba/ddg_marketdb.jsonl")
print("[VALIDATE_1_GENERATE] passed evidence_rows=", len(e))
PY

if [ "${SKIP_GENERATE:-0}" = "1" ]; then
  echo "[SKIP_GENERATE=1] Stop before generation"
  wc -l "${SOURCE_MANIFEST}"
  wc -l "${DDG_EVIDENCE}"
  exit 0
fi

echo "[4/4] Generate expert data with per-idea OpenAI calls"
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
  --idea_max_output_tokens "${IDEA_MAX_OUTPUT_TOKENS:-900}" \
  --openai_model "${OPENAI_MODEL:-gpt-4o}" \
  --openai_image_detail "${OPENAI_IMAGE_DETAIL:-low}" \
  --request_timeout "${OPENAI_REQUEST_TIMEOUT:-180}" \
  --max_retries "${OPENAI_MAX_RETRIES:-2}"

echo "[DONE]"
echo "Canonical source manifest: ${SOURCE_MANIFEST}"
echo "Canonical DDG market DB: ${DDG_EVIDENCE}"
echo "Output: ${OUT_DIR}"
