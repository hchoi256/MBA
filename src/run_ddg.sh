#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/workspace/data/mba"
SRC_DIR="${BASE_DIR}/source_images"
SOURCE_MANIFEST="${BASE_DIR}/source_manifest_2k.jsonl"
DDG_EVIDENCE="${BASE_DIR}/ddg_marketdb.jsonl"
OUT_DIR="${BASE_DIR}/expert_data"
WEB_CACHE_DIR="${BASE_DIR}/web_cache_ddg"
RAW_MANIFEST_TMP="${BASE_DIR}/.source_manifest_2k_raw.tmp.jsonl"
RAW_DDG_TMP="${BASE_DIR}/.ddg_marketdb_raw.tmp.jsonl"
CANON_MANIFEST_TMP="${BASE_DIR}/.source_manifest_2k.canon.tmp.jsonl"
CANON_DDG_TMP="${BASE_DIR}/.ddg_marketdb.canon.tmp.jsonl"
PREP_SCRIPT="${PREP_SCRIPT:-/workspace/0_prepare_market_evidence.py}"
GENERATOR_SCRIPT="${GENERATOR_SCRIPT:-/workspace/1_generate_expert_trajectories_openai.py}"

echo "[VERSION] MBA_MARKETDB_FINAL_CANONICAL_SKIP_VALID_DDG_SAFE_V2"
echo "[ACTIVE] SOURCE_MANIFEST=${SOURCE_MANIFEST}"
echo "[ACTIVE] DDG_EVIDENCE=${DDG_EVIDENCE}"
echo "[ACTIVE] OUT_DIR=${OUT_DIR}"
echo "[ACTIVE] PREP_SCRIPT=${PREP_SCRIPT}"
echo "[ACTIVE] GENERATOR_SCRIPT=${GENERATOR_SCRIPT}"
export PREP_SCRIPT GENERATOR_SCRIPT

export PYTHONWARNINGS="ignore::RuntimeWarning:duckduckgo_search.*,ignore::ResourceWarning"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='sk-...'"
  exit 1
fi

if [ ! -f "${PREP_SCRIPT}" ]; then
  echo "ERROR: PREP_SCRIPT does not exist: ${PREP_SCRIPT}"
  exit 64
fi
if [ ! -f "${GENERATOR_SCRIPT}" ]; then
  echo "ERROR: GENERATOR_SCRIPT does not exist: ${GENERATOR_SCRIPT}"
  exit 64
fi

# Fail before retries or canonical-file changes when an incompatible
# 0_prepare_market_evidence.py is mounted.
PREP_HELP="$(python "${PREP_SCRIPT}" --help 2>&1 || true)"
for REQUIRED_TOKEN in   "--mode"   "ddg_prefetch"   "validate_ddg"   "--source_manifest_jsonl"   "--output_jsonl"
do
  if ! grep -Fq -- "${REQUIRED_TOKEN}" <<<"${PREP_HELP}"; then
    echo "ERROR: incompatible PREP_SCRIPT: ${PREP_SCRIPT}"
    echo "Missing CLI token in --help: ${REQUIRED_TOKEN}"
    echo "Use the DDG-prefetch implementation expected by run_ddg.sh."
    exit 64
  fi
done

DDG_MAX_IMAGE_ATTEMPTS_EFFECTIVE="${DDG_MAX_IMAGE_ATTEMPTS:-3}"
if [ "${DDG_MAX_IMAGE_ATTEMPTS_EFFECTIVE}" = "0" ]; then
  DDG_MAX_IMAGE_ATTEMPTS_EFFECTIVE=3
fi
DDG_RETRY_SLEEP_EFFECTIVE="${DDG_RETRY_SLEEP:-30}"

echo "[CONFIG] DDG_MAX_IMAGE_ATTEMPTS_EFFECTIVE=${DDG_MAX_IMAGE_ATTEMPTS_EFFECTIVE}"
echo "[CONFIG] DDG_RETRY_SLEEP_EFFECTIVE=${DDG_RETRY_SLEEP_EFFECTIVE}"
echo "[CONFIG] GENERATE_IMAGE_WORKERS=${GENERATE_IMAGE_WORKERS:-1}"
echo "[CONFIG] EVIDENCE_CONTEXT_CHARS=${EVIDENCE_CONTEXT_CHARS:-360}"
echo "[CONFIG] IDEA_MAX_OUTPUT_TOKENS=${IDEA_MAX_OUTPUT_TOKENS:-2500}"
echo "[CONFIG] OPENAI_REQUEST_TIMEOUT=${OPENAI_REQUEST_TIMEOUT:-600}"
echo "[CONFIG] OPENAI_MAX_RETRIES=${OPENAI_MAX_RETRIES:-1}"

echo "[1/5] Install/verify dependencies only if missing"
python - <<'PY'
from importlib.util import find_spec
import subprocess
import sys

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
missing = [pkg for mod, pkg in pkgs.items() if find_spec(mod) is None]
if missing:
    print("installing missing packages:", " ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", *missing])
else:
    print("all required packages already installed")
PY

mkdir -p "${BASE_DIR}" "${WEB_CACHE_DIR}" "${OUT_DIR}"

if [ "${OVERWRITE:-0}" = "1" ]; then
  echo "[OVERWRITE=1] Removing canonical MarketDB and expert outputs"
  rm -f "${SOURCE_MANIFEST}" "${DDG_EVIDENCE}" "${BASE_DIR}/ddg_marketdb.failed.jsonl"
  rm -rf "${OUT_DIR}"
  mkdir -p "${OUT_DIR}"
fi

echo "[2/5] Check whether canonical files are already generation-ready"
set +e
python - <<'PY'
import importlib.util, json, os, sys
from pathlib import Path

base = Path("/workspace/data/mba")
manifest = base / "source_manifest_2k.jsonl"
ddg = base / "ddg_marketdb.jsonl"
lenses = ["cost_resource_efficiency", "intelligent_automation", "customer_operations_growth"]

def clean(x): return "" if x is None else str(x).strip()

def read_jsonl(p):
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

if not manifest.exists() or not ddg.exists():
    print("[READY] missing canonical files")
    sys.exit(1)

rows = read_jsonl(manifest)
ddg_rows = read_jsonl(ddg)
expected = len(rows) * 3

if len(rows) == 0 or len(ddg_rows) != expected:
    print(f"[READY] count mismatch manifest={len(rows)} ddg={len(ddg_rows)} expected_ddg={expected}")
    sys.exit(1)

pairs = {(clean(r.get("sample_id")), clean(r.get("business_lens"))) for r in ddg_rows}
bad = []
for r in rows:
    sid = clean(r.get("sample_id"))
    if not sid or not all((sid, lens) in pairs for lens in lenses):
        bad.append(sid)
if bad:
    print("[READY] missing lens rows for samples:", bad[:20])
    sys.exit(1)

generator_script = os.environ.get("GENERATOR_SCRIPT", "/workspace/1_generate_expert_trajectories_openai.py")
spec = importlib.util.spec_from_file_location("gen_validator", generator_script)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
try:
    e = mod.validate_precomputed_evidence(str(ddg))
except Exception as ex:
    print("[READY] 1_generate validator failed:", ex)
    sys.exit(1)

if len(e) != expected:
    print(f"[READY] validated evidence mismatch {len(e)}/{expected}")
    sys.exit(1)

print(f"[READY] canonical files generation-ready: images={len(rows)} ddg_rows={len(ddg_rows)} ideas_expected={len(rows)*15}")
sys.exit(0)
PY
READY_STATUS=$?
set -e

if [ "${READY_STATUS}" -ne 0 ]; then
  echo "[2b/5] Canonical files not ready; build raw manifest and run bounded DDG if needed"

  cat \
    "${SRC_DIR}/general/ade20k/ade20k_mba_metadata.jsonl" \
    "${SRC_DIR}/composition_layout/rico/rico_mba_metadata.jsonl" \
    "${SRC_DIR}/spatial_human_flow/coco_person_flow/coco_person_flow_mba_metadata.jsonl" \
    "${SRC_DIR}/visual_condition_anomaly/visa/visa_mba_metadata.jsonl" \
    "${SRC_DIR}/aesthetic_texture_material/dtd/dtd_mba_metadata.jsonl" \
    "${SRC_DIR}/technical_visual_artifact/deeppcb/deeppcb_mba_metadata.jsonl" \
    > "${RAW_MANIFEST_TMP}"

  # If canonical source_manifest is missing or not raw-sized, use raw temp only for DDG attempts.
  # ddg_prefetch needs a full candidate list to retry missing rows, but final canonical files are overwritten later.
  if [ ! -f "${DDG_EVIDENCE}" ]; then
    : > "${DDG_EVIDENCE}"
  fi

  count_completed_images_raw() {
python - <<'PY'
import json
from pathlib import Path

ddg_path = Path("/workspace/data/mba/ddg_marketdb.jsonl")
source_manifest = Path("/workspace/data/mba/.source_manifest_2k_raw.tmp.jsonl")
lenses = ["cost_resource_efficiency", "intelligent_automation", "customer_operations_growth"]

def clean(x): return "" if x is None else str(x).strip()

pairs = set()
if ddg_path.exists():
    with ddg_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try: r = json.loads(line)
            except Exception: continue
            sid = clean(r.get("sample_id"))
            lens = clean(r.get("business_lens"))
            if sid and lens in lenses and len(r.get("core_retrieval_queries") or []) == 5 and len(r.get("web_evidence_cards") or []) == 5:
                pairs.add((sid, lens))

total = complete = 0
if source_manifest.exists():
    with source_manifest.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            sid = clean(r.get("sample_id"))
            total += 1
            if sid and all((sid, lens) in pairs for lens in lenses):
                complete += 1
rows = sum(1 for _ in ddg_path.open(encoding="utf-8")) if ddg_path.exists() else 0
print(f"{complete} {total} {rows}")
PY
  }

  for ATTEMPT in $(seq 1 "${DDG_MAX_IMAGE_ATTEMPTS_EFFECTIVE}"); do
    read COMPLETED TOTAL ROWS < <(count_completed_images_raw)
    REMAINING=$(( TOTAL - COMPLETED ))
    echo "[DDG SHELL ATTEMPT ${ATTEMPT}/${DDG_MAX_IMAGE_ATTEMPTS_EFFECTIVE}] completed_images=${COMPLETED}/${TOTAL} remaining_images=${REMAINING} rows=${ROWS}"
    if [ "${REMAINING}" -le 0 ]; then
      echo "[DDG] all raw images completed"
      break
    fi

    set +e
    python -u "${PREP_SCRIPT}" \
      --mode ddg_prefetch \
      --source_manifest_jsonl "${RAW_MANIFEST_TMP}" \
      --output_jsonl "${DDG_EVIDENCE}" \
      --web_cache_dir "${WEB_CACHE_DIR}" \
      --resume \
      --max_ddg_results_per_query "${MAX_DDG_RESULTS_PER_QUERY:-5}" \
      --max_variants_per_query "${MAX_VARIANTS_PER_QUERY:-4}" \
      --max_fetch_candidates_per_query "${MAX_FETCH_CANDIDATES_PER_QUERY:-4}" \
      --fetch_workers "${FETCH_WORKERS:-1}" \
      --image_workers "${DDG_IMAGE_WORKERS:-4}" \
      --web_timeout "${WEB_TIMEOUT:-30}" \
      --ddg_sleep "${DDG_SLEEP:-0.5}" \
      --fetch_sleep "${FETCH_SLEEP:-0.1}" \
      --continue_on_image_error \
      --max_image_attempts 1 \
      --retry_sleep "${DDG_RETRY_SLEEP_EFFECTIVE}" \
      --retry_sleep_backoff "${DDG_RETRY_BACKOFF:-1.0}" \
      --retry_max_sleep "${DDG_RETRY_MAX_SLEEP:-30}" \
      --retry_workers "${DDG_RETRY_WORKERS:-1}" \
      --failed_jsonl "${BASE_DIR}/ddg_marketdb.failed.jsonl" \
      --max_chars_per_page "${MAX_CHARS_PER_PAGE:-4000}" \
      --max_image_side 768 \
      --openai_model "${OPENAI_MODEL:-gpt-4o}" \
      --openai_image_detail "${OPENAI_IMAGE_DETAIL:-low}"
    STATUS=$?
    set -e

    read NEW_COMPLETED NEW_TOTAL NEW_ROWS < <(count_completed_images_raw)
    NEW_REMAINING=$(( NEW_TOTAL - NEW_COMPLETED ))
    echo "[DDG ATTEMPT RESULT] status=${STATUS} completed_images=${NEW_COMPLETED}/${NEW_TOTAL} remaining_images=${NEW_REMAINING} rows=${NEW_ROWS}"

    if [ "${STATUS}" -eq 2 ]; then
      echo "ERROR: DDG command exited with argparse status 2; refusing to retry or canonicalize."
      exit 2
    fi

    if [ "${NEW_REMAINING}" -le 0 ]; then
      echo "[DDG] all raw images completed"
      break
    fi
    if [ "${ATTEMPT}" -lt "${DDG_MAX_IMAGE_ATTEMPTS_EFFECTIVE}" ]; then
      echo "[DDG SHELL RETRY] sleep=${DDG_RETRY_SLEEP_EFFECTIVE}s before next bounded attempt"
      sleep "${DDG_RETRY_SLEEP_EFFECTIVE}"
    fi
  done

  read FINAL_COMPLETED FINAL_TOTAL FINAL_ROWS < <(count_completed_images_raw)
  if [ "${FINAL_COMPLETED}" -le 0 ]; then
    echo "ERROR: DDG produced zero complete images (${FINAL_ROWS} rows); preserving canonical files."
    exit 65
  fi

  echo "[2c/5] Overwrite canonical source_manifest_2k.jsonl and ddg_marketdb.jsonl with final clean subset"
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

lenses = ["cost_resource_efficiency", "intelligent_automation", "customer_operations_growth"]

def clean(x): return "" if x is None else str(x).strip()
def sid_of(r): return clean(r.get("sample_id"))

def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
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

generator_script = os.environ.get("GENERATOR_SCRIPT", "/workspace/1_generate_expert_trajectories_openai.py")
spec = importlib.util.spec_from_file_location("gen_validator_for_cleaning", generator_script)
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

if not final_manifest:
    raise SystemExit("no final images; canonical files were not modified")
if len(final_ddg) != len(final_manifest) * 3:
    raise SystemExit("final ddg row mismatch; canonical files were not modified")

write_jsonl(tmp_manifest, final_manifest)
write_jsonl(tmp_ddg, final_ddg)
tmp_manifest.replace(active_manifest)
tmp_ddg.replace(active_ddg)

print("raw_manifest_images:", len(manifest_rows))
print("raw_ddg_rows:", len(ddg_rows))
print("bad_evidence_removed_sample_ids:", removed_bad)
print("final_source_manifest_2k_images:", len(final_manifest))
print("final_ddg_marketdb_rows:", len(final_ddg))
print("expected_ddg_rows:", len(final_manifest) * 3)
print("expected_ideas:", len(final_manifest) * 15)

PY
else
  echo "[2b/5] DDG step skipped because canonical files are already valid"
fi

echo "[3/5] Final validation of canonical files"
EXPECTED_DDG_ROWS="$(( $(wc -l < "${SOURCE_MANIFEST}" | tr -d ' ') * 3 ))"

set +e
python -u "${PREP_SCRIPT}" \
  --mode validate_ddg \
  --output_jsonl "${DDG_EVIDENCE}" \
  --expected_rows "${EXPECTED_DDG_ROWS}"
VALIDATE_STATUS=$?
set -e
if [ "${VALIDATE_STATUS}" -ne 0 ]; then
  echo "[WARN] validate_ddg returned status=${VALIDATE_STATUS}; 1_generate validator decides final pass/fail."
fi

python - <<'PY'
import importlib.util
import os
generator_script = os.environ.get("GENERATOR_SCRIPT", "/workspace/1_generate_expert_trajectories_openai.py")
spec = importlib.util.spec_from_file_location("gen_validator", generator_script)
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