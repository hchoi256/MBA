"""
Step 1: Build the FTS5 index once
Run: python build_fts.py
Output: /workspace/FActScore/data/enwiki-20260101-fts.db
Estimated time: approximately 30–60 minutes (for 21 GB)
"""
import sqlite3
import time
import os

# ── Path configuration ────────────────────────────────────────
SRC_DB  = "/workspace/FActScore/data/enwiki-20260101.db"
DST_DB  = "/workspace/FActScore/data/enwiki-20260101-fts.db"

# Automatically load SPECIAL_SEPARATOR from retrieval.py
import importlib.util, sys
_ret_path = None
for p in sys.path:
    candidate = os.path.join(p, "factscore", "retrieval.py")
    if os.path.exists(candidate):
        _ret_path = candidate
        break
if _ret_path:
    spec = importlib.util.spec_from_file_location("_ret", _ret_path)
    _ret = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_ret)
    SPECIAL_SEPARATOR = _ret.SPECIAL_SEPARATOR
    print(f"[INFO] SPECIAL_SEPARATOR = {repr(SPECIAL_SEPARATOR)}")
else:
    SPECIAL_SEPARATOR = "\n\n"   # fallback
    print(f"[WARN] retrieval.py not found. fallback SPECIAL_SEPARATOR = {repr(SPECIAL_SEPARATOR)}")

# ── Build the FTS5 index ─────────────────────────────────────
if os.path.exists(DST_DB):
    print(f"[SKIP] {DST_DB} already exists. Delete it and run again.")
    exit(0)

src = sqlite3.connect(SRC_DB)
dst = sqlite3.connect(DST_DB)

dst.execute("PRAGMA journal_mode=OFF")
dst.execute("PRAGMA synchronous=OFF")
dst.execute("PRAGMA page_size=4096")

dst.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts
    USING fts5(title UNINDEXED, para, tokenize='porter unicode61')
""")

print("[INFO] Build started...")
t0 = time.time()
cur = src.execute("SELECT title, text FROM documents")
batch, total_para, total_doc = [], 0, 0
BATCH_SIZE = 10000

for title, text in cur:
    total_doc += 1
    for para in text.split(SPECIAL_SEPARATOR):
        para = para.strip()
        if len(para) > 20:   # Exclude fragments that are too short
            batch.append((title, para))
            total_para += 1
    if len(batch) >= BATCH_SIZE:
        dst.executemany("INSERT INTO passages_fts(title, para) VALUES (?,?)", batch)
        dst.commit()
        batch = []
        elapsed = time.time() - t0
        print(f"  [{elapsed:.0f}s] doc={total_doc:,} para={total_para:,}", flush=True)

if batch:
    dst.executemany("INSERT INTO passages_fts(title, para) VALUES (?,?)", batch)
    dst.commit()

src.close()
dst.close()

elapsed = time.time() - t0
print(f"\n[DONE] {total_doc:,} documents / {total_para:,} passages")
print(f"[DONE] Elapsed time: {elapsed/60:.1f} minutes")
print(f"[DONE] Saved to: {DST_DB}")