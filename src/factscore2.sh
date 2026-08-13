set -euo pipefail

cd /workspace/FActScore/data

python -m pip install -q wikiextractor tqdm

wget -c https://dumps.wikimedia.org/enwiki/20260101/enwiki-20260101-pages-articles.xml.bz2 \
  -O enwiki-20260101-pages-articles.xml.bz2

rm -rf enwiki-20260101_extracted
python -m wikiextractor.WikiExtractor \
  --json \
  --no-templates \
  -o enwiki-20260101_extracted \
  enwiki-20260101-pages-articles.xml.bz2

python - <<'PY'
import json
from pathlib import Path

src = Path("/workspace/FActScore/data/enwiki-20260101_extracted")
out = Path("/workspace/FActScore/data/enwiki-20260101.jsonl")

with out.open("w", encoding="utf-8") as w:
    for fp in sorted(src.glob("**/wiki_*")):
        with fp.open("r", encoding="utf-8") as r:
            for line in r:
                if not line.strip():
                    continue
                x = json.loads(line)
                title = str(x.get("title", "")).strip()
                text = str(x.get("text", "")).strip()
                if title and text:
                    w.write(json.dumps({"title": title, "text": text}, ensure_ascii=False) + "\n")
PY

cd /workspace

python - <<'PY'
import sys
sys.path.insert(0, "/workspace/FActScore")

from factscore.retrieval import DocDB

db = DocDB(
    db_path="/workspace/FActScore/data/enwiki-20260101.db",
    data_path="/workspace/FActScore/data/enwiki-20260101.jsonl",
)
db.close()
PY

rm -f /workspace/FActScore/data/enwiki-20260101-pages-articles.xml.bz2
rm -f /workspace/FActScore/data/enwiki-20260101.jsonl
rm -rf /workspace/FActScore/data/enwiki-20260101_extracted
rm -rf /workspace/FActScore/data/enwiki_latest_extracted