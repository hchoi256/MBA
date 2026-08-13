#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import py_compile
import sqlite3
import sys
from typing import Dict, List, Tuple

EXPECTED: Dict[str, str] = {
    "torch": "2.2.0",
    "torchvision": "0.17.0",
    "torchaudio": "2.2.0",
    "transformers": "4.45.2",
    "tokenizers": "0.20.1",
    "huggingface-hub": "0.36.2",
    "accelerate": "1.13.0",
    "peft": "0.19.1",
    "safetensors": "0.8.0",
    "bitsandbytes": "0.43.3",
    "flash-attn": "2.5.9.post1",
    "timm": "0.9.12",
    "numpy": "1.26.4",
    "Pillow": "12.2.0",
    "scipy": "1.15.3",
    "scikit-learn": "1.7.2",
    "sentence-transformers": "5.5.1",
    "faiss-cpu": "1.7.4",
    "nltk": "3.9.4",
    "spacy": "3.8.14",
    "rank-bm25": "0.2.2",
    "openai": "2.41.1",
    "einops": "0.8.2",
    "sentencepiece": "0.2.1",
    "protobuf": "6.33.6",
}

REQUIRED_PATHS = (
    "2_sft.py",
    "3_grpo.py",
    "4_eval.py",
    "wiki_fact_scorer.py",
    "FActScore/factscore/factscorer.py",
    "FActScore/factscore/atomic_facts.py",
    "FActScore/factscore/llama_lm.py",
    "FActScore/data/enwiki-20260101-fts.db",
    "mistral-7b-instruct/config.json",
    "data/mba/expert_data_verified",
    "data/ADEChallengeData2016/images/validation",
    "data/mba/mbalibrary/config.json",
    "data/mba/mbalibrary/market_evidence.faiss",
    "data/mba/mbalibrary/market_evidence_meta.jsonl",
)

BAD_DISTRIBUTIONS = (
    "faiss-gpu",
    "vllm-flash-attn",
    "nvidia-cuda-runtime",
    "nvidia-cudnn-cu13",
)


def version_of(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_hash_manifest(path: Path) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    if not path.exists():
        return entries
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        expected, rel = line.split(maxsplit=1)
        entries.append((expected, rel.strip()))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument("--expected-gpus", type=int, choices=(0, 4, 8), default=0)
    parser.add_argument("--strict-versions", action="store_true")
    parser.add_argument("--check-hashes", action="store_true")
    parser.add_argument(
        "--hash-manifest",
        default="/workspace/repro/known-good-critical.sha256",
    )
    args = parser.parse_args()

    errors: List[str] = []
    warnings: List[str] = []
    root = Path(args.workspace)

    print("== Python ==")
    print(sys.version)
    if sys.version_info[:2] != (3, 10):
        errors.append(f"Expected Python 3.10, found {sys.version.split()[0]}")

    print("\n== Package versions ==")
    for dist, expected in EXPECTED.items():
        actual = version_of(dist)
        marker = "OK" if actual == expected else "DIFF"
        print(f"{marker:4s} {dist:24s} expected={expected:14s} actual={actual}")
        if actual is None:
            errors.append(f"Missing package: {dist}")
        elif actual != expected:
            message = f"{dist}: expected {expected}, found {actual}"
            (errors if args.strict_versions else warnings).append(message)

    print("\n== CUDA/PyTorch ==")
    try:
        import torch
        print("torch:", torch.__version__)
        print("torch CUDA:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())
        print("GPU count:", torch.cuda.device_count())
        if torch.__version__.split("+")[0] != "2.2.0":
            errors.append(f"Expected torch 2.2.0, found {torch.__version__}")
        if torch.version.cuda != "12.1":
            errors.append(f"Expected torch CUDA 12.1, found {torch.version.cuda}")
        if not torch.cuda.is_available():
            errors.append("CUDA is not available")
        if args.expected_gpus and torch.cuda.device_count() != args.expected_gpus:
            errors.append(
                f"Expected {args.expected_gpus} visible GPUs, found {torch.cuda.device_count()}"
            )
        if torch.cuda.device_count() not in (0, 4, 8):
            warnings.append(
                "Pipeline production topology expects exactly 4 or 8 visible GPUs"
            )
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(
                f"GPU{idx}: {props.name}, {props.total_memory / 2**30:.1f} GiB, "
                f"compute={props.major}.{props.minor}"
            )
    except Exception as exc:
        errors.append(f"PyTorch CUDA check failed: {exc!r}")

    print("\n== Extension imports ==")
    for module_name in (
        "flash_attn",
        "bitsandbytes",
        "transformers",
        "peft",
        "timm",
        "faiss",
        "sentence_transformers",
        "spacy",
    ):
        try:
            module = importlib.import_module(module_name)
            print("OK  ", module_name, getattr(module, "__file__", ""))
        except Exception as exc:
            errors.append(f"Import failed for {module_name}: {exc!r}")

    print("\n== SQLite FTS5 ==")
    try:
        conn = sqlite3.connect(":memory:")
        options = {row[0] for row in conn.execute("pragma compile_options")}
        print("SQLite:", sqlite3.sqlite_version)
        print("ENABLE_FTS5:", "ENABLE_FTS5" in options)
        if "ENABLE_FTS5" not in options:
            errors.append("SQLite was built without FTS5")
    except Exception as exc:
        errors.append(f"SQLite check failed: {exc!r}")

    print("\n== NLTK/spaCy resources ==")
    try:
        import nltk
        for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
            try:
                print("OK  ", resource, nltk.data.find(resource))
            except LookupError:
                errors.append(f"Missing NLTK resource: {resource}")
    except Exception as exc:
        errors.append(f"NLTK resource check failed: {exc!r}")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("OK   en_core_web_sm", nlp.meta.get("version"))
    except Exception as exc:
        errors.append(f"spaCy model check failed: {exc!r}")

    print("\n== Workspace assets ==")
    for rel in REQUIRED_PATHS:
        path = root / rel
        if path.exists():
            kind = "dir" if path.is_dir() else f"{path.stat().st_size} bytes"
            print("OK  ", rel, kind)
        else:
            errors.append(f"Missing required path: {path}")

    print("\n== Pipeline compilation ==")
    for rel in ("2_sft.py", "3_grpo.py", "4_eval.py"):
        path = root / rel
        if not path.exists():
            continue
        try:
            py_compile.compile(str(path), doraise=True)
            print("OK  ", rel)
        except Exception as exc:
            errors.append(f"Compilation failed for {rel}: {exc!r}")

    print("\n== Local FActScore imports ==")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "FActScore"))
    for module_name in (
        "factscore.factscorer",
        "factscore.atomic_facts",
        "factscore.llama_lm",
        "wiki_fact_scorer",
    ):
        try:
            module = importlib.import_module(module_name)
            print("OK  ", module_name, getattr(module, "__file__", ""))
        except Exception as exc:
            errors.append(f"Local import failed for {module_name}: {exc!r}")

    print("\n== Unwanted duplicate/conflicting distributions ==")
    for dist in BAD_DISTRIBUTIONS:
        actual = version_of(dist)
        if actual is not None:
            warnings.append(f"Unnecessary/conflicting package installed: {dist}=={actual}")
            print("WARN", dist, actual)
        else:
            print("OK  ", dist, "not installed")

    if args.check_hashes:
        manifest = Path(args.hash_manifest)
        print("\n== Known-good hashes ==")
        entries = read_hash_manifest(manifest)
        if not entries:
            warnings.append(f"No hash entries found: {manifest}")
        for expected, rel in entries:
            path = root / rel
            if not path.exists():
                warnings.append(f"Hash target missing: {path}")
                continue
            actual = sha256(path)
            marker = "OK" if actual == expected else "DIFF"
            print(marker, rel, actual)
            if actual != expected:
                warnings.append(f"Hash differs from captured environment: {rel}")

    print("\n== Summary ==")
    for item in warnings:
        print("WARNING:", item)
    for item in errors:
        print("ERROR:", item)
    if errors:
        print(f"FAILED with {len(errors)} error(s) and {len(warnings)} warning(s).")
        return 1
    print(f"PASSED with {len(warnings)} warning(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
