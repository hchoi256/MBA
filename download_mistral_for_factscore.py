#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_REPO = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_TARGET = "/workspace/mistral-7b-instruct"

COMMON_FILES = [
    "README.md",
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]

SAFETENSOR_FILES = [
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "model.safetensors.index.json",
]

PYTORCH_BIN_FILES = [
    "pytorch_model-00001-of-00003.bin",
    "pytorch_model-00002-of-00003.bin",
    "pytorch_model-00003-of-00003.bin",
    "pytorch_model.bin.index.json",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument("--target-dir", default=DEFAULT_TARGET)
    p.add_argument(
        "--safetensors-only",
        action="store_true",
        help="Download only safetensors weights plus tokenizer/config files.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    target = Path(args.target_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    allow_patterns = COMMON_FILES + SAFETENSOR_FILES
    if not args.safetensors_only:
        allow_patterns += PYTORCH_BIN_FILES

    print(f"[Mistral] repo   : {args.repo_id}")
    print(f"[Mistral] target : {target}")
    print(
        "[Mistral] weights: "
        + ("safetensors only" if args.safetensors_only else "safetensors + PyTorch .bin")
    )

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(target),
        allow_patterns=allow_patterns,
    )

    required = COMMON_FILES + SAFETENSOR_FILES
    if not args.safetensors_only:
        required += PYTORCH_BIN_FILES

    missing = [name for name in required if not (target / name).is_file()]
    if missing:
        raise RuntimeError("Missing files: " + ", ".join(missing))

    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(str(target), local_files_only=True)
    tok = AutoTokenizer.from_pretrained(str(target), local_files_only=True)

    print()
    print("[Mistral] validation passed")
    print("  model_type:", getattr(cfg, "model_type", "<unknown>"))
    print("  tokenizer :", type(tok).__name__)
    print("  path      :", target)
    print()
    print("MISTRAL DOWNLOAD COMPLETE")


if __name__ == "__main__":
    main()
