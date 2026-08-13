#!/usr/bin/env python3
"""Generate captions.jsonl beside every images/ directory under MBA source_images."""

from __future__ import annotations

import os

# PyTorch-only execution. Avoid importing an unrelated TensorFlow installation.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_ROOT = "/workspace/data/mba/source_images"
DEFAULT_MODEL = "google/paligemma2-10b-ft-docci-448"
DEFAULT_PROMPT = """
Describe the provided image in detailed and objective English.

Include all clearly visible information that may be necessary to understand the scene, including:
- objects, people, and observable actions;
- spatial relationships, scene layout, and relative positions;
- materials, textures, shapes, colors, and surface appearance;
- lighting, environmental conditions, and visual state;
- anomalies, defects, damage, wear, or unusual visual patterns;
- interface components, visible text, icons, and information hierarchy, when present;
- small or fine-grained technical features, when present.

Describe only information that is visible in the image. Do not infer hidden causes, intentions, emotions, identities, geographic locations, brands, or events unless they are directly supported by clear visual evidence. Do not propose business ideas, recommendations, solutions, or interpretations. Do not mention that you are an AI or that you are analyzing an image.

Return only one self-contained English caption as a single paragraph, with no title, bullet points, markdown, or additional explanation.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0, help="Images per dataset; 0 means all.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_image_dirs(root: Path) -> List[Path]:
    return sorted(
        path
        for path in root.rglob("images")
        if path.is_dir()
        and any(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in path.iterdir())
    )


def list_images(image_dir: Path) -> List[Path]:
    return sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def read_completed(output_path: Path) -> Dict[str, str]:
    completed: Dict[str, str] = {}
    if not output_path.exists():
        return completed
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            image_path = str(row.get("image_path", "")).strip()
            caption = str(row.get("caption", "")).strip()
            if image_path and caption:
                completed[image_path] = caption
    return completed


def to_device(batch, device: torch.device):
    moved = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif value.is_floating_point():
            moved[key] = value.to(device=device, dtype=torch.bfloat16)
        else:
            moved[key] = value.to(device=device)
    return moved


def generate_caption(
    image_path: Path,
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    # Follow the checkpoint's official inference format. The processor inserts
    # the required image-token prefix; do not put a literal <image> in the prompt.
    inputs = processor(
        text=DEFAULT_PROMPT,
        images=image,
        return_tensors="pt",
    )
    inputs = to_device(inputs, device)
    input_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids = output_ids[0, input_length:]
    caption = processor.decode(generated_ids, skip_special_tokens=True).strip()
    caption = " ".join(caption.split())
    if not caption:
        raise RuntimeError("Empty caption returned by model.")
    return caption


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if args.limit < 0 or args.max_new_tokens < 1:
        raise ValueError("Invalid --limit or --max-new-tokens value.")

    # torch 2.2.x binaries are built against NumPy 1.x and cannot reliably use NumPy 2.x.
    if int(np.__version__.split(".", 1)[0]) >= 2:
        raise RuntimeError(
            f"NumPy {np.__version__} is incompatible with this torch build. "
            "Install numpy==1.26.4 before running."
        )

    image_dirs = find_image_dirs(root)
    if not image_dirs:
        raise RuntimeError(f"No images/ directories found under {root}")

    print(f"[ROOT]  {root}")
    print(f"[MODEL] {args.model_id}")
    print(f"[FOUND] {len(image_dirs)} image directories")
    for image_dir in image_dirs:
        print(f"  - {image_dir}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device(args.device)

    processor = PaliGemmaProcessor.from_pretrained(args.model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    ).eval().to(device)

    success_count = 0
    failure_count = 0

    for image_dir in image_dirs:
        dataset_dir = image_dir.parent
        output_path = dataset_dir / "captions.jsonl"
        if args.overwrite:
            output_path.unlink(missing_ok=True)

        completed = read_completed(output_path)
        images = list_images(image_dir)
        if args.limit > 0:
            images = images[: args.limit]
        pending = [
            p for p in images
            if p.relative_to(dataset_dir).as_posix() not in completed
        ]

        print(f"\n[DATASET] {dataset_dir}")
        print(f"[OUTPUT]  {output_path}")
        print(f"[IMAGES]  total={len(images)} completed={len(images)-len(pending)} pending={len(pending)}")

        with output_path.open("a", encoding="utf-8") as f:
            for image_path in tqdm(pending, desc=dataset_dir.name, unit="image"):
                rel_path = image_path.relative_to(dataset_dir).as_posix()
                try:
                    caption = generate_caption(
                        image_path, model, processor, device, args.max_new_tokens
                    )
                except Exception as exc:
                    failure_count += 1
                    print(f"\n[ERROR] {image_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
                    continue

                f.write(json.dumps({"image_path": rel_path, "caption": caption}, ensure_ascii=False) + "\n")
                f.flush()
                success_count += 1
                print(f"\n[CAPTION] {rel_path}: {caption}")

    print(f"\n[DONE] success={success_count} failed={failure_count}")
    if failure_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
