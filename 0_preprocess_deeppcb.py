#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ANNOTATION_EXTS = {".txt", ".json", ".xml", ".csv"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--deeppcb_root", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=300)
    p.add_argument("--output_image_dir", type=str, default="/workspace/data/mba/source_images/technical_visual_artifact/deeppcb/images")
    p.add_argument("--output_template_dir", type=str, default="/workspace/data/mba/source_images/technical_visual_artifact/deeppcb/templates")
    p.add_argument("--output_annotation_dir", type=str, default="/workspace/data/mba/source_images/technical_visual_artifact/deeppcb/annotations")
    p.add_argument("--metadata_jsonl", type=str, default="/workspace/data/mba/source_images/technical_visual_artifact/deeppcb/deeppcb_mba_metadata.jsonl")
    p.add_argument("--manifest_csv", type=str, default="/workspace/data/mba/source_images/technical_visual_artifact/deeppcb/deeppcb_mba_manifest.csv")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--copy_mode", choices=["copy", "hardlink", "symlink"], default="copy")
    return p.parse_args()


def mkdir_clean(path, overwrite=False):
    if overwrite and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def is_image_file(p):
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def normalize_name(s):
    s = str(s or "").lower()
    s = re.sub(r"\.(jpg|jpeg|png|bmp|webp|tif|tiff|txt|json|xml|csv)$", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def canonical_id(path):
    name = normalize_name(path.name)
    name = re.sub(
        r"(^|_)(template|temp|reference|ref|test|tested|target|image|img|gt|label|labels|anno|annot|annotation|bbox)(_|$)",
        "_",
        name,
    )
    name = re.sub(r"_+", "_", name).strip("_")
    return name or normalize_name(path.stem)


def classify_image_role(path):
    s = str(path).lower()
    name = path.name.lower()
    if any(k in s for k in ["/template", "\\template", "/templates", "\\templates", "/reference", "\\reference"]):
        return "template"
    if any(k in name for k in ["template", "temp", "reference", "ref"]):
        return "template"
    if any(k in s for k in ["/test", "\\test", "/tested", "\\tested", "/images", "\\images"]):
        return "image"
    if any(k in name for k in ["test", "tested", "target", "image", "img"]):
        return "image"
    return "unknown"


def discover_images(root):
    return sorted([p for p in root.rglob("*") if is_image_file(p)])


def discover_annotations(root):
    anns = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in ANNOTATION_EXTS:
            continue
        low = str(p).lower()
        if any(k in low for k in ["annot", "annotation", "label", "labels", "bbox", "gt", "defect"]):
            anns.append(p)
    return sorted(anns)


def build_annotation_index(root):
    idx = defaultdict(list)
    for p in discover_annotations(root):
        idx[canonical_id(p)].append(p)
        idx[normalize_name(p.stem)].append(p)
    return idx


def match_annotation(ann_idx, sample_id):
    if sample_id in ann_idx and ann_idx[sample_id]:
        return sorted(ann_idx[sample_id])[0]
    for key, vals in ann_idx.items():
        if sample_id and (sample_id in key or key in sample_id):
            return sorted(vals)[0]
    return None


def discover_pairs(root):
    imgs = discover_images(root)
    ann_idx = build_annotation_index(root)
    grouped = defaultdict(lambda: {"image": [], "template": [], "unknown": []})

    for p in imgs:
        grouped[canonical_id(p)][classify_image_role(p)].append(p)

    pairs = []
    for cid, d in grouped.items():
        image_candidates = d["image"][:]
        template_candidates = d["template"][:]
        unknown_candidates = d["unknown"][:]

        if not image_candidates and unknown_candidates:
            image_candidates.append(unknown_candidates[0])
            unknown_candidates = unknown_candidates[1:]
        if not template_candidates and unknown_candidates:
            template_candidates.append(unknown_candidates[0])
        if not image_candidates:
            continue

        pairs.append({
            "sample_id": cid,
            "image_path": sorted(image_candidates)[0],
            "template_path": sorted(template_candidates)[0] if template_candidates else None,
            "annotation_path": match_annotation(ann_idx, cid),
        })
    return pairs


def read_image_stats(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        h, w = arr.shape[:2]
        gray = arr.mean(axis=2)

        hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=True)
        hist = hist + 1e-12
        entropy = float(-(hist * np.log(hist)).sum() / math.log(256))

        color_std = float(np.std(arr.reshape(-1, 3), axis=0).mean() / 255.0)
        gx = float(np.abs(np.diff(gray, axis=1)).mean() / 255.0) if w > 1 else 0.0
        gy = float(np.abs(np.diff(gray, axis=0)).mean() / 255.0) if h > 1 else 0.0
        edge_density_proxy = (gx + gy) / 2.0
        sharpness_proxy = float(np.var(gray) / (255.0 ** 2))

        return {
            "image_width": int(w),
            "image_height": int(h),
            "pixels": int(w * h),
            "image_entropy": entropy,
            "color_std": color_std,
            "edge_density_proxy": edge_density_proxy,
            "sharpness_proxy": sharpness_proxy,
        }
    except Exception:
        return {
            "image_width": 0,
            "image_height": 0,
            "pixels": 0,
            "image_entropy": 0.0,
            "color_std": 0.0,
            "edge_density_proxy": 0.0,
            "sharpness_proxy": 0.0,
        }


def score_sample(item):
    s = item["image_stats"]
    if s["pixels"] <= 0:
        return -1e9
    score = 0.0
    score += min(s["pixels"] / (1024 * 768), 2.0) * 25.0
    score += s["image_entropy"] * 25.0
    score += s["color_std"] * 15.0
    score += s["edge_density_proxy"] * 30.0
    score += s["sharpness_proxy"] * 15.0
    score += 3.0 if item.get("template_path") else 0.0
    score += 5.0 if item.get("annotation_path") else 0.0
    return score


def copy_or_link(src, dst, mode):
    if src is None:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except Exception:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    return str(dst)


def safe_suffix(path, fallback=".jpg"):
    if path is None:
        return fallback
    return path.suffix.lower() or fallback


def main():
    args = parse_args()
    root = Path(args.deeppcb_root)
    out_img = Path(args.output_image_dir)
    out_tmp = Path(args.output_template_dir)
    out_ann = Path(args.output_annotation_dir)
    metadata_path = Path(args.metadata_jsonl)
    manifest_path = Path(args.manifest_csv)

    if not root.exists():
        raise FileNotFoundError(f"DeepPCB root not found: {root}")

    mkdir_clean(out_img, args.overwrite)
    mkdir_clean(out_tmp, args.overwrite)
    mkdir_clean(out_ann, args.overwrite)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] deeppcb_root={root}")
    pairs = discover_pairs(root)
    print(f"[INFO] candidate samples={len(pairs)}")

    if len(pairs) < args.num_samples:
        print(f"[WARN] only {len(pairs)} candidates found; selecting all available.")

    scored = []
    for item in tqdm(pairs, desc="Scoring DeepPCB samples"):
        item["image_stats"] = read_image_stats(item["image_path"])
        item["score"] = score_sample(item)
        scored.append(item)

    scored.sort(key=lambda x: x["score"], reverse=True)
    selected = scored[:args.num_samples]
    print(f"[INFO] selected={len(selected)}")

    records = []
    for rank, item in enumerate(tqdm(selected, desc="Copying selected DeepPCB files"), start=1):
        sid = item["sample_id"]
        img_dst = out_img / f"{sid}{safe_suffix(item['image_path'])}"
        tmp_dst = out_tmp / f"{sid}{safe_suffix(item['template_path'])}" if item.get("template_path") else None
        ann_dst = out_ann / f"{sid}{safe_suffix(item['annotation_path'], '.txt')}" if item.get("annotation_path") else None

        image_path = copy_or_link(item["image_path"], img_dst, args.copy_mode)
        template_path = copy_or_link(item["template_path"], tmp_dst, args.copy_mode) if tmp_dst else None
        annotation_path = copy_or_link(item["annotation_path"], ann_dst, args.copy_mode) if ann_dst else None
        stats = item["image_stats"]

        visual_annotation = {
            "technical_type": "pcb_circuit_visual_inspection",
            "source_dataset_label": "DeepPCB",
            "has_template_image": template_path is not None,
            "has_defect_annotation": annotation_path is not None,
            "visual_cues_for_ideation": [
                "PCB layout structure",
                "copper trace geometry",
                "manufacturing defect patterns",
                "hardware inspection workflow",
                "quality assurance and diagnostic automation opportunities",
            ],
        }
        prompt_context = (
            "Domain: Technical Visual Artifact / PCB inspection. "
            "This image should be interpreted as a technical visual artifact rather than a general scene. "
            "When generating a startup idea, focus on how visual patterns in PCB layouts, circuit-board defects, "
            "manufacturing quality issues, or engineering review workflows can support B2B tools such as automated "
            "inspection, hardware design review, manufacturability checking, circuit-debugging copilots, or QA automation."
        )
        rec = {
            "rank": rank,
            "sample_id": sid,
            "source_dataset": "deeppcb",
            "subset_type": "multimodal_critical",
            "domain": "technical_visual_artifact",
            "domain_display_name": "PCB Manufacturing QA",
            "image_path": image_path,
            "selected_image_path": image_path,
            "template_image_path": template_path,
            "annotation_path": annotation_path,
            "selected_annotation_path": annotation_path,
            "raw_image_path": str(item["image_path"]),
            "raw_template_path": str(item["template_path"]) if item.get("template_path") else None,
            "raw_annotation_path": str(item["annotation_path"]) if item.get("annotation_path") else None,
            "prompt_context": prompt_context,
            "visual_annotation": visual_annotation,
            "question": "Generate a startup idea grounded in this technical visual artifact and retrieved market evidence.",
            "image_width": stats["image_width"],
            "image_height": stats["image_height"],
            "pixels": stats["pixels"],
            "image_entropy": round(stats["image_entropy"], 6),
            "color_std": round(stats["color_std"], 6),
            "edge_density_proxy": round(stats["edge_density_proxy"], 6),
            "sharpness_proxy": round(stats["sharpness_proxy"], 6),
            "score": round(item["score"], 6),
        }
        records.append(rec)

    with metadata_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    fieldnames = [
        "rank", "sample_id", "source_dataset", "subset_type", "domain", "domain_display_name",
        "image_path", "template_image_path", "annotation_path", "prompt_context", "question",
        "image_width", "image_height", "pixels", "image_entropy", "color_std", "edge_density_proxy",
        "sharpness_proxy", "score", "raw_image_path", "raw_template_path", "raw_annotation_path"
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Images saved to:      {out_img}")
    print(f"Templates saved to:   {out_tmp}")
    print(f"Annotations saved to: {out_ann}")
    print(f"Metadata JSONL:       {metadata_path}")
    print(f"Manifest CSV:         {manifest_path}")
    print(f"Selected count:       {len(records)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
