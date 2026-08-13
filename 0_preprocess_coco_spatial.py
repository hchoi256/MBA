#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


PERSON_CATEGORY_ID = 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coco_root", type=str, default="/workspace/data/raw/coco")
    p.add_argument("--image_dir", type=str, default="/workspace/data/raw/coco/val2017")
    p.add_argument("--annotation_json", type=str, default="/workspace/data/raw/coco/annotations/instances_val2017.json")
    p.add_argument("--num_samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_image_dir", type=str, default="/workspace/data/mba/source_images/spatial_human_flow/coco_person_flow/images")
    p.add_argument("--output_annotation_dir", type=str, default="/workspace/data/mba/source_images/spatial_human_flow/coco_person_flow/annotations")
    p.add_argument("--metadata_jsonl", type=str, default="/workspace/data/mba/source_images/spatial_human_flow/coco_person_flow/coco_person_flow_mba_metadata.jsonl")
    p.add_argument("--manifest_csv", type=str, default="/workspace/data/mba/source_images/spatial_human_flow/coco_person_flow/coco_person_flow_mba_manifest.csv")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--copy_mode", choices=["copy", "hardlink", "symlink"], default="copy")
    return p.parse_args()


def image_stats(path):
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        h, w = arr.shape[:2]
        gray = arr.mean(axis=2)
        hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=True)
        hist = hist + 1e-12
        entropy = float(-(hist * np.log(hist)).sum() / math.log(256))
        gx = float(np.abs(np.diff(gray, axis=1)).mean() / 255.0) if w > 1 else 0.0
        gy = float(np.abs(np.diff(gray, axis=0)).mean() / 255.0) if h > 1 else 0.0
        edge = (gx + gy) / 2.0
        color_std = float(np.std(arr.reshape(-1, 3), axis=0).mean() / 255.0)
        return {
            "image_width": int(w),
            "image_height": int(h),
            "pixels": int(w * h),
            "image_entropy": entropy,
            "edge_density_proxy": edge,
            "color_std": color_std,
        }
    except Exception:
        return {
            "image_width": 0,
            "image_height": 0,
            "pixels": 0,
            "image_entropy": 0.0,
            "edge_density_proxy": 0.0,
            "color_std": 0.0,
        }


def score_sample(item):
    stats = item["image_stats"]
    person_count = item["person_count"]
    object_diversity = item["object_diversity"]
    total_objects = item["total_object_count"]
    area_cov = item["person_area_ratio"]

    score = 0.0
    score += min(person_count, 18) * 5.0
    score += min(total_objects, 40) * 0.8
    score += min(object_diversity, 15) * 2.0
    score += min(area_cov, 0.65) * 25.0
    score += min(stats["pixels"] / (640 * 480), 2.0) * 8.0
    score += stats["image_entropy"] * 10.0
    score += stats["edge_density_proxy"] * 15.0

    if person_count < 2:
        score -= 20.0
    if person_count >= 5:
        score += 15.0
    if object_diversity >= 5:
        score += 8.0

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


def main():
    args = parse_args()
    random.seed(args.seed)

    image_dir = Path(args.image_dir)
    ann_json = Path(args.annotation_json)
    out_img = Path(args.output_image_dir)
    out_ann = Path(args.output_annotation_dir)
    metadata_path = Path(args.metadata_jsonl)
    manifest_path = Path(args.manifest_csv)

    if args.overwrite:
        for p in [out_img, out_ann]:
            if p.exists():
                shutil.rmtree(p)
        for p in [metadata_path, manifest_path]:
            if p.exists():
                p.unlink()

    out_img.mkdir(parents=True, exist_ok=True)
    out_ann.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] loading COCO annotations: {ann_json}")
    data = json.loads(ann_json.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    images = {im["id"]: im for im in data.get("images", [])}

    anns_by_img = defaultdict(list)
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            # keep crowd regions too; still relevant for flow, but count separately in metadata
            pass
        anns_by_img[ann["image_id"]].append(ann)

    scored = []
    for image_id, im in tqdm(images.items(), desc="Scoring COCO spatial/human-flow images"):
        anns = anns_by_img.get(image_id, [])
        if not anns:
            continue

        file_name = im["file_name"]
        img_path = image_dir / file_name
        if not img_path.exists():
            continue

        person_anns = [a for a in anns if a.get("category_id") == PERSON_CATEGORY_ID]
        if len(person_anns) < 2:
            continue

        w = im.get("width", 1)
        h = im.get("height", 1)
        img_area = max(float(w * h), 1.0)
        person_area = sum(float(a.get("area", 0.0)) for a in person_anns)
        object_names = [cats.get(a.get("category_id"), f"cat_{a.get('category_id')}") for a in anns]
        object_counts = Counter(object_names)

        stats = image_stats(img_path)
        item = {
            "image_id": image_id,
            "file_name": file_name,
            "raw_image_path": str(img_path),
            "raw_annotations": anns,
            "person_count": len(person_anns),
            "crowd_region_count": sum(1 for a in anns if a.get("iscrowd", 0) == 1),
            "total_object_count": len(anns),
            "object_diversity": len(object_counts),
            "object_summary": dict(object_counts.most_common(15)),
            "person_area_ratio": person_area / img_area,
            "image_stats": stats,
            "_tie": random.random(),
        }
        item["score"] = score_sample(item)
        scored.append(item)

    if len(scored) < args.num_samples:
        print(f"[WARN] only {len(scored)} candidates found; selecting all available.")

    scored.sort(key=lambda x: (-x["score"], x["_tie"]))
    selected = scored[:args.num_samples]
    print(f"[INFO] selected={len(selected)}")

    records = []
    for rank, item in enumerate(tqdm(selected, desc="Copying selected COCO files"), start=1):
        src_img = Path(item["raw_image_path"])
        sample_id = f"coco_{item['image_id']:012d}"
        dst_img = out_img / src_img.name
        image_path = copy_or_link(src_img, dst_img, args.copy_mode)

        ann_path = out_ann / f"{sample_id}.json"
        ann_obj = {
            "image_id": item["image_id"],
            "file_name": item["file_name"],
            "annotations": item["raw_annotations"],
        }
        ann_path.write_text(json.dumps(ann_obj, ensure_ascii=False), encoding="utf-8")

        stats = item["image_stats"]
        visual_annotation = {
            "spatial_type": "human_flow_public_scene",
            "person_count": item["person_count"],
            "crowd_region_count": item["crowd_region_count"],
            "object_summary": item["object_summary"],
            "visual_cues_for_ideation": [
                "human density",
                "crowd movement",
                "queue or congestion risk",
                "public-space operations",
                "safety and navigation",
                "service bottlenecks",
            ],
        }
        prompt_context = (
            "Domain: Spatial / Human Flow. "
            "Generate a startup idea by using visible people, crowd density, object layout, public-space flow, queueing, "
            "congestion, safety, navigation, or service bottlenecks in the image. "
            "The idea should target spatial analytics, facility operations, retail/transit/hospital flow, safety monitoring, or customer experience."
        )

        rec = {
            "rank": rank,
            "sample_id": sample_id,
            "source_dataset": "coco2017",
            "subset_type": "multimodal_critical",
            "domain": "spatial_human_flow",
            "domain_display_name": "Spatial / Human Flow",
            "image_path": image_path,
            "selected_image_path": image_path,
            "annotation_path": str(ann_path),
            "selected_annotation_path": str(ann_path),
            "raw_image_path": item["raw_image_path"],
            "prompt_context": prompt_context,
            "visual_annotation": visual_annotation,
            "question": "Generate a startup idea grounded in this spatial human-flow image and retrieved market evidence.",
            "person_count": item["person_count"],
            "crowd_region_count": item["crowd_region_count"],
            "total_object_count": item["total_object_count"],
            "object_diversity": item["object_diversity"],
            "person_area_ratio": round(item["person_area_ratio"], 6),
            "score": round(item["score"], 6),
            "image_width": stats["image_width"],
            "image_height": stats["image_height"],
            "pixels": stats["pixels"],
            "image_entropy": round(stats["image_entropy"], 6),
            "edge_density_proxy": round(stats["edge_density_proxy"], 6),
            "color_std": round(stats["color_std"], 6),
        }
        records.append(rec)

    with metadata_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    fieldnames = [
        "rank", "sample_id", "source_dataset", "subset_type", "domain", "domain_display_name",
        "image_path", "annotation_path", "prompt_context", "question", "person_count",
        "crowd_region_count", "total_object_count", "object_diversity", "person_area_ratio",
        "score", "image_width", "image_height", "pixels", "raw_image_path"
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
    print(f"Annotations saved to: {out_ann}")
    print(f"Metadata JSONL:       {metadata_path}")
    print(f"Manifest CSV:         {manifest_path}")
    print(f"Selected count:       {len(records)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
