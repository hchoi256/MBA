#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


POSITIVE_SCENE_KEYWORDS = {
    "airport", "terminal", "station", "subway", "train", "bus", "street", "road",
    "shop", "store", "market", "mall", "supermarket", "restaurant", "cafeteria",
    "coffee", "kitchen", "office", "classroom", "school", "library", "hospital",
    "clinic", "hotel", "lobby", "corridor", "hallway", "warehouse", "factory",
    "workshop", "gym", "bathroom", "bedroom", "living", "dining", "garage",
    "parking", "playground", "kiosk", "booth", "counter", "reception",
    "elevator", "escalator", "staircase", "crosswalk", "sidewalk", "urban",
    "plaza", "conference", "meeting", "laboratory", "lab"
}
NEGATIVE_SCENE_KEYWORDS = {
    "mountain", "forest", "sea", "ocean", "lake", "river", "waterfall",
    "desert", "field", "pasture", "sky", "cloud", "beach", "coast",
    "valley", "canyon", "glacier", "snowfield", "grassland", "underwater"
}
BUSINESS_OBJECT_KEYWORDS = {
    "person", "people", "man", "woman", "child", "chair", "seat", "bench",
    "table", "desk", "counter", "shelf", "cabinet", "case", "display",
    "screen", "monitor", "computer", "laptop", "keyboard", "phone", "tablet",
    "sign", "signboard", "poster", "board", "door", "window", "stairs",
    "escalator", "elevator", "car", "bus", "truck", "train", "bicycle",
    "motorcycle", "cart", "basket", "bottle", "cup", "plate", "food",
    "sink", "toilet", "bathtub", "bed", "sofa", "stove", "oven",
    "refrigerator", "microwave", "machine", "tool", "conveyor", "light",
    "lamp", "trash", "box", "bag", "book", "clock", "mirror"
}
LOW_INFO_OBJECT_KEYWORDS = {
    "wall", "floor", "ceiling", "sky", "earth", "ground", "grass", "tree",
    "plant", "mountain", "sea", "water", "river", "sand", "snow", "rock",
    "field", "road", "sidewalk", "building"
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ade_root", type=str, default="/workspace/data/ADEChallengeData2016")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_image_dir", type=str, default="/workspace/data/mba/source_images/general/ade20k/images")
    p.add_argument("--output_annotation_dir", type=str, default="/workspace/data/mba/source_images/general/ade20k/annotations")
    p.add_argument("--metadata_jsonl", type=str, default="/workspace/data/mba/source_images/general/ade20k/ade20k_mba_metadata.jsonl")
    p.add_argument("--manifest_csv", type=str, default="/workspace/data/mba/source_images/general/ade20k/ade20k_mba_manifest.csv")
    p.add_argument("--max_per_scene", type=int, default=25)
    p.add_argument("--copy_mode", choices=["copy", "hardlink", "symlink"], default="copy")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def normalize_text(s):
    s = str(s or "").lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9가-힣 ,/]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def keyword_hit(text, keywords):
    text = normalize_text(text)
    return any(k in text for k in keywords)


def parse_scene_categories(path):
    mapping = {}
    if not path.exists():
        print(f"[WARN] sceneCategories.txt not found: {path}")
        return mapping
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        key = parts[0]
        scene = " ".join(parts[1:])
        mapping[key] = scene
        mapping[Path(key).name] = scene
        mapping[Path(key).stem] = scene
    print(f"[INFO] loaded scene categories: {len(mapping)} keys")
    return mapping


def parse_object_info(path):
    class_map = {}
    if not path.exists():
        print(f"[WARN] objectInfo150.txt not found: {path}")
        return class_map
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cid = int(parts[0])
        except Exception:
            continue
        name = " ".join(parts[5:]) if len(parts) >= 6 else parts[-1]
        class_map[cid] = normalize_text(name)
    print(f"[INFO] loaded object classes: {len(class_map)}")
    return class_map


def image_basic_stats(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    gray = arr.mean(axis=2)
    hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=True)
    hist = hist + 1e-12
    entropy = float(-(hist * np.log(hist)).sum() / math.log(256))
    return {"image_width": int(w), "image_height": int(h), "pixels": int(w * h), "image_entropy": entropy}


def read_annotation_stats(ann_path, class_map):
    arr = np.array(Image.open(ann_path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    arr = arr.astype(np.int32)
    total = arr.size
    ids, counts = np.unique(arr, return_counts=True)

    classes, ratios = [], []
    for cid, cnt in zip(ids, counts):
        cid = int(cid)
        if cid == 0:
            continue
        ratio = float(cnt) / float(total)
        name = class_map.get(cid, f"class_{cid}")
        classes.append((cid, name, ratio))
        ratios.append(ratio)

    if len(ratios) > 1:
        p = np.array(ratios, dtype=np.float64)
        p = p / (p.sum() + 1e-12)
        entropy = float(-(p * np.log(p + 1e-12)).sum() / math.log(len(p) + 1e-12))
    else:
        entropy = 0.0
    return classes, entropy


def score_image(img_path, ann_path, scene, class_map):
    classes, seg_entropy = read_annotation_stats(ann_path, class_map)
    img_stats = image_basic_stats(img_path)

    meaningful, useful_area, business_hits, low_info_area = [], 0.0, set(), 0.0
    for cid, name, ratio in classes:
        is_low = keyword_hit(name, LOW_INFO_OBJECT_KEYWORDS)
        is_business = keyword_hit(name, BUSINESS_OBJECT_KEYWORDS)
        if ratio >= 0.001 and not is_low:
            meaningful.append((cid, name, ratio))
            useful_area += ratio
        if is_business and ratio >= 0.0005:
            business_hits.add(name)
        if is_low:
            low_info_area += ratio

    scene_norm = normalize_text(scene)
    pos_scene = keyword_hit(scene_norm, POSITIVE_SCENE_KEYWORDS)
    neg_scene = keyword_hit(scene_norm, NEGATIVE_SCENE_KEYWORDS)

    diversity = len(meaningful)
    business_count = len(business_hits)

    score = 0.0
    score += min(diversity, 20) * 1.6
    score += min(business_count, 10) * 3.0
    score += min(useful_area, 0.65) * 20.0
    score += seg_entropy * 8.0
    score += img_stats["image_entropy"] * 8.0
    if pos_scene:
        score += 10.0
    if neg_scene:
        score -= 12.0
    if diversity < 3:
        score -= 8.0
    if useful_area < 0.04:
        score -= 6.0
    if low_info_area > 0.85:
        score -= 5.0

    top_objects = sorted(meaningful, key=lambda x: x[2], reverse=True)[:15]
    visual_annotation = {
        "scene_category": scene,
        "object_summary": [
            {"class_id": cid, "name": name, "area_ratio": round(ratio, 6)}
            for cid, name, ratio in top_objects
        ],
        "ideation_relevant_cues": [
            "human environment",
            "objects and facilities",
            "possible service workflow",
            "spatial or operational improvement opportunity",
        ],
    }
    prompt_context = (
        f"Domain: General scene. Scene category: {scene}. "
        "Generate a startup idea by using visible objects, activities, spatial layout, user pain points, "
        "or service opportunities in the image. Treat this as a broad real-world scene rather than a specialized technical artifact."
    )

    rec = {
        "sample_id": img_path.stem,
        "source_dataset": "ade20k",
        "subset_type": "general",
        "domain": "general_scene",
        "domain_display_name": "General Scene",
        "scene_category": scene,
        "prompt_context": prompt_context,
        "visual_annotation": visual_annotation,
        "question": "Generate a startup idea grounded in this image and retrieved market evidence.",
        "score": round(score, 6),
        "object_diversity": diversity,
        "business_object_count": business_count,
        "useful_area_ratio": round(useful_area, 6),
        "low_info_area_ratio": round(low_info_area, 6),
        "segmentation_entropy": round(seg_entropy, 6),
        "positive_scene": bool(pos_scene),
        "negative_scene": bool(neg_scene),
        **img_stats,
    }
    return rec


def copy_or_link(src, dst, mode):
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

    ade_root = Path(args.ade_root)
    image_dir = ade_root / "images" / args.split
    ann_dir = ade_root / "annotations" / args.split
    out_img_dir = Path(args.output_image_dir)
    out_ann_dir = Path(args.output_annotation_dir)
    metadata_path = Path(args.metadata_jsonl)
    manifest_path = Path(args.manifest_csv)

    if args.overwrite:
        for p in [out_img_dir, out_ann_dir]:
            if p.exists():
                shutil.rmtree(p)
        for p in [metadata_path, manifest_path]:
            if p.exists():
                p.unlink()

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    scene_map = parse_scene_categories(ade_root / "sceneCategories.txt")
    class_map = parse_object_info(ade_root / "objectInfo150.txt")

    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png")))
    print(f"[INFO] split={args.split}")
    print(f"[INFO] input images={len(image_paths)}")
    print(f"[INFO] target samples={args.num_samples}")

    scored = []
    for img_path in tqdm(image_paths, desc="Scoring ADE images"):
        ann_path = ann_dir / f"{img_path.stem}.png"
        if not ann_path.exists():
            continue
        scene = scene_map.get(img_path.name) or scene_map.get(img_path.stem) or "unknown"
        try:
            rec = score_image(img_path, ann_path, scene, class_map)
            rec["_tie_breaker"] = random.random()
            rec["raw_image_path"] = str(img_path)
            rec["raw_annotation_path"] = str(ann_path)
            scored.append(rec)
        except Exception as e:
            print(f"[WARN] failed {img_path}: {e}")

    if len(scored) < args.num_samples:
        raise RuntimeError(f"not enough scored images: {len(scored)} < {args.num_samples}")

    scored.sort(key=lambda x: (-x["score"], -x["object_diversity"], x["_tie_breaker"]))

    selected, selected_ids, scene_counts = [], set(), Counter()
    for cap in [args.max_per_scene, args.max_per_scene * 2, 10**9]:
        for rec in scored:
            if len(selected) >= args.num_samples:
                break
            if rec["sample_id"] in selected_ids:
                continue
            scene = rec["scene_category"]
            if scene_counts[scene] >= cap:
                continue
            selected.append(rec)
            selected_ids.add(rec["sample_id"])
            scene_counts[scene] += 1
        if len(selected) >= args.num_samples:
            break
    selected = selected[:args.num_samples]

    print(f"[INFO] selected={len(selected)}")
    print("[INFO] top scene counts:")
    for scene, cnt in scene_counts.most_common(20):
        print(f"  {scene}: {cnt}")

    records = []
    for rank, rec in enumerate(tqdm(selected, desc="Copying selected files"), start=1):
        src_img = Path(rec["raw_image_path"])
        src_ann = Path(rec["raw_annotation_path"])
        dst_img = out_img_dir / src_img.name
        dst_ann = out_ann_dir / src_ann.name
        image_path = copy_or_link(src_img, dst_img, args.copy_mode)
        ann_path = copy_or_link(src_ann, dst_ann, args.copy_mode)

        rec.pop("_tie_breaker", None)
        rec["rank"] = rank
        rec["image_path"] = image_path
        rec["selected_image_path"] = image_path
        rec["annotation_path"] = ann_path
        rec["selected_annotation_path"] = ann_path
        records.append(rec)

    with metadata_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    fieldnames = [
        "rank", "sample_id", "source_dataset", "subset_type", "domain", "domain_display_name",
        "image_path", "annotation_path", "scene_category", "prompt_context", "question",
        "score", "object_diversity", "business_object_count", "useful_area_ratio",
        "segmentation_entropy", "image_width", "image_height", "pixels", "raw_image_path", "raw_annotation_path"
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Images saved to:      {out_img_dir}")
    print(f"Annotations saved to: {out_ann_dir}")
    print(f"Metadata JSONL:       {metadata_path}")
    print(f"Manifest CSV:         {manifest_path}")
    print(f"Selected count:       {len(records)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
