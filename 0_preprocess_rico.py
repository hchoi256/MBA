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


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MOBILE_RATIO_MIN = 1.25
MOBILE_RATIO_MAX = 2.6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rico_root", type=str, default="/workspace/data/raw/rico")
    p.add_argument("--num_samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ui_details_csv", type=str, default="/workspace/data/raw/rico/ui_details.csv")
    p.add_argument("--output_image_dir", type=str, default="/workspace/data/mba/source_images/composition_layout/rico/images")
    p.add_argument("--output_hierarchy_dir", type=str, default="/workspace/data/mba/source_images/composition_layout/rico/hierarchies")
    p.add_argument("--metadata_jsonl", type=str, default="/workspace/data/mba/source_images/composition_layout/rico/rico_mba_metadata.jsonl")
    p.add_argument("--manifest_csv", type=str, default="/workspace/data/mba/source_images/composition_layout/rico/rico_mba_manifest.csv")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--copy_mode", choices=["copy", "hardlink", "symlink"], default="copy")
    return p.parse_args()


def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s or "").lower()).strip("_")


def load_ui_details(path):
    path = Path(path)
    out = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys = [row.get(k) for k in row.keys()]
            possible_ids = []
            for v in keys:
                if v is None:
                    continue
                if str(v).isdigit():
                    possible_ids.append(str(v))
            for sid in possible_ids:
                out[sid] = row
    print(f"[INFO] loaded ui_details rows by id: {len(out)}")
    return out


def is_image_file(p):
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def discover_images(root):
    imgs = []
    for p in Path(root).rglob("*"):
        if not is_image_file(p):
            continue
        low = str(p).lower()
        if any(x in low for x in ["/source_images/", "/mba/"]):
            continue
        imgs.append(p)
    return sorted(imgs)


def build_json_index(root):
    idx = defaultdict(list)
    for p in Path(root).rglob("*.json"):
        idx[norm(p.stem)].append(p)
    return idx


def find_hierarchy(img_path, json_index):
    stem = norm(img_path.stem)
    if stem in json_index and json_index[stem]:
        return sorted(json_index[stem])[0]
    # common RICO names are numeric IDs; partial fallback
    for k, vals in json_index.items():
        if stem and (stem == k or stem in k or k in stem):
            return sorted(vals)[0]
    return None


def image_stats(img_path):
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
        edge = (gx + gy) / 2.0
        aspect_ratio = h / max(w, 1)

        return {
            "image_width": int(w),
            "image_height": int(h),
            "pixels": int(w * h),
            "image_entropy": entropy,
            "color_std": color_std,
            "edge_density_proxy": edge,
            "aspect_ratio_h_over_w": aspect_ratio,
            "mobile_like_aspect_ratio": MOBILE_RATIO_MIN <= aspect_ratio <= MOBILE_RATIO_MAX,
        }
    except Exception:
        return {
            "image_width": 0,
            "image_height": 0,
            "pixels": 0,
            "image_entropy": 0.0,
            "color_std": 0.0,
            "edge_density_proxy": 0.0,
            "aspect_ratio_h_over_w": 0.0,
            "mobile_like_aspect_ratio": False,
        }


def flatten_hierarchy(node, depth=0, out=None):
    if out is None:
        out = []
    if isinstance(node, dict):
        out.append((node, depth))
        for key in ["children", "child", "views"]:
            children = node.get(key)
            if isinstance(children, list):
                for c in children:
                    flatten_hierarchy(c, depth + 1, out)
    elif isinstance(node, list):
        for c in node:
            flatten_hierarchy(c, depth, out)
    return out


def load_hierarchy_stats(json_path):
    if json_path is None or not Path(json_path).exists():
        return {
            "hierarchy_node_count": 0,
            "clickable_count": 0,
            "text_node_count": 0,
            "max_depth": 0,
            "class_diversity": 0,
            "has_hierarchy": False,
            "top_classes": [],
            "text_examples": [],
        }

    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8", errors="ignore"))
        nodes = flatten_hierarchy(data)
        classes = []
        clickable = 0
        text_nodes = 0
        texts = []
        max_depth = 0

        for node, depth in nodes:
            max_depth = max(max_depth, depth)
            cls = node.get("class") or node.get("className") or node.get("type") or node.get("componentLabel")
            if cls:
                classes.append(str(cls))
            if bool(node.get("clickable")) or bool(node.get("longClickable")) or bool(node.get("enabled")):
                clickable += 1
            txt = node.get("text") or node.get("content-desc") or node.get("content_description") or node.get("label")
            if txt:
                t = str(txt).strip()
                if t:
                    text_nodes += 1
                    if len(texts) < 8:
                        texts.append(t[:80])

        cnt = Counter(classes)
        return {
            "hierarchy_node_count": len(nodes),
            "clickable_count": clickable,
            "text_node_count": text_nodes,
            "max_depth": max_depth,
            "class_diversity": len(cnt),
            "has_hierarchy": True,
            "top_classes": [c for c, _ in cnt.most_common(10)],
            "text_examples": texts,
        }
    except Exception:
        return {
            "hierarchy_node_count": 0,
            "clickable_count": 0,
            "text_node_count": 0,
            "max_depth": 0,
            "class_diversity": 0,
            "has_hierarchy": False,
            "top_classes": [],
            "text_examples": [],
        }


def score_sample(img_s, ui_s, detail):
    score = 0.0
    score += min(img_s["pixels"] / (1080 * 1920), 1.5) * 20.0
    score += img_s["image_entropy"] * 12.0
    score += img_s["edge_density_proxy"] * 25.0
    score += img_s["color_std"] * 10.0

    score += min(ui_s["hierarchy_node_count"], 120) * 0.25
    score += min(ui_s["clickable_count"], 40) * 0.8
    score += min(ui_s["text_node_count"], 60) * 0.45
    score += min(ui_s["class_diversity"], 25) * 0.8
    score += min(ui_s["max_depth"], 12) * 1.0

    if ui_s["has_hierarchy"]:
        score += 8.0
    if img_s["mobile_like_aspect_ratio"]:
        score += 8.0

    # Penalize splash/empty/dialog-like screens.
    if ui_s["hierarchy_node_count"] < 8:
        score -= 18.0
    if ui_s["clickable_count"] < 2:
        score -= 8.0
    if ui_s["text_node_count"] < 2:
        score -= 6.0
    if img_s["image_entropy"] < 0.25:
        score -= 10.0

    cat_text = " ".join(str(v) for v in (detail or {}).values()).lower()
    if any(k in cat_text for k in ["shopping", "finance", "health", "medical", "education", "travel", "food", "business", "productivity"]):
        score += 6.0

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


def detail_for_sample(sample_id, ui_details):
    return ui_details.get(sample_id) or ui_details.get(str(int(sample_id))) if str(sample_id).isdigit() else ui_details.get(sample_id, {})


def main():
    args = parse_args()
    random.seed(args.seed)

    root = Path(args.rico_root)
    out_img = Path(args.output_image_dir)
    out_h = Path(args.output_hierarchy_dir)
    metadata_path = Path(args.metadata_jsonl)
    manifest_path = Path(args.manifest_csv)

    if args.overwrite:
        for p in [out_img, out_h]:
            if p.exists():
                shutil.rmtree(p)
        for p in [metadata_path, manifest_path]:
            if p.exists():
                p.unlink()

    out_img.mkdir(parents=True, exist_ok=True)
    out_h.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    ui_details = load_ui_details(args.ui_details_csv)
    json_index = build_json_index(root)
    images = discover_images(root)
    print(f"[INFO] candidate images={len(images)}")
    print(f"[INFO] hierarchy json keys={len(json_index)}")

    scored = []
    for img_path in tqdm(images, desc="Scoring RICO UI screenshots"):
        sample_id = norm(img_path.stem)
        h_path = find_hierarchy(img_path, json_index)
        img_s = image_stats(img_path)
        ui_s = load_hierarchy_stats(h_path)
        detail = detail_for_sample(sample_id, ui_details) or {}
        score = score_sample(img_s, ui_s, detail)

        scored.append({
            "sample_id": sample_id,
            "raw_image_path": str(img_path),
            "raw_hierarchy_path": str(h_path) if h_path else None,
            "image_stats": img_s,
            "ui_stats": ui_s,
            "ui_details": detail,
            "score": score,
            "_tie": random.random(),
        })

    if len(scored) < args.num_samples:
        print(f"[WARN] only {len(scored)} candidates; selecting all available.")
    scored.sort(key=lambda x: (-x["score"], x["_tie"]))
    selected = scored[:args.num_samples]
    print(f"[INFO] selected={len(selected)}")

    records = []
    for rank, item in enumerate(tqdm(selected, desc="Copying selected RICO files"), start=1):
        src_img = Path(item["raw_image_path"])
        dst_img = out_img / f"{item['sample_id']}{src_img.suffix.lower()}"
        image_path = copy_or_link(src_img, dst_img, args.copy_mode)

        hierarchy_path = None
        if item["raw_hierarchy_path"]:
            src_h = Path(item["raw_hierarchy_path"])
            dst_h = out_h / f"{item['sample_id']}.json"
            hierarchy_path = copy_or_link(src_h, dst_h, args.copy_mode)

        visual_annotation = {
            "ui_type": "mobile_app_screen",
            "layout_cues": {
                "hierarchy_node_count": item["ui_stats"]["hierarchy_node_count"],
                "clickable_count": item["ui_stats"]["clickable_count"],
                "text_node_count": item["ui_stats"]["text_node_count"],
                "max_depth": item["ui_stats"]["max_depth"],
                "class_diversity": item["ui_stats"]["class_diversity"],
                "top_classes": item["ui_stats"]["top_classes"],
                "text_examples": item["ui_stats"]["text_examples"],
            },
            "visual_cues_for_ideation": [
                "information hierarchy",
                "button and CTA placement",
                "screen complexity",
                "accessibility and readability",
                "user-flow friction",
                "layout composition",
            ],
        }
        prompt_context = (
            "Domain: Composition / Layout, specifically UI/UX mobile screen design. "
            "Generate a startup idea by using visual layout, information hierarchy, CTA placement, text density, "
            "navigation friction, accessibility issues, or workflow bottlenecks visible in the screen. "
            "The idea should focus on a product or service that improves UX, app conversion, UI auditing, accessibility, or design operations."
        )
        stats = item["image_stats"]
        ui = item["ui_stats"]
        rec = {
            "rank": rank,
            "sample_id": item["sample_id"],
            "source_dataset": "rico",
            "subset_type": "multimodal_critical",
            "domain": "composition_layout",
            "domain_display_name": "Composition / Layout / UI-UX",
            "image_path": image_path,
            "selected_image_path": image_path,
            "hierarchy_path": hierarchy_path,
            "annotation_path": hierarchy_path,
            "selected_annotation_path": hierarchy_path,
            "raw_image_path": item["raw_image_path"],
            "raw_hierarchy_path": item["raw_hierarchy_path"],
            "prompt_context": prompt_context,
            "visual_annotation": visual_annotation,
            "question": "Generate a startup idea grounded in this UI screenshot and retrieved market evidence.",
            "score": round(item["score"], 6),
            "image_width": stats["image_width"],
            "image_height": stats["image_height"],
            "pixels": stats["pixels"],
            "image_entropy": round(stats["image_entropy"], 6),
            "edge_density_proxy": round(stats["edge_density_proxy"], 6),
            "color_std": round(stats["color_std"], 6),
            "hierarchy_node_count": ui["hierarchy_node_count"],
            "clickable_count": ui["clickable_count"],
            "text_node_count": ui["text_node_count"],
            "max_depth": ui["max_depth"],
            "class_diversity": ui["class_diversity"],
            "ui_details": item["ui_details"],
        }
        records.append(rec)

    with metadata_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    fieldnames = [
        "rank", "sample_id", "source_dataset", "subset_type", "domain", "domain_display_name",
        "image_path", "hierarchy_path", "prompt_context", "question", "score",
        "image_width", "image_height", "pixels", "hierarchy_node_count", "clickable_count",
        "text_node_count", "max_depth", "class_diversity", "raw_image_path", "raw_hierarchy_path"
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
    print(f"Hierarchies saved to: {out_h}")
    print(f"Metadata JSONL:       {metadata_path}")
    print(f"Manifest CSV:         {manifest_path}")
    print(f"Selected count:       {len(records)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
