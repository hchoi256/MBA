#!/usr/bin/env python3
# 0_preprocess_dtd.py
#
# DTD (Describable Textures Dataset) -> mbaBench aesthetic_texture_material domain.
#
# Replaces SciPostLayout when the goal is visual aesthetics, texture, material feel,
# surface pattern, color/form composition, and abstract visual mood.

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

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dtd_root', type=str, default='/workspace/data/raw/dtd/dtd')
    p.add_argument('--num_samples', type=int, default=300)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_image_dir', type=str, default='/workspace/data/mba/source_images/aesthetic_texture_material/dtd/images')
    p.add_argument('--output_annotation_dir', type=str, default='/workspace/data/mba/source_images/aesthetic_texture_material/dtd/annotations')
    p.add_argument('--metadata_jsonl', type=str, default='/workspace/data/mba/source_images/aesthetic_texture_material/dtd/dtd_mba_metadata.jsonl')
    p.add_argument('--manifest_csv', type=str, default='/workspace/data/mba/source_images/aesthetic_texture_material/dtd/dtd_mba_manifest.csv')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--copy_mode', choices=['copy', 'hardlink', 'symlink'], default='copy')
    p.add_argument('--max_per_attribute', type=int, default=12)
    return p.parse_args()


def norm(x):
    return re.sub(r'[^a-z0-9]+', '_', str(x or '').lower()).strip('_')


def is_image(p: Path):
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def find_image_root(root: Path):
    candidates = [root / 'images', root]
    for c in candidates:
        if c.exists() and any(is_image(p) for p in c.rglob('*')):
            return c
    raise FileNotFoundError(f'Could not find DTD image folder under: {root}')


def image_stats(path: Path):
    try:
        img = Image.open(path).convert('RGB')
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
        aspect = w / max(h, 1)
        local_contrast = float(np.std(gray) / 255.0)
        return {
            'image_width': int(w),
            'image_height': int(h),
            'pixels': int(w * h),
            'image_entropy': entropy,
            'color_std': color_std,
            'edge_density_proxy': edge,
            'local_contrast_proxy': local_contrast,
            'aspect_ratio_w_over_h': aspect,
        }
    except Exception:
        return {
            'image_width': 0, 'image_height': 0, 'pixels': 0,
            'image_entropy': 0.0, 'color_std': 0.0,
            'edge_density_proxy': 0.0, 'local_contrast_proxy': 0.0,
            'aspect_ratio_w_over_h': 0.0,
        }


def score_sample(stats):
    if stats['pixels'] <= 0:
        return -1e9
    score = 0.0
    score += min(stats['pixels'] / (512 * 512), 2.0) * 18.0
    score += stats['image_entropy'] * 30.0
    score += stats['color_std'] * 25.0
    score += stats['edge_density_proxy'] * 30.0
    score += stats['local_contrast_proxy'] * 20.0
    if stats['pixels'] < 256 * 256:
        score -= 25.0
    if stats['image_entropy'] < 0.15:
        score -= 20.0
    return score


def copy_or_link(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == 'copy':
        shutil.copy2(src, dst)
    elif mode == 'hardlink':
        try:
            dst.hardlink_to(src)
        except Exception:
            shutil.copy2(src, dst)
    elif mode == 'symlink':
        dst.symlink_to(src.resolve())
    return str(dst)


def discover_samples(root: Path):
    image_root = find_image_root(root)
    rows = []
    for p in image_root.rglob('*'):
        if not is_image(p):
            continue
        rel = p.relative_to(image_root)
        attr = norm(rel.parts[0]) if len(rel.parts) >= 2 else norm(p.parent.name)
        rows.append({'raw_image_path': str(p), 'texture_attribute': attr})
    return rows, image_root


def select_balanced(scored, num_samples, max_per_attribute):
    selected, counts, used = [], Counter(), set()
    for item in scored:
        if len(selected) >= num_samples:
            break
        if item['raw_image_path'] in used:
            continue
        if counts[item['texture_attribute']] >= max_per_attribute:
            continue
        selected.append(item)
        used.add(item['raw_image_path'])
        counts[item['texture_attribute']] += 1
    for item in scored:
        if len(selected) >= num_samples:
            break
        if item['raw_image_path'] in used:
            continue
        selected.append(item)
        used.add(item['raw_image_path'])
        counts[item['texture_attribute']] += 1
    return selected[:num_samples], counts


def main():
    args = parse_args()
    random.seed(args.seed)
    root = Path(args.dtd_root)
    if not root.exists() and Path('/workspace/data/raw/dtd/dtd').exists():
        root = Path('/workspace/data/raw/dtd/dtd')
    if not root.exists():
        raise FileNotFoundError(f'DTD root not found: {root}')

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

    samples, image_root = discover_samples(root)
    print(f'[INFO] dtd_root={root}')
    print(f'[INFO] image_root={image_root}')
    print(f'[INFO] discovered source images={len(samples)}')
    if not samples:
        raise RuntimeError('No DTD images discovered. Check extracted directory structure.')

    scored = []
    for item in tqdm(samples, desc='Scoring DTD texture/material images'):
        stats = image_stats(Path(item['raw_image_path']))
        item['image_stats'] = stats
        item['score'] = score_sample(stats)
        item['_tie'] = random.random()
        scored.append(item)

    scored.sort(key=lambda x: (-x['score'], x['_tie']))
    selected, attr_counts = select_balanced(scored, args.num_samples, args.max_per_attribute)
    print(f'[INFO] selected={len(selected)}')
    print('[INFO] selected texture attribute counts:')
    for k, v in attr_counts.most_common():
        print(f'  {k}: {v}')

    records = []
    for rank, item in enumerate(tqdm(selected, desc='Copying selected DTD files'), start=1):
        raw = Path(item['raw_image_path'])
        attr = item['texture_attribute']
        sample_id = f'dtd_{attr}_{norm(raw.stem)}'
        dst_img = out_img / f'{sample_id}{raw.suffix.lower()}'
        image_path = copy_or_link(raw, dst_img, args.copy_mode)

        ann_path = out_ann / f'{sample_id}.json'
        s = item['image_stats']
        ann_obj = {
            'source_dataset': 'dtd',
            'raw_image_path': str(raw),
            'texture_attribute': attr,
            'image_stats': s,
        }
        ann_path.write_text(json.dumps(ann_obj, ensure_ascii=False), encoding='utf-8')

        visual_annotation = {
            'aesthetic_type': 'texture_material_semantics',
            'texture_type': attr,
            'texture_attribute': attr,
            'material_type': attr,
            'visual_cues_for_ideation': [
                'texture pattern', 'material feel', 'surface quality', 'color harmony',
                'abstract visual mood', 'tactile impression',
                'visual branding or product-design cue',
                'fashion, interior, UI theme, or digital asset opportunity',
            ],
        }
        prompt_context = (
            'Domain: Aesthetic / Texture / Material Semantics. '
            'Generate a startup idea using visual texture, material feel, surface pattern, color harmony, abstract form, '
            'tactile impression, or aesthetic mood visible in the image. '
            'The idea may target branding, digital assets, generative design, material/product design, fashion, interior mood, '
            'UI themes, creative marketplaces, or aesthetic personalization.'
        )
        rec = {
            'rank': rank,
            'sample_id': sample_id,
            'source_dataset': 'dtd',
            'subset_type': 'multimodal_critical',
            'domain': 'aesthetic_texture_material',
            'domain_display_name': 'Aesthetic / Texture / Material Semantics',
            'image_path': image_path,
            'selected_image_path': image_path,
            'annotation_path': str(ann_path),
            'selected_annotation_path': str(ann_path),
            'raw_image_path': str(raw),
            'prompt_context': prompt_context,
            'visual_annotation': visual_annotation,
            'question': 'Generate a startup idea grounded in this aesthetic texture/material image and retrieved market evidence.',
            'texture_attribute': attr,
            'score': round(item['score'], 6),
            'image_width': s['image_width'],
            'image_height': s['image_height'],
            'pixels': s['pixels'],
            'image_entropy': round(s['image_entropy'], 6),
            'edge_density_proxy': round(s['edge_density_proxy'], 6),
            'color_std': round(s['color_std'], 6),
            'local_contrast_proxy': round(s['local_contrast_proxy'], 6),
        }
        records.append(rec)

    with metadata_path.open('w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    fieldnames = [
        'rank', 'sample_id', 'source_dataset', 'subset_type', 'domain', 'domain_display_name',
        'image_path', 'annotation_path', 'texture_attribute', 'prompt_context', 'question',
        'score', 'image_width', 'image_height', 'pixels', 'raw_image_path'
    ]
    with manifest_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, '') for k in fieldnames})

    print('\n' + '=' * 80)
    print('DONE')
    print('=' * 80)
    print(f'Images saved to:      {out_img}')
    print(f'Annotations saved to: {out_ann}')
    print(f'Metadata JSONL:       {metadata_path}')
    print(f'Manifest CSV:         {manifest_path}')
    print(f'Selected count:       {len(records)}')
    print('=' * 80)


if __name__ == '__main__':
    main()
