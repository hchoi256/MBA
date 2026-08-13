#!/usr/bin/env python3
# VisA -> MBA-Bench visual_condition_anomaly
# Robust category parsing + category/status-stratified round-robin + visual deduplication.

import argparse
import csv
import hashlib
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

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
MASK_HINTS = {'mask', 'masks', 'gt', 'ground_truth', 'label', 'labels', 'segmentation'}
NORMAL_HINTS = {'good', 'normal', 'ok'}
ANOMALY_HINTS = {'bad', 'anomaly', 'anomalous', 'defect', 'defective'}
IMAGE_DIR_HINTS = {'image', 'images', 'img', 'imgs'}
MEDIA_DIR_HINTS = IMAGE_DIR_HINTS | {'mask', 'masks'}
WRAPPER_DIR_HINTS = {
    'data', 'dataset', 'datasets', 'raw', 'raw_data', 'image_data',
    'visa', 'visa_20220922', 'visa20220922',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--visa_root', default='/workspace/data/raw/visa')
    p.add_argument('--num_samples', type=int, default=300)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_image_dir', default='/workspace/data/mba/source_images/visual_condition_anomaly/visa/images')
    p.add_argument('--output_mask_dir', default='/workspace/data/mba/source_images/visual_condition_anomaly/visa/masks')
    p.add_argument('--metadata_jsonl', default='/workspace/data/mba/source_images/visual_condition_anomaly/visa/visa_mba_metadata.jsonl')
    p.add_argument('--manifest_csv', default='/workspace/data/mba/source_images/visual_condition_anomaly/visa/visa_mba_manifest.csv')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--copy_mode', choices=['copy', 'hardlink', 'symlink'], default='copy')
    p.add_argument('--prefer_anomaly_ratio', type=float, default=0.85)
    p.add_argument(
        '--dedup_hamming', type=int, default=4,
        help='Reject near-duplicates within the same category/status when dHash distance <= this value. Use -1 to disable.',
    )
    return p.parse_args()


def norm(s):
    return re.sub(r'[^a-z0-9]+', '_', str(s or '').lower()).strip('_')


def is_img(p):
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def path_parts(p):
    return [x.lower() for x in Path(p).parts]


def is_mask(p):
    ps = set(path_parts(p))
    name = p.name.lower()
    return bool(ps & MASK_HINTS) or any(x in name for x in ['mask', '_gt', 'ground_truth'])


def is_source_img(p):
    return is_img(p) and not is_mask(p) and '/__macosx/' not in str(p).lower()


def infer_cat_label(path, root):
    """Parse VisA layouts, including <category>/Data/Images/<status>/file."""
    rel = path.relative_to(root)
    raw = list(rel.parts)
    ns = [norm(x) for x in raw]

    label = 'unknown'
    status_idx = None
    for i, token in enumerate(ns[:-1]):
        if token in NORMAL_HINTS:
            label = 'good'
            status_idx = i
            break
        if token in ANOMALY_HINTS:
            label = 'anomaly'
            status_idx = i
            break

    # Locate the Images/Masks directory closest to the status directory.
    media_idx = None
    search_end = status_idx if status_idx is not None else len(ns) - 1
    for i in range(search_end - 1, -1, -1):
        if ns[i] in MEDIA_DIR_HINTS:
            media_idx = i
            break

    # Walk backwards from Images/Masks and skip generic wrapper directories.
    # Handles both:
    #   Data/<category>/Images/Anomaly/file
    #   <category>/Data/Images/Anomaly/file   (actual VisA archive layout)
    category = 'unknown'
    category_search_start = (media_idx - 1) if media_idx is not None else ((status_idx - 1) if status_idx else len(ns) - 2)
    for i in range(category_search_start, -1, -1):
        token = ns[i]
        if not token:
            continue
        if token in WRAPPER_DIR_HINTS or token in MEDIA_DIR_HINTS:
            continue
        if token in NORMAL_HINTS or token in ANOMALY_HINTS:
            continue
        category = norm(raw[i])
        break

    # Last-resort fallback for unusual layouts.
    if category == 'unknown' and len(raw) >= 2:
        category = norm(raw[-2])

    return category, label, label == 'anomaly'


def image_stats_and_fingerprint(path):
    try:
        with Image.open(path) as opened:
            img = opened.convert('RGB')
        arr = np.array(img)
        h, w = arr.shape[:2]
        gray = arr.mean(axis=2)
        hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=True)
        hist = hist + 1e-12
        entropy = float(-(hist * np.log(hist)).sum() / math.log(256))
        color_std = float(np.std(arr.reshape(-1, 3), axis=0).mean() / 255.0)
        gx = float(np.abs(np.diff(gray, axis=1)).mean() / 255.0) if w > 1 else 0.0
        gy = float(np.abs(np.diff(gray, axis=0)).mean() / 255.0) if h > 1 else 0.0
        sharpness = float(np.var(gray) / (255.0 ** 2))

        normalized = img.resize((64, 64), Image.Resampling.LANCZOS)
        visual_sha1 = hashlib.sha1(np.asarray(normalized, dtype=np.uint8).tobytes()).hexdigest()

        dhash_img = img.convert('L').resize((16, 16), Image.Resampling.LANCZOS)
        dhash_arr = np.asarray(dhash_img, dtype=np.int16)
        bits = (dhash_arr[:, 1:] > dhash_arr[:, :-1]).reshape(-1)
        dhash = 0
        for bit in bits:
            dhash = (dhash << 1) | int(bit)

        return {
            'image_width': int(w),
            'image_height': int(h),
            'pixels': int(w * h),
            'image_entropy': entropy,
            'color_std': color_std,
            'edge_density_proxy': (gx + gy) / 2.0,
            'sharpness_proxy': sharpness,
        }, visual_sha1, dhash
    except Exception:
        return {
            'image_width': 0, 'image_height': 0, 'pixels': 0,
            'image_entropy': 0.0, 'color_std': 0.0,
            'edge_density_proxy': 0.0, 'sharpness_proxy': 0.0,
        }, None, None


def mask_stats(path):
    if not path or not Path(path).exists():
        return {'has_mask': False, 'mask_area_ratio': 0.0}
    try:
        arr = np.array(Image.open(path).convert('L'))
        return {'has_mask': True, 'mask_area_ratio': float((arr > 0).sum()) / max(float(arr.size), 1.0)}
    except Exception:
        return {'has_mask': False, 'mask_area_ratio': 0.0}


def build_mask_index(root):
    idx = defaultdict(list)
    for p in root.rglob('*'):
        if is_img(p) and is_mask(p):
            key = norm(p.stem.replace('_mask', '').replace('-mask', '').replace('mask', ''))
            cat, _, _ = infer_cat_label(p, root)
            idx[key].append(p)
            idx[f'{cat}__{key}'].append(p)
    return idx


def find_mask(img, root, idx, cat, is_anomaly):
    if not is_anomaly:
        return None
    stem_key = norm(img.stem)
    for key in [f'{cat}__{stem_key}', stem_key, norm(stem_key.replace('_bad', '').replace('_anomaly', ''))]:
        if idx.get(key):
            return sorted(idx[key])[0]

    parts = list(img.parts)
    for i, token in enumerate(parts):
        if token.lower() in IMAGE_DIR_HINTS:
            for mask_dir in ['Masks', 'masks']:
                candidate_parts = parts[:]
                candidate_parts[i] = mask_dir
                candidate = Path(*candidate_parts)
                for cand in [candidate.with_suffix('.png'), candidate.with_name(candidate.stem + '_mask.png')]:
                    if cand.exists():
                        return cand
    return None


def discover(root):
    idx = build_mask_index(root)
    out = []
    for p in root.rglob('*'):
        if not is_source_img(p):
            continue
        cat, label, is_anomaly = infer_cat_label(p, root)
        mask = find_mask(p, root, idx, cat, is_anomaly)
        out.append({
            'category': cat,
            'defect_type': label,
            'is_anomaly': is_anomaly,
            'raw_image_path': str(p),
            'raw_mask_path': str(mask) if mask else None,
        })
    return out


def score(item):
    s = item['image_stats']
    m = item['mask_stats']
    if s['pixels'] <= 0:
        return -1e9
    value = (
        min(s['pixels'] / (1024 * 768), 2) * 20
        + s['image_entropy'] * 20
        + s['edge_density_proxy'] * 25
        + s['color_std'] * 15
        + s['sharpness_proxy'] * 10
    )
    if item['is_anomaly']:
        value += 30
    if m['has_mask']:
        value += 15 + min(m['mask_area_ratio'], 0.35) * 25
    return value


def copy_or_link(src, dst, mode):
    if not src:
        return None
    src = Path(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == 'copy':
        shutil.copy2(src, dst)
    elif mode == 'symlink':
        dst.symlink_to(src.resolve())
    else:
        try:
            dst.hardlink_to(src)
        except Exception:
            shutil.copy2(src, dst)
    return str(dst)


def hamming_distance(a, b):
    return (a ^ b).bit_count()


def select_balanced(scored, num_samples, anomaly_ratio, dedup_hamming):
    """Balance categories first, then approximate anomaly/normal ratio inside each category."""
    groups = defaultdict(lambda: {'anomaly': [], 'good': [], 'unknown': []})
    for item in scored:
        status = 'anomaly' if item['is_anomaly'] else ('good' if item['defect_type'] == 'good' else 'unknown')
        groups[item['category']][status].append(item)

    categories = sorted(groups)
    positions = {cat: {'anomaly': 0, 'good': 0, 'unknown': 0} for cat in categories}
    selected = []
    selected_paths = set()
    selected_visual_hashes = set()
    selected_dhashes = defaultdict(list)
    category_counts = Counter()
    status_counts = Counter()
    rejected_exact = 0
    rejected_near = 0

    def take_next(cat, preferred_status):
        nonlocal rejected_exact, rejected_near
        fallback_order = [preferred_status]
        for status in ['anomaly', 'good', 'unknown']:
            if status not in fallback_order:
                fallback_order.append(status)

        for status in fallback_order:
            queue = groups[cat][status]
            while positions[cat][status] < len(queue):
                item = queue[positions[cat][status]]
                positions[cat][status] += 1
                path = item['raw_image_path']
                visual_sha1 = item.get('visual_sha1')
                dhash = item.get('dhash')
                dedup_key = (cat, status)

                if path in selected_paths or (visual_sha1 and visual_sha1 in selected_visual_hashes):
                    rejected_exact += 1
                    continue
                if dedup_hamming >= 0 and dhash is not None:
                    if any(hamming_distance(dhash, prev) <= dedup_hamming for prev in selected_dhashes[dedup_key]):
                        rejected_near += 1
                        continue
                return item, status
        return None, None

    while len(selected) < num_samples:
        progressed = False
        for cat in categories:
            if len(selected) >= num_samples:
                break

            cat_total = category_counts[cat]
            cat_anomaly = status_counts[(cat, 'anomaly')]
            # Choose the status that keeps this category closest to the requested ratio.
            preferred = 'anomaly' if cat_anomaly < round((cat_total + 1) * anomaly_ratio) else 'good'
            item, status = take_next(cat, preferred)
            if item is None:
                continue

            selected.append(item)
            selected_paths.add(item['raw_image_path'])
            if item.get('visual_sha1'):
                selected_visual_hashes.add(item['visual_sha1'])
            if item.get('dhash') is not None:
                selected_dhashes[(cat, status)].append(item['dhash'])
            category_counts[cat] += 1
            status_counts[(cat, status)] += 1
            progressed = True
        if not progressed:
            break

    return selected, category_counts, status_counts, rejected_exact, rejected_near


def main():
    args = parse_args()
    if not 0.0 <= args.prefer_anomaly_ratio <= 1.0:
        raise ValueError('--prefer_anomaly_ratio must be between 0 and 1')
    random.seed(args.seed)
    root = Path(args.visa_root)
    if not root.exists():
        raise FileNotFoundError(root)

    out_img = Path(args.output_image_dir)
    out_mask = Path(args.output_mask_dir)
    metadata_path = Path(args.metadata_jsonl)
    manifest_path = Path(args.manifest_csv)

    if args.overwrite:
        for p in [out_img, out_mask]:
            if p.exists():
                shutil.rmtree(p)
        for p in [metadata_path, manifest_path]:
            if p.exists():
                p.unlink()

    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    samples = discover(root)
    print(f'[INFO] discovered source images={len(samples)}')
    discovered_counts = Counter(x['category'] for x in samples)
    print('[INFO] discovered categories:')
    for k, v in sorted(discovered_counts.items()):
        print(f'  {k}: {v}')
    if not samples:
        raise RuntimeError('No VisA images discovered.')
    if len(discovered_counts) <= 1:
        raise RuntimeError(
            f'VisA category parsing found only {len(discovered_counts)} category: {dict(discovered_counts)}. '
            'Check the extracted directory layout before sampling.'
        )

    scored = []
    for item in tqdm(samples, desc='Scoring/fingerprinting VisA images'):
        stats, visual_sha1, dhash = image_stats_and_fingerprint(Path(item['raw_image_path']))
        item['image_stats'] = stats
        item['visual_sha1'] = visual_sha1
        item['dhash'] = dhash
        item['mask_stats'] = mask_stats(item['raw_mask_path'])
        item['score'] = score(item)
        item['_tie'] = random.random()
        scored.append(item)

    scored.sort(key=lambda x: (-x['score'], x['_tie']))
    selected, category_counts, status_counts, rejected_exact, rejected_near = select_balanced(
        scored, args.num_samples, args.prefer_anomaly_ratio, args.dedup_hamming
    )
    if len(selected) < args.num_samples:
        raise RuntimeError(
            f'Could only select {len(selected)}/{args.num_samples} images after deduplication. '
            f'Rerun with a smaller --dedup_hamming (or -1 to disable near-duplicate filtering).'
        )

    print(f'[INFO] selected={len(selected)} anomaly={sum(x["is_anomaly"] for x in selected)}')
    print(f'[INFO] rejected duplicate visuals: exact={rejected_exact}, near={rejected_near}')
    print('[INFO] selected category/status counts:')
    for cat in sorted(category_counts):
        print(
            f'  {cat}: total={category_counts[cat]}, '
            f'anomaly={status_counts[(cat, "anomaly")]}, '
            f'good={status_counts[(cat, "good")]}, '
            f'unknown={status_counts[(cat, "unknown")]}'
        )

    rows = []
    for rank, item in enumerate(tqdm(selected, desc='Copying selected VisA files'), 1):
        raw = Path(item['raw_image_path'])
        path_hash = hashlib.sha1(str(raw).encode('utf-8')).hexdigest()[:10]
        sample_id = f'visa_{rank:04d}_{item["category"]}_{item["defect_type"]}_{norm(raw.stem)}_{path_hash}'
        image_path = copy_or_link(raw, out_img / f'{sample_id}{raw.suffix.lower()}', args.copy_mode)
        mask_path = None
        if item['raw_mask_path']:
            raw_mask = Path(item['raw_mask_path'])
            mask_path = copy_or_link(raw_mask, out_mask / f'{sample_id}{raw_mask.suffix.lower()}', args.copy_mode)

        s = item['image_stats']
        m = item['mask_stats']
        visual_annotation = {
            'condition_type': 'industrial_visual_condition_or_anomaly',
            'category': item['category'],
            'defect_type': item['defect_type'],
            'is_anomaly': item['is_anomaly'],
            'has_mask': m['has_mask'],
            'mask_area_ratio': round(m['mask_area_ratio'], 6),
            'visual_cues_for_ideation': [
                'defect pattern', 'surface anomaly', 'quality inspection signal',
                'manufacturing or packaging QA risk', 'visual inspection automation opportunity'
            ],
        }
        prompt_context = (
            'Domain: Visual Condition / Anomaly. Generate a startup idea using visible defects, abnormal visual conditions, '
            'product quality cues, damage, contamination, wear, maintenance risk, or inspection patterns. '
            'Target quality inspection, industrial monitoring, defect triage, predictive maintenance, or visual QA workflows.'
        )
        row = {
            'rank': rank,
            'sample_id': sample_id,
            'source_dataset': 'visa',
            'subset_type': 'multimodal_critical',
            'domain': 'visual_condition_anomaly',
            'domain_display_name': 'Visual Condition / Anomaly',
            'image_path': image_path,
            'selected_image_path': image_path,
            'mask_path': mask_path,
            'annotation_path': mask_path,
            'selected_annotation_path': mask_path,
            'raw_image_path': item['raw_image_path'],
            'raw_mask_path': item['raw_mask_path'],
            'prompt_context': prompt_context,
            'visual_annotation': visual_annotation,
            'question': 'Generate a startup idea grounded in this visual condition/anomaly image and retrieved market evidence.',
            'category': item['category'],
            'defect_type': item['defect_type'],
            'is_anomaly': item['is_anomaly'],
            'has_mask': m['has_mask'],
            'mask_area_ratio': round(m['mask_area_ratio'], 6),
            'score': round(item['score'], 6),
            **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in s.items()},
        }
        rows.append(row)

    with metadata_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    fields = [
        'rank', 'sample_id', 'source_dataset', 'subset_type', 'domain', 'domain_display_name',
        'image_path', 'mask_path', 'category', 'defect_type', 'is_anomaly', 'has_mask',
        'mask_area_ratio', 'prompt_context', 'question', 'score', 'image_width', 'image_height',
        'pixels', 'raw_image_path', 'raw_mask_path'
    ]
    with manifest_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fields})

    final_image_count = sum(1 for p in out_img.rglob('*') if is_img(p))
    print('\n' + '=' * 80)
    print('DONE')
    print('=' * 80)
    print(f'Images saved to: {out_img}')
    print(f'Masks saved to:  {out_mask}')
    print(f'Metadata JSONL:  {metadata_path}')
    print(f'Manifest CSV:    {manifest_path}')
    print(f'Selected count:  {len(rows)}')
    print(f'Final image files: {final_image_count}')
    if final_image_count != len(rows):
        raise RuntimeError(f'Output image count mismatch: files={final_image_count}, metadata_rows={len(rows)}')


if __name__ == '__main__':
    main()
