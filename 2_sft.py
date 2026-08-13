#!/usr/bin/env python3
"""Final SFT training script for Qwen2-VL-7B LoRA.

Launch examples
---------------
4 GPUs:
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 2_sft.py
8 GPUs:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 2_sft.py

The script keeps the effective global batch size constant across 4/8 GPUs by
adjusting gradient accumulation automatically.
"""

import argparse
import json
import math
import os
import random
import re
import types
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model

if not hasattr(torch, "compiler"):
    torch.compiler = types.SimpleNamespace()
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

os.environ.setdefault("ACCELERATE_USE_DEEPSPEED", "false")
os.environ.setdefault("DISABLE_DEEPSPEED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
OUTPUT_ROOT = "/workspace/checkpoints"
DEFAULT_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "sft")
DATA_DIR = "/workspace/data/mba/expert_data/"
IMAGE_DIR = "/workspace/data/ADEChallengeData2016/images/validation/"

SOURCE_MANIFEST_FILE = os.environ.get(
    "MBA_SOURCE_MANIFEST",
    "/workspace/data/mba/source_manifest_2k.jsonl",
)
DDG_EVIDENCE_FILE = os.environ.get(
    "MBA_DDG_EVIDENCE",
    "/workspace/data/mba/ddg_marketdb.jsonl",
)
CAPTION_ROOT = os.environ.get(
    "MBA_CAPTION_ROOT",
    "/workspace/data/mba/source_images",
)
STRICT_IMAGE_CAPTIONS = os.environ.get("MBA_STRICT_CAPTIONS", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
CAPTION_SECTION_NAME = "Auxiliary Image Caption"
_CAPTION_RESOURCE_CACHE = None
CAPTION_DOMAIN_ALIASES = {
    "general_scene": ("general", "ade20k"),
    "composition_layout": ("composition_layout", "rico"),
    "spatial_human_flow": ("spatial_human_flow", "coco_person_flow"),
    "visual_condition_anomaly": ("visual_condition_anomaly", "visa"),
    "aesthetic_texture_material": ("aesthetic_texture_material", "dtd"),
    "technical_visual_artifact": ("technical_visual_artifact", "deeppcb"),
}
INCONTEXT_PROMPT_REVISION = "mba_clean_ddg_caption_incontext_v3"
STRICT_INCONTEXT_CONTEXT = os.environ.get("MBA_STRICT_INCONTEXT", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
_INCONTEXT_RESOURCE_CACHE = None

DOMAIN_QUESTIONS = {
    "general_scene": "Identify market-grounded startup opportunities from visible objects, activities, spatial context, and everyday physical-environment needs.",
    "composition_layout": "Identify startup opportunities from UI layout, information hierarchy, interaction friction, accessibility, conversion, and digital service workflow.",
    "spatial_human_flow": "Identify startup opportunities from human movement, spatial interaction, crowd or queue behavior, navigation, service flow, and space utilization.",
    "visual_condition_anomaly": "Identify startup opportunities from condition, visible defects, anomaly signals, quality inspection, maintenance, and returns prevention.",
    "aesthetic_texture_material": "Identify startup opportunities from texture, material quality, surface appearance, tactile or visual differentiation, design, and business experience.",
    "technical_visual_artifact": "Identify B2B manufacturing opportunities from PCB trace or pad patterns, possible defect risks, AOI automation, yield optimization, and electronics quality control.",
}

BUSINESS_LENS_QUESTIONS = {
    "cost_resource_efficiency": "Focus on cost reduction, resource productivity, waste reduction, energy or material efficiency, and sustainability-linked savings.",
    "intelligent_automation": "Focus on AI automation, sensing, computer vision, predictive analytics, technical intelligence, and scalable decision support.",
    "customer_operations_growth": "Focus on customer experience, service workflow, conversion, retention, operational throughput, and business growth.",
}
TEST_FILE = os.path.join(OUTPUT_ROOT, "grpo", "grpo_test_prompts.json")
TEST_SPLIT_SUMMARY_FILE = os.path.join(OUTPUT_ROOT, "grpo", "grpo_test_split_summary.json")
SEED = 2026
FINAL_TEST_RATIO = 0.05
DOMAIN_STRATIFIED_SPLIT_REVISION = "domain_stratified_image_v1"
EXPECTED_DOMAINS = tuple(DOMAIN_QUESTIONS.keys())

POLICY_IMAGE_SIZE = 512
POLICY_MIN_PIXELS = 4 * 28 * 28
POLICY_MAX_PIXELS = 128 * 28 * 28

IDEA_JSON_FIELDS = ("title", "description", "implementation", "differentiation")
IDEA_FORMAT_INSTRUCTION = """
Return exactly one business idea as valid JSON only.
Use exactly these four string keys:
{
  "title": "concise business idea name",
  "description": "core function, target user, need, and benefit",
  "implementation": "how the business idea would be built or deployed using current technology and the image, request, visual annotation, and DDG market-evidence context",
  "differentiation": "what makes it meaningfully different or defensible compared with existing solutions"
}
Do not include markdown, numbering, extra keys, or text outside the JSON object.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--image_dir", default=IMAGE_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_batch_size", type=int, default=0, help="0=auto from GPU memory")
    parser.add_argument("--target_global_batch_size", type=int, default=32)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=-1, help="-1=auto per DDP process")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--prepare_split_only",
        action="store_true",
        help="Create/verify the 95/5 image-level train-test manifest and exit.",
    )
    parser.add_argument(
        "--overwrite_test_manifest",
        action="store_true",
        help=(
            "Replace an existing incompatible grpo_test_prompts.json. "
            "Use this once when migrating from the old sample-row split."
        ),
    )
    parser.add_argument(
        "--preview_prompts_only",
        action="store_true",
        help="Build and export the exact canonical input questions, then exit before model loading.",
    )
    parser.add_argument(
        "--preview_prompts_path",
        default=os.path.join(OUTPUT_ROOT, "grpo", "all_input_prompts.jsonl"),
        help="JSONL output path used by --preview_prompts_only.",
    )
    parser.add_argument(
        "--preview_print_limit",
        type=int,
        default=3,
        help="Number of exported questions to print to the terminal; 0 prints none, -1 prints all.",
    )
    return parser.parse_args()


def clean_qwen_vl_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    bad_tokens = [
        "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
        "<image>", "</image>", "<img>", "</img>", "<video>", "</video>",
    ]
    for token in bad_tokens:
        text = text.replace(token, "")
    text = re.sub(r"<\|[^>]*?\|>", "", text)
    text = re.sub(r"<\s*/?\s*(image|img|video)\s*>", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def attach_idea_format_instruction(question: str) -> str:
    base = clean_incontext_prompt_text(question)
    if not base:
        base = "Generate one creative and practical B2B business idea based on this image."
    lowered = base.lower()
    if all(f'"{field}"' in lowered for field in IDEA_JSON_FIELDS):
        return base
    return f"{base}\n\n{IDEA_FORMAT_INSTRUCTION}"


def _shorten_for_field(text: str, max_chars: int = 520) -> str:
    text = clean_qwen_vl_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _parse_json_object(text):
    if isinstance(text, dict):
        return text
    raw = str(text or "").strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def structured_idea_from_item(item: Dict[str, Any], topic: str = "", idea: Any = "") -> Dict[str, str]:
    candidates = []
    for key in ("idea_json", "structured_idea", "idea", "business_idea"):
        if key in item:
            candidates.append(item.get(key))
    candidates.append({
        "title": item.get("title") or topic,
        "description": item.get("description"),
        "implementation": item.get("implementation"),
        "differentiation": item.get("differentiation"),
    })
    for candidate in candidates:
        obj = _parse_json_object(candidate)
        if not isinstance(obj, dict):
            continue
        if all(clean_qwen_vl_text(obj.get(field)) for field in IDEA_JSON_FIELDS):
            return {field: clean_qwen_vl_text(obj[field]) for field in IDEA_JSON_FIELDS}
    title = _shorten_for_field(topic or "Image-grounded B2B Business Idea", 96)
    description = _shorten_for_field(idea or topic or "A practical business idea grounded in the image and market evidence.", 700)
    implementation = _shorten_for_field(
        "Build and deploy the business idea using current software, AI, data, and operational integration methods; "
        f"use the image/request and market evidence to define the target workflow for: {description}",
        700,
    )
    differentiation = _shorten_for_field(
        "Unlike a generic AI platform or dashboard, this business idea is differentiated by a specific customer workflow, "
        f"market-grounded use case, and defensible execution focus around: {title}",
        600,
    )
    return {
        "title": title,
        "description": description,
        "implementation": implementation,
        "differentiation": differentiation,
    }


def format_structured_idea_json(item: Dict[str, Any], topic: str = "", idea: Any = "") -> str:
    obj = structured_idea_from_item(item, topic=topic, idea=idea)
    return json.dumps({field: obj[field] for field in IDEA_JSON_FIELDS}, ensure_ascii=False, indent=2)


def load_policy_image(image_path: str) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if image.size[0] > POLICY_IMAGE_SIZE or image.size[1] > POLICY_IMAGE_SIZE:
        image = image.resize(
            (POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )
    return image


def resolve_expert_image_path(data: Dict[str, Any], image_dir: str):
    candidates = []
    for key in ["image_path", "selected_image_path", "raw_image_path"]:
        value = data.get(key)
        if value:
            candidates.append(str(value))
    image_id = data.get("image_id") or data.get("image_name")
    if image_id:
        candidates.append(str(image_id))

    for candidate in candidates:
        path = candidate
        if os.path.isabs(path) and os.path.exists(path):
            return path
        if os.path.exists(path):
            return os.path.abspath(path)
        joined = os.path.join(image_dir, path)
        if os.path.exists(joined):
            return joined
        joined_base = os.path.join("/workspace", path.lstrip("/"))
        if os.path.exists(joined_base):
            return joined_base
    return None



def clean_incontext_prompt_text(text: str) -> str:
    """Remove vision-control tokens while preserving section and JSON line breaks."""
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    bad_tokens = [
        "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
        "<image>", "</image>", "<img>", "</img>", "<video>", "</video>",
    ]
    for token in bad_tokens:
        text = text.replace(token, "")
    text = re.sub(r"<\|[^>]*?\|>", "", text)
    text = re.sub(r"<\s*/?\s*(image|img|video)\s*>", "", text, flags=re.IGNORECASE)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    out = []
    blank = False
    for line in lines:
        if line:
            out.append(line)
            blank = False
        elif out and not blank:
            out.append("")
            blank = True
    return "\n".join(out).strip()


def _read_jsonl_strict(path: str):
    rows = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required in-context source file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected a JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def _clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()



def _normalize_caption_path(value: Any) -> str:
    if value is None:
        return ""
    return os.path.normpath(str(value).strip()).replace("\\", "/")


def _infer_caption_domain(caption_file: str) -> str:
    normalized = _normalize_caption_path(os.path.relpath(caption_file, CAPTION_ROOT)).lower()
    parts = set(part for part in normalized.split("/") if part)
    for domain, aliases in CAPTION_DOMAIN_ALIASES.items():
        if any(alias.lower() in parts for alias in aliases):
            return domain
    return ""


def _register_unique_caption(mapping: Dict[Any, Any], key: Any, caption: str) -> None:
    if not key:
        return
    previous = mapping.get(key)
    if previous is None and key in mapping:
        return
    if previous is not None and previous != caption:
        mapping[key] = None
    elif key not in mapping:
        mapping[key] = caption


def _load_caption_resources() -> Dict[str, Any]:
    """Index every generated captions.jsonl under MBA_CAPTION_ROOT once."""
    global _CAPTION_RESOURCE_CACHE
    if _CAPTION_RESOURCE_CACHE is not None:
        return _CAPTION_RESOURCE_CACHE
    if not os.path.isdir(CAPTION_ROOT):
        raise FileNotFoundError(
            f"Caption root not found: {CAPTION_ROOT}. Run generate_captions.py first "
            "or set MBA_CAPTION_ROOT."
        )

    by_absolute: Dict[str, str] = {}
    by_root_relative: Dict[str, Any] = {}
    by_domain_basename: Dict[Any, Any] = {}
    by_basename: Dict[str, Any] = {}
    caption_files = []
    row_count = 0

    for current_root, _, filenames in os.walk(CAPTION_ROOT):
        if "captions.jsonl" not in filenames:
            continue
        caption_file = os.path.join(current_root, "captions.jsonl")
        caption_files.append(caption_file)
        caption_domain = _infer_caption_domain(caption_file)
        with open(caption_file, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception as exc:
                    raise RuntimeError(
                        f"Invalid caption JSONL at {caption_file}:{line_no}"
                    ) from exc
                if not isinstance(row, dict):
                    raise RuntimeError(
                        f"Expected caption JSON object at {caption_file}:{line_no}"
                    )
                relative_path = _normalize_caption_path(row.get("image_path"))
                caption = _clean_scalar(row.get("caption"))
                if not relative_path or not caption:
                    raise RuntimeError(
                        f"Missing image_path/caption at {caption_file}:{line_no}"
                    )
                absolute_path = relative_path
                if not os.path.isabs(absolute_path):
                    absolute_path = os.path.join(current_root, relative_path)
                absolute_key = os.path.realpath(os.path.abspath(absolute_path))
                root_relative = _normalize_caption_path(
                    os.path.relpath(absolute_key, CAPTION_ROOT)
                )
                basename = os.path.basename(absolute_key)

                previous = by_absolute.get(absolute_key)
                if previous is not None and previous != caption:
                    raise RuntimeError(
                        f"Conflicting captions for image: {absolute_key}"
                    )
                by_absolute[absolute_key] = caption
                _register_unique_caption(by_root_relative, root_relative, caption)
                _register_unique_caption(by_basename, basename, caption)
                if caption_domain:
                    _register_unique_caption(
                        by_domain_basename, (caption_domain, basename), caption
                    )
                row_count += 1

    if not caption_files or row_count == 0:
        raise RuntimeError(
            f"No populated captions.jsonl files found under {CAPTION_ROOT}"
        )

    _CAPTION_RESOURCE_CACHE = {
        "by_absolute": by_absolute,
        "by_root_relative": by_root_relative,
        "by_domain_basename": by_domain_basename,
        "by_basename": by_basename,
        "caption_files": tuple(sorted(caption_files)),
        "row_count": row_count,
    }
    return _CAPTION_RESOURCE_CACHE


def resolve_image_caption(
    image_path: str,
    domain: str,
    data: Dict[str, Any] = None,
    source_row: Dict[str, Any] = None,
) -> str:
    """Resolve the generated caption for an expert-data image without changing image identity."""
    resources = _load_caption_resources()
    candidate_values = [image_path]
    for record in (data or {}, source_row or {}):
        for key in ("image_path", "selected_image_path", "raw_image_path", "image_id", "image_name"):
            value = record.get(key)
            if value not in (None, ""):
                candidate_values.append(value)

    for value in candidate_values:
        normalized = _normalize_caption_path(value)
        if not normalized:
            continue
        absolute_candidates = []
        if os.path.isabs(normalized):
            absolute_candidates.append(normalized)
        else:
            absolute_candidates.extend((
                normalized,
                os.path.join("/workspace", normalized.lstrip("/")),
                os.path.join(CAPTION_ROOT, normalized),
            ))
        for candidate in absolute_candidates:
            key = os.path.realpath(os.path.abspath(candidate))
            caption = resources["by_absolute"].get(key)
            if caption:
                return caption

        marker = "/source_images/"
        marker_index = normalized.lower().find(marker)
        if marker_index >= 0:
            relative_key = normalized[marker_index + len(marker):]
            caption = resources["by_root_relative"].get(relative_key)
            if caption:
                return caption

    basename = os.path.basename(_normalize_caption_path(image_path))
    caption = resources["by_domain_basename"].get((domain, basename))
    if caption:
        return caption
    caption = resources["by_basename"].get(basename)
    if caption:
        return caption

    if STRICT_IMAGE_CAPTIONS:
        raise RuntimeError(
            f"No unique generated caption found for domain={domain!r}, image={image_path!r}. "
            f"Check captions.jsonl under {CAPTION_ROOT} or set MBA_STRICT_CAPTIONS=0."
        )
    return ""


def _json_context(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        try:
            parsed = json.loads(stripped)
        except Exception:
            return stripped
        value = parsed
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _annotation_from_row(row: Dict[str, Any]):
    preferred_keys = (
        "visual_annotation", "visual_context", "image_annotation", "image_annotations",
        "annotation", "annotations", "scene_annotation", "material_annotation",
        "visual_metadata", "image_metadata",
    )
    payload = {}
    for key in preferred_keys:
        value = row.get(key)
        if value not in (None, "", [], {}):
            payload[key] = value
    if payload:
        return payload

    # Backward-compatible fallback for manifests that store visual attributes as top-level fields.
    ignored = {
        "sample_id", "domain", "domain_display_name", "source_dataset", "image_path",
        "selected_image_path", "raw_image_path", "domain_question",
    }
    for key, value in row.items():
        lowered = str(key).lower()
        if key in ignored or value in (None, "", [], {}):
            continue
        if any(token in lowered for token in (
            "visual", "caption", "object", "scene", "texture", "material", "layout",
            "flow", "anomaly", "defect", "attribute", "label", "description",
        )):
            payload[key] = value
    return payload


def _card_index(card: Dict[str, Any]):
    try:
        return int(card.get("source_query_index"))
    except Exception:
        evidence_id = _clean_scalar(card.get("evidence_id"))
        match = re.fullmatch(r"W(\d+)", evidence_id, flags=re.IGNORECASE)
        return int(match.group(1)) if match else None


def _load_incontext_resources():
    global _INCONTEXT_RESOURCE_CACHE
    if _INCONTEXT_RESOURCE_CACHE is not None:
        return _INCONTEXT_RESOURCE_CACHE

    manifest_by_sample = {}
    for row in _read_jsonl_strict(SOURCE_MANIFEST_FILE):
        sample_id = _clean_scalar(row.get("sample_id"))
        if not sample_id:
            continue
        if sample_id in manifest_by_sample:
            raise RuntimeError(f"Duplicate sample_id in source manifest: {sample_id}")
        manifest_by_sample[sample_id] = row

    ddg_by_pair = {}
    for row in _read_jsonl_strict(DDG_EVIDENCE_FILE):
        sample_id = _clean_scalar(row.get("sample_id"))
        lens = _clean_scalar(row.get("business_lens"))
        queries = row.get("core_retrieval_queries") or []
        cards = row.get("web_evidence_cards") or []
        if not sample_id or not lens:
            continue
        cards_by_index = {
            _card_index(card): card
            for card in cards
            if isinstance(card, dict) and _card_index(card) is not None
        }
        for query_index in range(1, max(len(queries), len(cards_by_index)) + 1):
            source_query = _clean_scalar(queries[query_index - 1]) if query_index <= len(queries) else ""
            card = dict(cards_by_index.get(query_index) or {})
            if source_query:
                card.setdefault("source_query", source_query)
            card.setdefault("source_query_index", query_index)
            card.setdefault("evidence_id", f"W{query_index}")
            key = (sample_id, lens, query_index)
            if key in ddg_by_pair:
                raise RuntimeError(f"Duplicate DDG query/evidence pair: {key}")
            ddg_by_pair[key] = {
                "source_query": source_query or _clean_scalar(card.get("source_query")),
                "card": card,
            }

    _INCONTEXT_RESOURCE_CACHE = {
        "manifest_by_sample": manifest_by_sample,
        "ddg_by_pair": ddg_by_pair,
    }
    return _INCONTEXT_RESOURCE_CACHE


def _local_evidence_card(result: Dict[str, Any], item: Dict[str, Any], query_index: int) -> Dict[str, Any]:
    candidates = []
    for card in result.get("web_evidence_cards") or []:
        if isinstance(card, dict) and _card_index(card) == query_index:
            candidates.append(card)
    for card in item.get("matched_web_evidence") or []:
        if isinstance(card, dict) and (_card_index(card) in (None, query_index)):
            candidates.append(card)
    merged = {}
    for card in reversed(candidates):
        merged.update({k: v for k, v in card.items() if v not in (None, "", [], {})})
    return merged


def build_domain_aware_question(data: Dict[str, Any], result: Dict[str, Any], item: Dict[str, Any], image_path: str) -> str:
    """Build the canonical clean in-context prompt shared by SFT, GRPO, and Eval6M."""
    resources = _load_incontext_resources()
    sample_id = _clean_scalar(data.get("sample_id"))
    lens = _clean_scalar(result.get("business_lens") or item.get("business_lens"))
    try:
        query_index = int(item.get("query_index"))
    except Exception:
        query_index = 0

    source_row = resources["manifest_by_sample"].get(sample_id, {})
    pair = resources["ddg_by_pair"].get((sample_id, lens, query_index), {})
    local_card = _local_evidence_card(result, item, query_index)
    evidence_card = dict(local_card)
    evidence_card.update({
        key: value for key, value in (pair.get("card") or {}).items()
        if value not in (None, "", [], {})
    })

    source_query = _clean_scalar(pair.get("source_query") or item.get("source_query"))
    item_query = _clean_scalar(item.get("source_query"))
    if STRICT_INCONTEXT_CONTEXT and item_query and source_query and item_query != source_query:
        raise RuntimeError(
            f"DDG query mismatch for sample={sample_id} lens={lens} Q{query_index}: "
            f"expert_data={item_query!r} canonical_ddg={source_query!r}"
        )
    source_query = source_query or item_query

    domain = _clean_scalar(data.get("domain") or source_row.get("domain"))
    domain_display = _clean_scalar(
        data.get("domain_display_name") or source_row.get("domain_display_name") or domain
    )
    domain_question = _clean_scalar(
        data.get("domain_question")
        or result.get("domain_question")
        or item.get("domain_question")
        or source_row.get("domain_question")
        or DOMAIN_QUESTIONS.get(domain, "")
    )
    lens_focus = _clean_scalar(BUSINESS_LENS_QUESTIONS.get(lens, ""))
    base_question = _clean_scalar(
        result.get("user_question")
        or result.get("question")
        or item.get("user_question")
        or item.get("question")
        or data.get("user_question")
        or data.get("question")
        or "Generate exactly one creative, practical, market-grounded business idea based on this image and context."
    )
    annotation = _annotation_from_row(source_row) or _annotation_from_row(data)
    caption = resolve_image_caption(
        image_path=image_path,
        domain=domain,
        data=data,
        source_row=source_row,
    )

    evidence_text = _clean_scalar(
        evidence_card.get("text")
        or evidence_card.get("salient_market_text")
        or evidence_card.get("text_preview")
        or evidence_card.get("snippet")
    )
    extracted_facts = evidence_card.get("extracted_market_facts") or []
    if not isinstance(extracted_facts, list):
        extracted_facts = [extracted_facts]
    extracted_facts = [_clean_scalar(value) for value in extracted_facts if _clean_scalar(value)]

    missing = []
    if not sample_id:
        missing.append("sample_id")
    if not source_row:
        missing.append("canonical_source_manifest_row")
    if not domain:
        missing.append("domain")
    if not domain_question:
        missing.append("domain_question")
    if not annotation:
        missing.append("visual_annotation")
    if not caption:
        missing.append("image_caption")
    if not lens:
        missing.append("business_lens")
    if query_index <= 0:
        missing.append("query_index")
    if not pair:
        missing.append("canonical_ddg_query_evidence_pair")
    if not source_query:
        missing.append("source_query")
    if not evidence_card:
        missing.append("paired_ddg_evidence")
    if not evidence_text:
        missing.append("ddg_evidence_text")
    if STRICT_INCONTEXT_CONTEXT and missing:
        raise RuntimeError(
            f"Incomplete clean in-context prompt for sample={sample_id or '<missing>'}: "
            + ", ".join(missing)
        )

    # Keep only ideation-relevant evidence fields. The URL and retrieval metadata
    # remain in the source DDG database for provenance, but are not model inputs.
    clean_evidence = {
        "evidence_id": _clean_scalar(evidence_card.get("evidence_id")) or f"W{query_index}",
        "title": _clean_scalar(evidence_card.get("title")),
        "source_domain": _clean_scalar(evidence_card.get("domain")),
        "evidence_text": evidence_text,
    }
    if extracted_facts:
        clean_evidence["extracted_market_facts"] = extracted_facts
    clean_evidence = {
        key: value for key, value in clean_evidence.items()
        if value not in (None, "", [], {})
    }

    prompt = f"""[Prompt Revision]
{INCONTEXT_PROMPT_REVISION}

[Task]
{base_question}
Use the visible image and the paired context below to generate exactly one grounded business idea. Use only this Q{query_index}/W{query_index} pair.

[Domain Context]
domain: {domain_display}
domain_specific_focus: {domain_question}

[Image-Specific Annotation]
{_json_context(annotation)}

[Auxiliary Image Caption]
{caption}

[Business Lens]
lens: {lens}
lens_focus: {lens_focus}

[DDG Query]
Q{query_index}: {source_query}

[DDG Evidence]
{_json_context(clean_evidence)}

[Grounding Requirements]
- Treat the actual image as the primary visual source and the annotation as structured supporting context.
- Use the generated image caption only as auxiliary semantic context; it may be incomplete and must not override visible image evidence.
- Ground the idea in the DDG query and evidence text without copying the source wording.
- Make the target user, need, benefit, implementation, and differentiation concrete.
- Do not mention source URLs, citations, evidence IDs, or retrieval metadata in the final idea.
"""
    return clean_incontext_prompt_text(prompt)

def resolve_sample_domain(data: Dict[str, Any]) -> str:
    """Resolve the canonical six-domain label used for stratified splitting."""
    sample_id = _clean_scalar(data.get("sample_id"))
    source_row = _load_incontext_resources()["manifest_by_sample"].get(sample_id, {})
    domain = _clean_scalar(data.get("domain") or source_row.get("domain"))
    if domain not in EXPECTED_DOMAINS:
        raise RuntimeError(
            f"Unknown or missing domain for sample_id={sample_id or '<missing>'}: {domain!r}. "
            f"Expected one of {EXPECTED_DOMAINS}."
        )
    return domain


def extract_all_samples(data_dir: str, image_dir: str) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    files = sorted(name for name in os.listdir(data_dir) if name.endswith(".json"))
    for filename in files:
        path = os.path.join(data_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue

        image_path = resolve_expert_image_path(data, image_dir)
        if not image_path:
            continue
        domain = None

        for result in data.get("results", []):
            if not isinstance(result, dict):
                continue
            for item in result.get("data", []):
                if not isinstance(item, dict):
                    continue
                topic = clean_qwen_vl_text(item.get("topic") or item.get("title") or "")
                idea = item.get("idea", "")
                if not topic and not idea:
                    continue
                if domain is None:
                    domain = resolve_sample_domain(data)

                question = attach_idea_format_instruction(build_domain_aware_question(data, result, item, image_path))
                samples.append({
                    "image_path": image_path,
                    "domain": domain,
                    "question": question,
                    "expert_answers": format_structured_idea_json(item, topic=topic, idea=idea),
                })
    if not samples:
        raise RuntimeError("No valid SFT samples found.")
    return samples

def canonical_image_key(image_path: str) -> str:
    """Return a stable key used to keep every sample from one image together."""
    return os.path.realpath(os.path.abspath(image_path))


def count_unique_images(samples: List[Dict[str, Any]]) -> int:
    return len({canonical_image_key(item["image_path"]) for item in samples})


def _allocate_domain_test_image_counts(
    domain_to_images: Dict[str, List[str]],
    test_ratio: float,
) -> Dict[str, int]:
    """Allocate an exact global test size while staying proportional by domain.

    Every domain receives at least one test image and retains at least one train
    image. Remaining test slots are assigned by the largest quota deficit, which
    keeps each domain as close as possible to the requested test ratio.
    """
    domains = sorted(domain_to_images)
    for domain in domains:
        if len(domain_to_images[domain]) < 2:
            raise RuntimeError(
                f"Domain {domain!r} needs at least 2 unique images for a train/test split; "
                f"found {len(domain_to_images[domain])}."
            )

    total_images = sum(len(domain_to_images[domain]) for domain in domains)
    target_total = int(round(test_ratio * total_images))
    target_total = max(len(domains), target_total)
    target_total = min(
        target_total,
        sum(len(domain_to_images[domain]) - 1 for domain in domains),
    )

    ideal = {domain: test_ratio * len(domain_to_images[domain]) for domain in domains}
    counts = {
        domain: min(
            len(domain_to_images[domain]) - 1,
            max(1, int(math.floor(ideal[domain]))),
        )
        for domain in domains
    }

    while sum(counts.values()) < target_total:
        candidates = [
            domain for domain in domains
            if counts[domain] < len(domain_to_images[domain]) - 1
        ]
        if not candidates:
            raise RuntimeError("Unable to allocate the requested stratified test size.")
        chosen = max(
            candidates,
            key=lambda domain: (
                ideal[domain] - counts[domain],
                len(domain_to_images[domain]),
                domain,
            ),
        )
        counts[chosen] += 1

    while sum(counts.values()) > target_total:
        candidates = [domain for domain in domains if counts[domain] > 1]
        if not candidates:
            raise RuntimeError("Unable to reduce the stratified test allocation.")
        chosen = max(
            candidates,
            key=lambda domain: (
                counts[domain] - ideal[domain],
                counts[domain],
                domain,
            ),
        )
        counts[chosen] -= 1

    return counts


def deterministic_domain_stratified_image_group_split(
    samples: List[Dict[str, Any]],
    seed: int,
):
    """Create a deterministic domain-stratified image-level train/test split.

    The global test set is exactly as close as possible to FINAL_TEST_RATIO after
    integer rounding. Each of the six domains contributes approximately the same
    ratio of its unique images. Every row from one image stays in one partition,
    preventing image-level leakage across SFT, GRPO, and final evaluation.
    """
    image_to_domain: Dict[str, str] = {}
    domain_to_images: Dict[str, List[str]] = {domain: [] for domain in EXPECTED_DOMAINS}

    for item in samples:
        domain = _clean_scalar(item.get("domain"))
        if domain not in EXPECTED_DOMAINS:
            raise RuntimeError(
                f"Sample has unknown or missing domain {domain!r}; expected {EXPECTED_DOMAINS}."
            )
        image_key = canonical_image_key(item["image_path"])
        previous = image_to_domain.get(image_key)
        if previous is not None and previous != domain:
            raise RuntimeError(
                f"One image is assigned to multiple domains: {image_key} -> "
                f"{previous!r}, {domain!r}."
            )
        image_to_domain[image_key] = domain

    for image_key, domain in image_to_domain.items():
        domain_to_images[domain].append(image_key)

    missing_domains = [domain for domain in EXPECTED_DOMAINS if not domain_to_images[domain]]
    if missing_domains:
        raise RuntimeError(
            f"Domain-stratified split requires all six domains; missing={missing_domains}."
        )

    test_counts = _allocate_domain_test_image_counts(
        domain_to_images, FINAL_TEST_RATIO
    )
    train_images = set()
    test_images = set()

    for domain_index, domain in enumerate(sorted(EXPECTED_DOMAINS)):
        images = sorted(domain_to_images[domain])
        domain_rng = random.Random(seed + (domain_index + 1) * 1_000_003)
        domain_rng.shuffle(images)
        domain_test_count = test_counts[domain]
        test_images.update(images[:domain_test_count])
        train_images.update(images[domain_test_count:])

    all_images = set(image_to_domain)
    if train_images & test_images:
        raise RuntimeError("Domain-stratified image split overlap detected.")
    if train_images | test_images != all_images:
        raise RuntimeError("Domain-stratified split does not cover every image exactly once.")

    train_samples = [
        item for item in samples
        if canonical_image_key(item["image_path"]) in train_images
    ]
    final_test = [
        item for item in samples
        if canonical_image_key(item["image_path"]) in test_images
    ]
    if not train_samples or not final_test:
        raise RuntimeError("Domain-stratified split produced an empty train or test partition.")

    per_domain = {}
    for domain in EXPECTED_DOMAINS:
        domain_total_images = len(domain_to_images[domain])
        domain_test_images = test_counts[domain]
        domain_train_images = domain_total_images - domain_test_images
        domain_total_samples = sum(1 for item in samples if item["domain"] == domain)
        domain_train_samples = sum(1 for item in train_samples if item["domain"] == domain)
        domain_test_samples = sum(1 for item in final_test if item["domain"] == domain)
        per_domain[domain] = {
            "total_images": domain_total_images,
            "train_images": domain_train_images,
            "test_images": domain_test_images,
            "test_image_ratio": domain_test_images / domain_total_images,
            "total_samples": domain_total_samples,
            "train_samples": domain_train_samples,
            "test_samples": domain_test_samples,
        }

    split_info = {
        "split_revision": DOMAIN_STRATIFIED_SPLIT_REVISION,
        "requested_test_ratio": FINAL_TEST_RATIO,
        "total_images": len(all_images),
        "train_images": len(train_images),
        "test_images": len(test_images),
        "actual_test_ratio": len(test_images) / len(all_images),
        "per_domain": per_domain,
    }
    return train_samples, final_test, split_info

def test_manifest_payload(final_test: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {
            "image_path": item["image_path"],
            "domain": item["domain"],
            "question": item["question"],
            "expert_answers": item["expert_answers"],
        }
        for item in final_test
    ]


def verify_or_create_test_manifest(
    final_test: List[Dict[str, Any]],
    overwrite: bool = False,
) -> None:
    """Write the canonical domain-stratified image-held-out test manifest.

    SFT is the first training stage, so it owns creation of this experiment-level
    artifact. GRPO and Eval must only load and verify the same file.
    """
    payload = test_manifest_payload(final_test)
    os.makedirs(os.path.dirname(TEST_FILE), exist_ok=True)

    if os.path.exists(TEST_FILE):
        with open(TEST_FILE, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing == payload:
            return
        if not overwrite:
            raise RuntimeError(
                "Existing test manifest is incompatible with the domain-stratified image split: "
                f"{TEST_FILE}. Re-run SFT once with --overwrite_test_manifest "
                "or remove the old row-level manifest explicitly."
            )
        print(f"Overwriting incompatible test manifest: {TEST_FILE}")

    temporary_path = TEST_FILE + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary_path, TEST_FILE)


def write_split_summary(split_info: Dict[str, Any]) -> None:
    """Persist reproducible per-domain image/sample counts for reporting."""
    os.makedirs(os.path.dirname(TEST_SPLIT_SUMMARY_FILE), exist_ok=True)
    temporary_path = TEST_SPLIT_SUMMARY_FILE + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(split_info, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary_path, TEST_SPLIT_SUMMARY_FILE)


def export_prompt_preview(
    all_samples: List[Dict[str, Any]],
    train_samples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    output_path: str,
    print_limit: int,
) -> None:
    """Export the exact user questions consumed by SFT/GRPO/Eval."""
    train_ids = {id(sample) for sample in train_samples}
    test_ids = {id(sample) for sample in test_samples}
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    rows = []
    for index, sample in enumerate(all_samples):
        split = "train" if id(sample) in train_ids else ("test" if id(sample) in test_ids else "unknown")
        rows.append({
            "index": index,
            "split": split,
            "image_path": sample["image_path"],
            "domain": sample["domain"],
            "question": sample["question"],
            "expert_answers": sample["expert_answers"],
        })

    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Exported exact input prompts: {len(rows)} -> {output_path}")
    limit = len(rows) if print_limit < 0 else min(max(0, print_limit), len(rows))
    for row in rows[:limit]:
        print("\n" + "=" * 100)
        print(f"INDEX={row['index']} SPLIT={row['split']}")
        print(f"IMAGE={row['image_path']}")
        print(row["question"])

class IdeationDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], processor: Qwen2VLProcessor):
        self.samples = samples
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = load_policy_image(sample["image_path"])
        user_messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["question"]},
            ],
        }]
        full_messages = user_messages + [{
            "role": "assistant",
            "content": sample["expert_answers"],
        }]

        prompt_text = self.processor.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        prompt_inputs = self.processor(
            text=[prompt_text], images=[image], return_tensors="pt"
        )
        full_inputs = self.processor(
            text=[full_text], images=[image], return_tensors="pt"
        )

        labels = full_inputs["input_ids"].clone()
        prompt_len = int(prompt_inputs["attention_mask"].sum().item())
        labels[:, :prompt_len] = -100
        labels[full_inputs["attention_mask"] == 0] = -100
        return {
            "input_ids": full_inputs["input_ids"][0],
            "attention_mask": full_inputs["attention_mask"][0],
            "pixel_values": full_inputs["pixel_values"],
            "image_grid_thw": full_inputs["image_grid_thw"],
            "labels": labels[0],
        }


@dataclass
class QwenVLCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [item["input_ids"] for item in features],
                batch_first=True,
                padding_value=self.pad_token_id,
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                [item["attention_mask"] for item in features],
                batch_first=True,
                padding_value=0,
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                [item["labels"] for item in features],
                batch_first=True,
                padding_value=-100,
            ),
            "pixel_values": torch.cat([item["pixel_values"] for item in features], dim=0),
            "image_grid_thw": torch.cat([item["image_grid_thw"] for item in features], dim=0),
        }


def run_sft(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    # Fix the canonical image-level train/test split before model loading.
    all_samples = extract_all_samples(args.data_dir, args.image_dir)
    train_samples, final_test, split_info = deterministic_domain_stratified_image_group_split(
        all_samples, args.seed
    )
    process_rank = int(os.environ.get("RANK", "0"))
    if process_rank == 0:
        verify_or_create_test_manifest(
            final_test, overwrite=args.overwrite_test_manifest
        )
        write_split_summary(split_info)
        print(
            f"Samples: total={len(all_samples)} | train={len(train_samples)} | "
            f"final_test={len(final_test)}\n"
            f"Images: total={split_info['total_images']} | "
            f"train={split_info['train_images']} | "
            f"final_test={split_info['test_images']} | "
            f"test_ratio={split_info['actual_test_ratio']:.4f} | "
            f"split={split_info['split_revision']}"
        )
        for domain in EXPECTED_DOMAINS:
            stats = split_info["per_domain"][domain]
            print(
                f"  [Domain Split] {domain}: images "
                f"{stats['train_images']} train / {stats['test_images']} test "
                f"({stats['test_image_ratio']:.4f} test) | samples "
                f"{stats['train_samples']} train / {stats['test_samples']} test"
            )

    if args.preview_prompts_only:
        if process_rank == 0:
            export_prompt_preview(
                all_samples,
                train_samples,
                final_test,
                args.preview_prompts_path,
                args.preview_print_limit,
            )
        return

    if args.prepare_split_only:
        if process_rank == 0:
            print(f"Split manifest ready: {TEST_FILE}")
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in (4, 8):
        raise RuntimeError(
            f"This final configuration is tuned for 4 or 8 DDP processes; found WORLD_SIZE={world_size}."
        )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    processor = Qwen2VLProcessor.from_pretrained(args.model_id, use_fast=False)
    processor.tokenizer.padding_side = "right"
    processor.image_processor.min_pixels = POLICY_MIN_PIXELS
    processor.image_processor.max_pixels = POLICY_MAX_PIXELS
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    try:
        model.enable_input_require_grads()
    except Exception:
        pass

    if args.per_device_batch_size > 0:
        per_device = args.per_device_batch_size
    else:
        min_gpu_gib = min(
            int(torch.cuda.get_device_properties(index).total_memory // (1024 ** 3))
            for index in range(torch.cuda.device_count())
        )
        per_device = 4 if min_gpu_gib >= 70 else (2 if min_gpu_gib >= 38 else 1)
    num_workers = args.num_workers
    if num_workers < 0:
        cpu_per_rank = max(1, (os.cpu_count() or 8) // world_size)
        num_workers = min(4, max(2, cpu_per_rank // 2))
    grad_accum = max(1, math.ceil(args.target_global_batch_size / (world_size * per_device)))
    actual_global_batch = world_size * per_device * grad_accum

    if int(os.environ.get("RANK", "0")) == 0:
        model.print_trainable_parameters()
        print(
            f"DDP world_size={world_size} | per_device_batch={per_device} | "
            f"grad_accum={grad_accum} | effective_global_batch={actual_global_batch} | "
            f"dataloader_workers/rank={num_workers}"
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        logging_steps=args.logging_steps,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=25,
        remove_unused_columns=False,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=num_workers > 0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=IdeationDataset(train_samples, processor),
        data_collator=QwenVLCollator(processor.tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        os.makedirs(args.output_dir, exist_ok=True)
        trainer.model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"Saved final SFT adapter -> {args.output_dir}")


if __name__ == "__main__":
    run_sft(parse_args())
