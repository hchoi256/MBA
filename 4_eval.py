#!/usr/bin/env python3
"""Independent InternVL PBIG-6 evaluation for the MBA SFT -> GRPO pipeline.

Core guarantees
---------------
1. The held-out manifest is loaded and verified through the active 3_grpo.py.
2. Base, SFT, GRPO, and GPT generators receive the exact same canonical question.
3. Local Qwen generators use the same image preprocessing, prompt builder, and
   rollout decoding defaults as GRPO training.
4. GPT receives the same resized policy image and the exact same question text.
5. GRPO training keeps its frozen Qwen2-VL-72B reward Judge, while final
   evaluation uses an independent OpenGVLab/InternVL2_5-78B Judge.
6. Generators receive the auxiliary image caption inside the canonical question.
7. The evaluation Judge receives the same sample-matched auxiliary image caption inside
   the leakage-free neutral context, executing six independent single-axis absolute-scoring calls per candidate.
8. Evaluation is branch-invariant: open- and closed-branch GRPO adapters are both
   evaluated with the same reference-free PBIG-6 protocol.
9. GT/reference/expert answers and the training-only creativity/feasibility
   reference context are never sent to any PBIG evaluation Judge.
10. Overall and per-domain summaries are produced from the same domain-stratified
   held-out manifest created by 2_sft.py.

Examples
--------
Base:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -u 4_eval_6m.py --model_type base

SFT:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -u 4_eval_6m.py --model_type sft

GRPO:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -u 4_eval_6m.py --model_type grpo

GPT:
  export OPENAI_API_KEY=...
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -u 4_eval_6m.py --model_type gpt --gpt_model gpt-4o

The GPT model string is passed to the OpenAI Responses API. Convenience aliases
such as gpt4o, gpt5, and gpt5-mini are normalized to gpt-4o, gpt-5, and
gpt-5-mini respectively.
"""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import importlib.util
import io
import json
import math
import mimetypes
import os
import random
import re
import shutil
import statistics
import sys
import threading
import time
import types
from datetime import datetime, timezone
from functools import lru_cache
import warnings
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import transformers
from PIL import Image
from tqdm import tqdm
from packaging import version as packaging_version

if not hasattr(torch, "compiler"):
    torch.compiler = types.SimpleNamespace(is_compiling=lambda: False)
elif not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

warnings.filterwarnings(
    "ignore",
    message=r"`do_sample` is set to `False`\. However, `(temperature|top_p|top_k)` is set.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"MatMul8bitLt: inputs will be cast.*",
    category=UserWarning,
)

os.environ.setdefault("ACCELERATE_USE_DEEPSPEED", "false")
os.environ.setdefault("DISABLE_DEEPSPEED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PIPELINE_REVISION = "pbig6-internvl2_5-78b-branch-invariant-reference-free-caption-v6"
EXPECTED_INCONTEXT_PROMPT_REVISION = "mba_clean_ddg_caption_incontext_v3"
EXPECTED_SPLIT_REVISION = "domain_stratified_image_v1"

# ---------------------------------------------------------------------------
# Load the active GRPO module. This intentionally shares exact prompt/context,
# image preprocessing, LoRA loading, checkpoint discovery, and rubric constants.
# ---------------------------------------------------------------------------
sys.path.insert(0, "/workspace")
GRPO_TRAIN_PATH = os.environ.get("GRPO_TRAIN_PATH", "/workspace/3_grpo.py")
if not os.path.exists(GRPO_TRAIN_PATH):
    fallback = "/workspace/3_grpo_final.py"
    if os.path.exists(fallback):
        GRPO_TRAIN_PATH = fallback
    else:
        raise FileNotFoundError(
            f"Active GRPO module not found: {GRPO_TRAIN_PATH}. "
            "Place the compatible 3_grpo.py in /workspace or set GRPO_TRAIN_PATH."
        )

_spec = importlib.util.spec_from_file_location("mba_active_grpo", GRPO_TRAIN_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot import GRPO module from {GRPO_TRAIN_PATH}")
grpo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grpo)

_REQUIRED_GRPO_ATTRIBUTES = (
    "MODEL_ID",
    "SFT_LORA_DIR",
    "OUTPUT_DIR",
    "TEST_FILE",
    "DATA_DIR",
    "IMAGE_DIR",
    "INCONTEXT_PROMPT_REVISION",
    "DOMAIN_STRATIFIED_SPLIT_REVISION",
    "EXPECTED_DOMAINS",
    "SIX_DIMENSION_WEIGHTS",
    "SIX_DIMENSION_RUBRICS",
    "PBIG_DIMENSION_ORDER",
    "JUDGE_MODEL_ID",
    "MAX_NEW_TOKENS",
    "SEED",
    "load_image",
    "make_prompt_text",
    "clean_qwen_vl_text",
    "clean_incontext_prompt_text",
    "format_idea_for_judge",
    "build_judge_evaluation_context",
    "build_single_dimension_ranking_judge_text",
    "build_creativity_ranking_judge_text",
    "load_lora_adapter_compat",
    "extract_all_samples",
    "partition_samples_by_test_manifest",
    "test_manifest_payload",
    "canonical_image_key",
)
_missing_attrs = [name for name in _REQUIRED_GRPO_ATTRIBUTES if not hasattr(grpo, name)]
if _missing_attrs:
    raise RuntimeError(
        "The active 3_grpo.py is incompatible with this evaluator; missing: "
        + ", ".join(_missing_attrs)
    )
if grpo.INCONTEXT_PROMPT_REVISION != EXPECTED_INCONTEXT_PROMPT_REVISION:
    raise RuntimeError(
        "Prompt revision mismatch: "
        f"eval={EXPECTED_INCONTEXT_PROMPT_REVISION}, "
        f"grpo={grpo.INCONTEXT_PROMPT_REVISION}"
    )
if grpo.DOMAIN_STRATIFIED_SPLIT_REVISION != EXPECTED_SPLIT_REVISION:
    raise RuntimeError(
        "Split revision mismatch: "
        f"eval={EXPECTED_SPLIT_REVISION}, "
        f"grpo={grpo.DOMAIN_STRATIFIED_SPLIT_REVISION}"
    )

BASE_MODEL_ID = grpo.MODEL_ID
SFT_LORA_DIR = grpo.SFT_LORA_DIR
GRPO_LORA_DIR = grpo.OUTPUT_DIR
TEST_FILE = grpo.TEST_FILE
TRAINING_JUDGE_MODEL_ID = grpo.JUDGE_MODEL_ID
EVAL_JUDGE_MODEL_ID = "OpenGVLab/InternVL2_5-78B"
JUDGE_MODEL_ID = EVAL_JUDGE_MODEL_ID
INTERNVL_TRANSFORMERS_VERSION = "4.45.2"
INTERNVL_IMAGE_SIZE = 448
INTERNVL_MAX_TILES = 12
INTERNVL_MEAN = (0.485, 0.456, 0.406)
INTERNVL_STD = (0.229, 0.224, 0.225)
SEED = int(grpo.SEED)
EXPECTED_DOMAINS = tuple(grpo.EXPECTED_DOMAINS)
PBIG_DIMENSION_ORDER = tuple(grpo.PBIG_DIMENSION_ORDER)
TRAIN_ALIGNED_DIMENSION_WEIGHTS = OrderedDict(
    (name, float(grpo.SIX_DIMENSION_WEIGHTS[name]))
    for name in PBIG_DIMENSION_ORDER
)

# The absolute scales are the PBIG-6 evaluation scales used by the existing
# evaluation scripts. The axis definitions are imported from 3_grpo.py.
PBIG_DIMENSIONS = OrderedDict([
    ("specificity", {
        "display_name": "Specificity",
        "scale_min": 1,
        "scale_max": 4,
        "levels": {
            1: "Not coherent enough to identify a business idea.",
            2: "Broad or ambiguous; target user, need, offering, or mechanism is largely missing.",
            3: "A concrete business idea is identifiable, but one or more important elements remain underspecified.",
            4: "One clearly defined offering with a specific target user, need, benefit, and operating mechanism.",
        },
    }),
    ("technical_validity", {
        "display_name": "Technical validity",
        "scale_min": 1,
        "scale_max": 4,
        "levels": {
            1: "Technically or operationally unsuitable, impossible, or unsupported.",
            2: "A prototype is challenging but potentially possible with major unresolved assumptions.",
            3: "A credible prototype can be built with present-day technology and operations.",
            4: "The idea is credible for production deployment, integration, and operation.",
        },
    }),
    ("innovativeness", {
        "display_name": "Innovativeness",
        "scale_min": 1,
        "scale_max": 5,
        "levels": {
            1: "A well-known or generic application with little novelty.",
            2: "A familiar use case or wrapper with limited novelty.",
            3: "Some non-obvious combination or application, but only moderate originality.",
            4: "A surprising and meaningfully novel mechanism, workflow, combination, or application.",
            5: "Clearly distinctive and potentially field-shaping innovation.",
        },
    }),
    ("competitive_advantage", {
        "display_name": "Competitive advantage",
        "scale_min": 1,
        "scale_max": 4,
        "levels": {
            1: "Easy to imitate and lacks an essential differentiated capability.",
            2: "Some essential capability is present, but the idea remains easy to imitate.",
            3: "Meaningfully hard to imitate, but the essential core capability or defensibility is incomplete.",
            4: "Hard to imitate with a strong essential moat such as proprietary data, workflow integration, switching costs, specialized models, know-how, or network effects.",
        },
    }),
    ("need_validity", {
        "display_name": "Need validity",
        "scale_min": 0,
        "scale_max": 3,
        "levels": {
            0: "Not a credible B2B need or not a business idea.",
            1: "Weak, speculative, or low-urgency need with limited business impact.",
            2: "Clear business need with meaningful monetary or qualitative impact.",
            3: "Clear and urgent need with both strong monetary and strategic/qualitative impact.",
        },
    }),
    ("market_size", {
        "display_name": "Market size",
        "scale_min": 0,
        "scale_max": 3,
        "levels": {
            0: "No credible reachable market or not a business idea.",
            1: "Very narrow, one-off, or weakly reachable market.",
            2: "Meaningful beachhead B2B segment with plausible adoption.",
            3: "Large reachable market with a clear beachhead and credible expansion path.",
        },
    }),
])

if tuple(PBIG_DIMENSIONS) != PBIG_DIMENSION_ORDER:
    raise RuntimeError(
        f"PBIG dimension order mismatch: eval={tuple(PBIG_DIMENSIONS)}, "
        f"grpo={PBIG_DIMENSION_ORDER}"
    )


# ---------------------------------------------------------------------------
# Independent InternVL2.5-78B Judge helpers
# ---------------------------------------------------------------------------
def _internvl_normalize_image(
    image: Image.Image,
    input_size: int = INTERNVL_IMAGE_SIZE,
) -> torch.Tensor:
    image = image.convert("RGB").resize(
        (input_size, input_size), Image.Resampling.BICUBIC
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = torch.tensor(INTERNVL_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(INTERNVL_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def _internvl_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Sequence[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    best_ratio = (1, 1)
    best_diff = float("inf")
    area = width * height
    for ratio in target_ratios:
        target_aspect = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_aspect)
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio
        elif diff == best_diff:
            target_area = (
                0.5 * image_size * image_size * ratio[0] * ratio[1]
            )
            if area > target_area:
                best_ratio = ratio
    return best_ratio


def _internvl_dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = INTERNVL_MAX_TILES,
    image_size: int = INTERNVL_IMAGE_SIZE,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    original_width, original_height = image.size
    aspect_ratio = original_width / max(original_height, 1)
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    target_ratio = _internvl_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        original_width,
        original_height,
        image_size,
    )
    target_width = image_size * target_ratio[0]
    target_height = image_size * target_ratio[1]
    block_count = target_ratio[0] * target_ratio[1]
    resized = image.resize(
        (target_width, target_height), Image.Resampling.BICUBIC
    )

    tiles: List[Image.Image] = []
    tiles_per_row = target_width // image_size
    for index in range(block_count):
        left = (index % tiles_per_row) * image_size
        upper = (index // tiles_per_row) * image_size
        tiles.append(
            resized.crop(
                (left, upper, left + image_size, upper + image_size)
            )
        )
    if use_thumbnail and len(tiles) != 1:
        tiles.append(
            image.resize(
                (image_size, image_size), Image.Resampling.BICUBIC
            )
        )
    return tiles


def load_internvl_pixel_values(
    image_path: str,
    input_size: int = INTERNVL_IMAGE_SIZE,
    max_num: int = INTERNVL_MAX_TILES,
) -> torch.Tensor:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    tiles = _internvl_dynamic_preprocess(
        image,
        image_size=input_size,
        max_num=max_num,
        use_thumbnail=True,
    )
    return torch.stack(
        [_internvl_normalize_image(tile, input_size) for tile in tiles]
    )


@lru_cache(maxsize=8)
def _cached_internvl_pixels(image_path: str) -> torch.Tensor:
    # CPU tensor cache only. A copy is transferred to the Judge input GPU.
    return load_internvl_pixel_values(image_path)


def assert_internvl_transformers_compatibility() -> str:
    current = packaging_version.parse(transformers.__version__.split("+")[0])
    required = packaging_version.parse(INTERNVL_TRANSFORMERS_VERSION)
    if current != required:
        raise RuntimeError(
            "This tested InternVL2.5-78B pipeline requires transformers=="
            f"{INTERNVL_TRANSFORMERS_VERSION}, but found "
            f"{transformers.__version__}. Install matching transformers and "
            "tokenizers==0.20.1 before evaluation."
        )
    return transformers.__version__


def internvl_max_memory(judge_gpus: Sequence[int], reserve_gib: int = 4):
    memory = {}
    for gpu_id in judge_gpus:
        total_gib = int(
            torch.cuda.get_device_properties(int(gpu_id)).total_memory
            // (1024 ** 3)
        )
        memory[int(gpu_id)] = f"{max(8, total_gib - reserve_gib)}GiB"
    return memory


def split_internvl_78b(
    judge_gpus: Sequence[int],
) -> Dict[str, int]:
    """Official-style InternVL split generalized to a four-GPU worker group."""
    from transformers import AutoConfig

    device_ids = tuple(int(value) for value in judge_gpus)
    if len(device_ids) < 2:
        raise ValueError(
            "InternVL2.5-78B evaluation requires at least two Judge GPUs."
        )

    config = AutoConfig.from_pretrained(
        EVAL_JUDGE_MODEL_ID, trust_remote_code=True
    )
    llm_config = getattr(config, "llm_config", None)
    num_layers = int(getattr(llm_config, "num_hidden_layers", 80))

    # GPU 0 also owns the vision encoder, so it receives approximately half
    # the language-layer share of the other devices.
    layers_per_gpu = math.ceil(num_layers / (len(device_ids) - 0.5))
    allocation = [layers_per_gpu] * len(device_ids)
    allocation[0] = math.ceil(allocation[0] * 0.5)

    device_map: Dict[str, int] = {}
    layer_index = 0
    for slot, layer_count in enumerate(allocation):
        gpu_id = device_ids[slot]
        for _ in range(layer_count):
            if layer_index >= num_layers:
                break
            device_map[
                f"language_model.model.layers.{layer_index}"
            ] = gpu_id
            layer_index += 1

    first_gpu = device_ids[0]
    device_map.update({
        "vision_model": first_gpu,
        "mlp1": first_gpu,
        "language_model.model.embed_tokens": first_gpu,
        "language_model.model.rotary_emb": first_gpu,
        "language_model.model.norm": first_gpu,
        "language_model.lm_head": first_gpu,
        f"language_model.model.layers.{num_layers - 1}": first_gpu,
    })
    if layer_index != num_layers:
        raise RuntimeError(
            f"InternVL device-map assignment failed: "
            f"{layer_index}/{num_layers} layers"
        )
    return device_map


def _move_module_buffers_to_device(module, device):
    target = torch.device(device)
    for name, buffer in list(module.named_buffers(recurse=False)):
        if buffer is not None and buffer.device != target:
            setattr(module, name, buffer.to(device=target))
    for name in ("inv_freq", "original_inv_freq"):
        value = getattr(module, name, None)
        if torch.is_tensor(value) and value.device != target:
            setattr(module, name, value.to(device=target))
    return module


def pin_internvl_runtime_buffers(model, target_device) -> str:
    language_model = getattr(model, "language_model", None)
    core = getattr(language_model, "model", None)
    rotary = getattr(core, "rotary_emb", None)
    if rotary is None:
        raise RuntimeError("InternVL Qwen2 rotary_emb module was not found.")
    _move_module_buffers_to_device(rotary, target_device)
    inv_freq = getattr(rotary, "inv_freq", None)
    if (
        not torch.is_tensor(inv_freq)
        or inv_freq.device != torch.device(target_device)
    ):
        raise RuntimeError(
            f"Failed to pin InternVL rotary buffer to {target_device}; "
            f"found={getattr(inv_freq, 'device', None)}"
        )
    return str(inv_freq.device)


GPT_MODEL_ALIASES = {
    "gpt4o": "gpt-4o",
    "gpt-4o": "gpt-4o",
    "gpt4.1": "gpt-4.1",
    "gpt-4.1": "gpt-4.1",
    "gpt5": "gpt-5",
    "gpt-5": "gpt-5",
    "gpt5-mini": "gpt-5-mini",
    "gpt-5-mini": "gpt-5-mini",
}
_OPENAI_THREAD_LOCAL = threading.local()


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def clean_text(text: Any) -> str:
    return grpo.clean_qwen_vl_text("" if text is None else str(text))


def safe_slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")
    return value or "model"


def stable_sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def atomic_json_dump(payload: Any, path: str, indent: Optional[int] = 2) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    temporary = path_obj.with_suffix(path_obj.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)
    os.replace(temporary, path_obj)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.mean(clean) if clean else None


def normalize_score_0_100(dimension: str, score: int) -> float:
    meta = PBIG_DIMENSIONS[dimension]
    lo, hi = int(meta["scale_min"]), int(meta["scale_max"])
    return 100.0 * (float(score) - lo) / float(hi - lo)


def parse_idea_json(text: str) -> Optional[Dict[str, str]]:
    raw = clean_text(text)
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return None
    fields = ("title", "description", "implementation", "differentiation")
    if not isinstance(obj, dict) or set(obj) != set(fields):
        return None
    if not all(isinstance(obj[field], str) and clean_text(obj[field]) for field in fields):
        return None
    return {field: clean_text(obj[field]) for field in fields}


def generation_config_payload(args: argparse.Namespace) -> Dict[str, Any]:
    if args.model_type == "gpt":
        return {
            "provider": "openai_responses",
            "model": args.gpt_model,
            "max_output_tokens": args.gpt_max_output_tokens,
            "temperature": None if args.gpt_omit_sampling_params else args.gpt_temperature,
            "top_p": None if args.gpt_omit_sampling_params else args.gpt_top_p,
            "reasoning_effort": args.gpt_reasoning_effort,
            "image_detail": args.gpt_image_detail,
        }
    return {
        "provider": "local_transformers",
        "base_model": BASE_MODEL_ID,
        "adapter": resolve_local_adapter_path(args),
        "max_new_tokens": args.local_max_new_tokens,
        "do_sample": True,
        "temperature": args.local_temperature,
        "top_p": args.local_top_p,
        "top_k": args.local_top_k,
        "repetition_penalty": args.local_repetition_penalty,
    }


# ---------------------------------------------------------------------------
# Held-out manifest verification
# ---------------------------------------------------------------------------
def load_verified_test_manifest(max_samples: int = 0) -> List[Dict[str, Any]]:
    if not os.path.exists(TEST_FILE):
        raise RuntimeError(
            f"Missing held-out manifest: {TEST_FILE}. "
            "Run the compatible 2_sft.py first."
        )

    all_samples = grpo.extract_all_samples(grpo.DATA_DIR, grpo.IMAGE_DIR)
    train_samples, test_samples = grpo.partition_samples_by_test_manifest(
        all_samples, TEST_FILE
    )
    payload = grpo.test_manifest_payload(test_samples)

    if not train_samples or not payload:
        raise RuntimeError("Verified split contains an empty train or test partition.")

    domains = Counter()
    test_images = set()
    train_images = {
        grpo.canonical_image_key(item["image_path"]) for item in train_samples
    }
    for index, item in enumerate(payload):
        required = {"image_path", "domain", "question", "expert_answers"}
        if not required.issubset(item):
            raise RuntimeError(
                f"Malformed held-out row {index}; missing={sorted(required - set(item))}"
            )
        domain = clean_text(item["domain"])
        if domain not in EXPECTED_DOMAINS:
            raise RuntimeError(f"Unknown held-out domain at row {index}: {domain!r}")
        domains[domain] += 1
        image_key = grpo.canonical_image_key(item["image_path"])
        test_images.add(image_key)
        if image_key in train_images:
            raise RuntimeError(f"Image leakage detected: {item['image_path']}")

        question = str(item["question"])
        required_sections = (
            f"[Prompt Revision]\n{EXPECTED_INCONTEXT_PROMPT_REVISION}",
            "[Task]",
            "[Domain Context]",
            "[Image-Specific Annotation]",
            "[Auxiliary Image Caption]",
            "[Business Lens]",
            "[DDG Query]",
            "[DDG Evidence]",
            "[Grounding Requirements]",
            '"title"',
            '"description"',
            '"implementation"',
            '"differentiation"',
        )
        missing_sections = [
            section for section in required_sections if section not in question
        ]
        if missing_sections:
            raise RuntimeError(
                f"Non-canonical generation prompt at held-out row {index}; "
                f"missing={missing_sections}"
            )

        # This is also a hard leakage check: no full-prompt fallback is permitted.
        grpo.build_judge_evaluation_context(question)

    missing_domains = [domain for domain in EXPECTED_DOMAINS if domains[domain] == 0]
    if missing_domains:
        raise RuntimeError(
            f"Held-out manifest does not cover all six domains: {missing_domains}"
        )

    if max_samples > 0:
        payload = payload[: int(max_samples)]

    print(
        f"[Manifest] verified={TEST_FILE} | samples={len(payload)} | "
        f"unique_test_images={len(test_images)} | split={EXPECTED_SPLIT_REVISION}",
        flush=True,
    )
    print(
        "[Manifest domains] "
        + " | ".join(f"{domain}={domains[domain]}" for domain in EXPECTED_DOMAINS),
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# Generator selection and exact shared prompt
# ---------------------------------------------------------------------------
def _valid_adapter_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    config = os.path.join(path, "adapter_config.json")
    weights = (
        os.path.join(path, "adapter_model.safetensors"),
        os.path.join(path, "adapter_model.bin"),
    )
    return os.path.isfile(config) and any(os.path.isfile(item) for item in weights)


def resolve_local_adapter_path(args: argparse.Namespace) -> Optional[str]:
    if args.model_type == "base":
        return None
    if args.model_type == "sft":
        path = args.sft_adapter or SFT_LORA_DIR
        if not _valid_adapter_dir(path):
            raise RuntimeError(f"Invalid SFT adapter directory: {path}")
        return path
    if args.model_type == "grpo":
        requested = args.grpo_adapter or GRPO_LORA_DIR
        if _valid_adapter_dir(requested):
            return requested
        latest = grpo.find_latest_resume_checkpoint(requested)
        if latest and _valid_adapter_dir(latest):
            print(
                f"[GRPO adapter] final root not found; using latest checkpoint: {latest}",
                flush=True,
            )
            return latest
        raise RuntimeError(
            f"No valid GRPO adapter found at {requested} or checkpoint-N beneath it."
        )
    return None


def exact_generation_question(item: Dict[str, Any]) -> str:
    # The manifest already stores the exact SFT/GRPO generation prompt.
    question = str(item["question"])
    if not question.strip():
        raise RuntimeError("Encountered empty generation question.")
    return question


def local_generation_worker(
    worker_rank: int,
    shard: List[Tuple[int, Dict[str, Any]]],
    model_type: str,
    adapter_path: Optional[str],
    args_dict: Dict[str, Any],
    output_path: str,
) -> None:
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

    args = argparse.Namespace(**args_dict)
    device = torch.device(f"cuda:{worker_rank}")
    torch.cuda.set_device(worker_rank)

    processor = Qwen2VLProcessor.from_pretrained(BASE_MODEL_ID, use_fast=False)
    processor.tokenizer.padding_side = "left"
    processor.image_processor.min_pixels = int(grpo.MIN_PIXELS)
    processor.image_processor.max_pixels = int(grpo.MAX_PIXELS)

    print(
        f"[Generation worker {worker_rank}] loading {BASE_MODEL_ID} "
        f"model_type={model_type} adapter={adapter_path}",
        flush=True,
    )
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": worker_rank},
    )
    if adapter_path:
        model = grpo.load_lora_adapter_compat(
            base_model=base_model,
            adapter_dir=adapter_path,
            adapter_name="default",
            is_trainable=False,
        )
    else:
        model = base_model
    model.eval()

    records: List[Dict[str, Any]] = []
    for global_idx, item in tqdm(
        shard,
        desc=f"[Generation GPU{worker_rank}]",
        position=worker_rank,
        leave=True,
        dynamic_ncols=True,
    ):
        question = exact_generation_question(item)
        image = grpo.load_image(item["image_path"])
        prompt_text = grpo.make_prompt_text(
            processor, item["image_path"], question
        )
        inputs = processor(
            text=[prompt_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(device)

        sample_seed = int(args.seed) + int(global_idx)
        torch.manual_seed(sample_seed)
        torch.cuda.manual_seed(sample_seed)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=int(args.local_max_new_tokens),
                do_sample=True,
                temperature=float(args.local_temperature),
                top_p=float(args.local_top_p),
                top_k=int(args.local_top_k),
                repetition_penalty=float(args.local_repetition_penalty),
                num_return_sequences=1,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
            )

        input_width = int(inputs["input_ids"].shape[1])
        answer = processor.tokenizer.decode(
            generated[0, input_width:],
            skip_special_tokens=True,
        ).strip()
        if not answer:
            raise RuntimeError(f"Empty local generation at idx={global_idx}")

        records.append({
            "idx": int(global_idx),
            "domain": item["domain"],
            "image_path": item["image_path"],
            "question_sha256": stable_sha256_text(question),
            "generator_type": model_type,
            "generator_id": adapter_path or BASE_MODEL_ID,
            "Generated_Answer": answer,
            "valid_structured_json": parse_idea_json(answer) is not None,
        })
        del inputs, generated

    atomic_json_dump(records, output_path, indent=None)
    del model
    if adapter_path:
        del base_model
    torch.cuda.empty_cache()
    gc.collect()


def _openai_client(timeout_seconds: int):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI Python SDK is required for --model_type gpt. "
            "Install it with: pip install -U openai"
        ) from exc
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    cached = getattr(_OPENAI_THREAD_LOCAL, "client", None)
    cached_timeout = getattr(_OPENAI_THREAD_LOCAL, "timeout_seconds", None)
    if cached is None or cached_timeout != timeout_seconds:
        cached = OpenAI(api_key=api_key, timeout=timeout_seconds)
        _OPENAI_THREAD_LOCAL.client = cached
        _OPENAI_THREAD_LOCAL.timeout_seconds = timeout_seconds
    return cached


def normalize_gpt_model_name(name: str) -> str:
    stripped = str(name).strip()
    return GPT_MODEL_ALIASES.get(stripped.lower(), stripped)


def policy_image_data_url(image_path: str) -> str:
    # Use the exact same 512px policy preprocessing as local generators.
    image = grpo.load_image(image_path)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def extract_openai_response_text(response: Any) -> str:
    direct = getattr(response, "output_text", None)
    if direct:
        return str(direct).strip()
    output = response.get("output", []) if isinstance(response, dict) else (
        getattr(response, "output", []) or []
    )
    parts: List[str] = []
    for item in output:
        content = item.get("content", []) if isinstance(item, dict) else (
            getattr(item, "content", []) or []
        )
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("output_text")
            else:
                text = getattr(block, "text", None) or getattr(
                    block, "output_text", None
                )
            if text:
                parts.append(str(text).strip())
    return "\n".join(part for part in parts if part).strip()


def call_openai_with_retries(
    kwargs: Dict[str, Any],
    timeout_seconds: int,
    max_retries: int,
    base_sleep: float,
) -> Any:
    last_error: Optional[Exception] = None
    omit_sampling = False
    omit_reasoning = False

    for attempt in range(max_retries + 1):
        request = dict(kwargs)
        if omit_sampling:
            request.pop("temperature", None)
            request.pop("top_p", None)
        if omit_reasoning:
            request.pop("reasoning", None)
        try:
            return _openai_client(timeout_seconds).responses.create(**request)
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            if (
                not omit_sampling
                and ("temperature" in message or "top_p" in message)
                and any(token in message for token in (
                    "unsupported", "not supported", "unknown parameter", "invalid"
                ))
            ):
                omit_sampling = True
                continue
            if (
                not omit_reasoning
                and "reasoning" in message
                and any(token in message for token in (
                    "unsupported", "not supported", "unknown parameter", "invalid"
                ))
            ):
                omit_reasoning = True
                continue
            if attempt >= max_retries:
                break
            delay = min(60.0, base_sleep * (2 ** attempt)) + random.random()
            time.sleep(delay)

    raise RuntimeError(
        f"OpenAI Responses API failed after retries: {last_error}"
    ) from last_error


def generate_gpt_one(
    idx: int,
    item: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    question = exact_generation_question(item)
    request: Dict[str, Any] = {
        "model": args.gpt_model,
        "input": [{
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": policy_image_data_url(item["image_path"]),
                    "detail": args.gpt_image_detail,
                },
                {"type": "input_text", "text": question},
            ],
        }],
        "max_output_tokens": int(args.gpt_max_output_tokens),
        "store": False,
    }
    if not args.gpt_omit_sampling_params:
        request["temperature"] = float(args.gpt_temperature)
        request["top_p"] = float(args.gpt_top_p)
    if args.gpt_reasoning_effort:
        request["reasoning"] = {"effort": args.gpt_reasoning_effort}

    response = call_openai_with_retries(
        kwargs=request,
        timeout_seconds=int(args.gpt_timeout),
        max_retries=int(args.gpt_max_retries),
        base_sleep=float(args.gpt_retry_base_sleep),
    )
    answer = extract_openai_response_text(response)
    if not answer:
        raise RuntimeError(f"OpenAI returned empty output at idx={idx}")

    return {
        "idx": int(idx),
        "domain": item["domain"],
        "image_path": item["image_path"],
        "question_sha256": stable_sha256_text(question),
        "generator_type": "gpt",
        "generator_id": args.gpt_model,
        "Generated_Answer": answer,
        "valid_structured_json": parse_idea_json(answer) is not None,
    }


def generate_gpt_all(
    test_data: List[Dict[str, Any]],
    args: argparse.Namespace,
    generation_file: str,
) -> List[Dict[str, Any]]:
    records: List[Optional[Dict[str, Any]]] = [None] * len(test_data)
    max_workers = max(1, int(args.gpt_max_concurrency))

    print(
        f"[GPT generation] model={args.gpt_model} samples={len(test_data)} "
        f"concurrency={max_workers}",
        flush=True,
    )
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="mba-openai-generation",
    ) as executor:
        futures = {
            executor.submit(generate_gpt_one, idx, item, args): idx
            for idx, item in enumerate(test_data)
        }
        for completed, future in enumerate(
            tqdm(
                as_completed(futures),
                total=len(futures),
                desc="[GPT generation]",
                dynamic_ncols=True,
            ),
            start=1,
        ):
            idx = futures[future]
            records[idx] = future.result()
            if completed % int(args.save_every) == 0:
                atomic_json_dump(
                    [item for item in records if item is not None],
                    generation_file,
                    indent=None,
                )

    if any(item is None for item in records):
        missing = [idx for idx, item in enumerate(records) if item is None]
        raise RuntimeError(f"Missing GPT generations: {missing[:20]}")
    final_records = [item for item in records if item is not None]
    atomic_json_dump(final_records, generation_file, indent=2)
    return final_records


def balanced_chunks(
    items: Sequence[Any],
    worker_count: int,
    cost_fn,
) -> List[List[int]]:
    worker_count = max(1, min(int(worker_count), len(items)))
    bins: List[List[int]] = [[] for _ in range(worker_count)]
    loads = [0.0] * worker_count
    ordered = sorted(
        range(len(items)),
        key=lambda idx: (-float(cost_fn(items[idx])), idx),
    )
    for idx in ordered:
        target = min(range(worker_count), key=lambda worker: (loads[worker], worker))
        bins[target].append(idx)
        loads[target] += float(cost_fn(items[idx]))
    for chunk in bins:
        chunk.sort()
    return [chunk for chunk in bins if chunk]


def local_generation_cost(item: Dict[str, Any]) -> float:
    with Image.open(item["image_path"]) as image:
        width, height = image.size
    scale = min(1.0, float(grpo.POLICY_IMAGE_SIZE) / max(width, height, 1))
    area = width * height * scale * scale
    return 1.0 + area / float(28 * 28) + len(item["question"]) / 128.0


def generate_local_all(
    test_data: List[Dict[str, Any]],
    args: argparse.Namespace,
    generation_file: str,
    tmp_dir: str,
) -> List[Dict[str, Any]]:
    import torch.multiprocessing as mp

    gpu_count = torch.cuda.device_count()
    adapter_path = resolve_local_adapter_path(args)
    chunks = balanced_chunks(test_data, gpu_count, local_generation_cost)
    args_dict = vars(args).copy()
    processes = []
    part_paths = []

    for worker_rank, chunk in enumerate(chunks):
        shard = [(idx, test_data[idx]) for idx in chunk]
        part_path = os.path.join(tmp_dir, f"generation_part_{worker_rank}.json")
        process = mp.Process(
            target=local_generation_worker,
            args=(
                worker_rank,
                shard,
                args.model_type,
                adapter_path,
                args_dict,
                part_path,
            ),
        )
        process.start()
        processes.append(process)
        part_paths.append(part_path)

    for process in processes:
        process.join()
    failed = [
        (idx, process.exitcode)
        for idx, process in enumerate(processes)
        if process.exitcode != 0
    ]
    if failed:
        raise RuntimeError(f"Local generation worker failure(s): {failed}")

    records: Dict[int, Dict[str, Any]] = {}
    for part_path in part_paths:
        if not os.path.exists(part_path):
            raise RuntimeError(f"Missing generation part: {part_path}")
        for record in read_json(part_path):
            idx = int(record["idx"])
            if idx in records:
                raise RuntimeError(f"Duplicate generated idx={idx}")
            records[idx] = record

    if set(records) != set(range(len(test_data))):
        missing = sorted(set(range(len(test_data))) - set(records))
        raise RuntimeError(f"Local generation missing indices: {missing[:20]}")
    payload = [records[idx] for idx in range(len(test_data))]
    atomic_json_dump(payload, generation_file, indent=2)
    return payload


def validate_generation_records(
    records: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    if len(records) != len(test_data):
        raise RuntimeError(
            f"Generation count mismatch: {len(records)} vs {len(test_data)}"
        )
    seen = set()
    for record in records:
        idx = int(record["idx"])
        if idx in seen or not (0 <= idx < len(test_data)):
            raise RuntimeError(f"Invalid or duplicate generation idx={idx}")
        seen.add(idx)
        item = test_data[idx]
        if record.get("domain") != item["domain"]:
            raise RuntimeError(f"Generation domain mismatch at idx={idx}")
        expected_hash = stable_sha256_text(exact_generation_question(item))
        if record.get("question_sha256") != expected_hash:
            raise RuntimeError(
                f"Generation prompt hash mismatch at idx={idx}; "
                "do not reuse generations from another manifest/prompt revision."
            )
        if record.get("generator_type") != args.model_type:
            raise RuntimeError(
                f"Generation model_type mismatch at idx={idx}: "
                f"file={record.get('generator_type')!r}, requested={args.model_type!r}"
            )
        expected_generator_id = (
            args.gpt_model if args.model_type == "gpt"
            else (resolve_local_adapter_path(args) or BASE_MODEL_ID)
        )
        if record.get("generator_id") != expected_generator_id:
            raise RuntimeError(
                f"Generation model identifier mismatch at idx={idx}: "
                f"file={record.get('generator_id')!r}, expected={expected_generator_id!r}"
            )
        if not clean_text(record.get("Generated_Answer")):
            raise RuntimeError(f"Empty generation at idx={idx}")


# ---------------------------------------------------------------------------
# Train-aligned absolute Judge prompt and parsing
# ---------------------------------------------------------------------------
def format_absolute_axis_rubric(dimension: str) -> str:
    meta = PBIG_DIMENSIONS[dimension]
    levels = "\n".join(
        f"  {score}: {description}"
        for score, description in meta["levels"].items()
    )
    return (
        f"- {dimension} ({meta['display_name']}), integer "
        f"{meta['scale_min']}-{meta['scale_max']}\n"
        f"  Training-aligned focus: {grpo.SIX_DIMENSION_RUBRICS[dimension]}\n"
        f"  Absolute score levels:\n{levels}"
    )


def build_absolute_axis_judge_text(
    topic: str,
    generation: str,
    dimension: str,
) -> str:
    """Build one isolated InternVL prompt for exactly one PBIG dimension."""
    if dimension not in PBIG_DIMENSION_ORDER:
        raise ValueError(f"Unknown PBIG dimension: {dimension!r}")

    evaluation_context = grpo.build_judge_evaluation_context(topic)
    meta = PBIG_DIMENSIONS[dimension]
    axis_rubric = format_absolute_axis_rubric(dimension)

    return f"""You are an independent InternVL multimodal business-idea Judge.
The GRPO reward model was Qwen2-VL-72B. Eval is identical for open and closed
training branches: one Judge inference considers exactly one reference-free PBIG
dimension and assigns one absolute score to one candidate.

Your only task is to EVALUATE the supplied candidate. Do not generate, rewrite,
complete, compare with an unstated reference, or improve the business idea.

Evaluate exactly ONE target dimension: {dimension} ({meta['display_name']}).
Do not score, discuss, infer, or combine any other PBIG dimension.

Target-dimension rubric:
{axis_rubric}

Validity and grounding rules for this target dimension:
- A valid idea should be JSON with exactly four non-empty string keys:
  title, description, implementation, differentiation.
- Penalize invalid JSON, missing keys, extra prose outside JSON, or an
  unstructured answer only insofar as it affects the target dimension.
- Use the visible image and only the neutral evaluation context below, including the same sample-matched auxiliary image caption seen by the generator.
- Treat the visible image as primary evidence; use the generated caption only as auxiliary semantic context and do not let it override the image.
- Penalize ideas that ignore or contradict the image/context, hallucinate
  unsupported visual details, or are not business ideas.
- No GT, reference, expert idea, or training-only creativity/feasibility anchor is provided. Never infer one.
- Score using only the integer scale {meta['scale_min']}-{meta['scale_max']}.
- Confidence must be an integer 0-100.
- The reason must be one concise evidence-based sentence about {dimension} only.
- The later aggregation weight is {TRAIN_ALIGNED_DIMENSION_WEIGHTS[dimension]:.2f};
  do not let that weight change this axis score.

[Evaluation Context]
{evaluation_context}

[Candidate Business Idea]
{grpo.format_idea_for_judge(generation)}

Return ONLY one valid JSON object with exactly these three keys and no markdown
or extra text:
{{
  "score": <integer {meta['scale_min']}-{meta['scale_max']}>,
  "reason": "<one concise evidence-based sentence about {dimension}>",
  "confidence": <integer 0-100>
}}
"""


def parse_absolute_axis_json(
    raw: str,
    dimension: str,
) -> Dict[str, Any]:
    """Parse one Judge response that contains exactly one dimension score."""
    if dimension not in PBIG_DIMENSION_ORDER:
        raise ValueError(f"Unknown PBIG dimension: {dimension!r}")

    raw_text = str(raw or "").strip()
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        raise RuntimeError(
            f"Judge returned no JSON object for {dimension}: {raw_text!r}"
        )
    try:
        obj = json.loads(match.group(0))
    except Exception as exc:
        raise RuntimeError(
            f"Judge returned invalid JSON for {dimension}: {raw_text!r}"
        ) from exc

    expected_keys = {"score", "reason", "confidence"}
    if not isinstance(obj, dict) or set(obj) != expected_keys:
        raise RuntimeError(
            f"Judge keys for {dimension} must be exactly "
            f"{sorted(expected_keys)}; found="
            f"{sorted(obj) if isinstance(obj, dict) else type(obj)}"
        )

    meta = PBIG_DIMENSIONS[dimension]
    lo, hi = int(meta["scale_min"]), int(meta["scale_max"])

    score_float = float(obj["score"])
    score = int(round(score_float))
    if abs(score_float - score) > 1e-6 or not (lo <= score <= hi):
        raise RuntimeError(
            f"{dimension} score must be integer {lo}-{hi}: {obj['score']!r}"
        )

    confidence_float = float(obj["confidence"])
    confidence = int(round(confidence_float))
    if (
        abs(confidence_float - confidence) > 1e-6
        or not (0 <= confidence <= 100)
    ):
        raise RuntimeError(
            f"{dimension} confidence must be integer 0-100: "
            f"{obj['confidence']!r}"
        )

    reason = clean_text(obj["reason"])
    if not reason:
        raise RuntimeError(f"{dimension} reason is empty.")

    return {
        "score": score,
        "normalized_0_100": normalize_score_0_100(dimension, score),
        "reason": reason,
        "confidence": confidence,
    }


def majority_vote_axis(
    runs: List[Dict[str, Any]],
    dimension: str,
) -> Dict[str, Any]:
    """Majority vote repeated independent runs for one dimension only."""
    if not runs:
        raise RuntimeError(f"No valid Judge runs for {dimension}.")

    counts = Counter(item["score"] for item in runs)
    avg_confidence = {
        score: statistics.mean(
            item["confidence"] for item in runs if item["score"] == score
        )
        for score in counts
    }
    selected_score = sorted(
        counts,
        key=lambda score: (
            -counts[score],
            -avg_confidence[score],
            score,
        ),
    )[0]
    candidates = [
        item for item in runs if item["score"] == selected_score
    ]
    representative = sorted(
        candidates,
        key=lambda item: (
            -item["confidence"],
            len(item["reason"]),
            item["reason"],
        ),
    )[0]
    return {
        "score": int(selected_score),
        "normalized_0_100": normalize_score_0_100(
            dimension, int(selected_score)
        ),
        "reason": representative["reason"],
        "confidence": int(round(avg_confidence[selected_score])),
        "vote_count": int(counts[selected_score]),
        "num_runs": len(runs),
    }


# ---------------------------------------------------------------------------
# Independent InternVL2.5-78B absolute evaluator
# ---------------------------------------------------------------------------
class InternVLAbsolutePBig6Evaluator:
    def __init__(
        self,
        judge_gpus: Sequence[int],
        precision: str,
        repeats: int,
        do_sample: bool,
        temperature: float,
        max_new_tokens: int,
        parse_retries: int,
    ):
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

        self.judge_gpus = tuple(int(gpu) for gpu in judge_gpus)
        if len(self.judge_gpus) != 4:
            raise ValueError(
                "Each InternVL2.5-78B evaluator process requires exactly "
                f"four GPUs; received={self.judge_gpus}"
            )
        self.input_device = torch.device(f"cuda:{self.judge_gpus[0]}")
        self.repeats = max(1, int(repeats))
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.parse_retries = max(0, int(parse_retries))

        requested = str(precision).lower()
        if requested not in {"4bit", "8bit", "bf16"}:
            raise ValueError(
                "judge_precision must be one of: 4bit, 8bit, bf16"
            )
        self.precision = requested
        assert_internvl_transformers_compatibility()

        quantization_config = None
        reserve_gib = 4
        if requested == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif requested == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            reserve_gib = 3
        else:
            reserve_gib = 6

        max_memory = internvl_max_memory(
            self.judge_gpus, reserve_gib=reserve_gib
        )
        print(
            f"[InternVL Judge {self.judge_gpus}] loading "
            f"{EVAL_JUDGE_MODEL_ID} precision={requested} "
            f"max_memory={max_memory}",
            flush=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            EVAL_JUDGE_MODEL_ID,
            trust_remote_code=True,
            use_fast=False,
        )
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "use_flash_attn": True,
            "device_map": split_internvl_78b(self.judge_gpus),
            "max_memory": max_memory,
            "offload_state_dict": True,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        self.model = AutoModel.from_pretrained(
            EVAL_JUDGE_MODEL_ID,
            **load_kwargs,
        ).eval()
        rotary_device = pin_internvl_runtime_buffers(
            self.model, self.input_device
        )
        used_devices = sorted({
            str(value)
            for value in getattr(self.model, "hf_device_map", {}).values()
        })
        print(
            f"[InternVL Judge {self.judge_gpus}] ready | "
            f"input={self.input_device} rotary={rotary_device} "
            f"targets={used_devices} transformers={transformers.__version__}",
            flush=True,
        )

    def _run_one(
        self,
        pixel_values: torch.Tensor,
        prompt: str,
        image_path: str,
        run_index: int,
    ) -> str:
        stable_key = (
            str(image_path)
            + "\n"
            + str(prompt)
            + "\n"
            + str(run_index)
        )
        seed = int(
            hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:8],
            16,
        )
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        generation_config: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "num_beams": 1,
        }
        if self.do_sample:
            generation_config.update({
                "temperature": self.temperature,
                "top_p": 0.9,
            })

        with torch.inference_mode():
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question="<image>\n" + prompt,
                generation_config=generation_config,
                history=None,
                return_history=False,
            )
        if isinstance(response, tuple):
            response = response[0]
        return str(response).strip()

    def score_records(
        self,
        records: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        save_path: str,
        save_every: int,
        worker_position: int,
    ) -> List[Dict[str, Any]]:
        scored: List[Dict[str, Any]] = []
        pbar = tqdm(
            records,
            desc=f"[InternVL Judge worker {worker_position}]",
            position=worker_position,
            leave=True,
            dynamic_ncols=True,
        )

        for record in pbar:
            idx = int(record["idx"])
            item = test_data[idx]
            image_path = item["image_path"]
            topic = exact_generation_question(item)
            generation = record["Generated_Answer"]

            pixel_values = _cached_internvl_pixels(image_path).clone().to(
                device=self.input_device,
                dtype=torch.bfloat16,
                non_blocking=False,
            )

            dimensions: Dict[str, Dict[str, Any]] = OrderedDict()
            raw_runs_by_dimension: Dict[str, List[str]] = OrderedDict()
            try:
                for dimension_index, dimension in enumerate(
                    PBIG_DIMENSION_ORDER
                ):
                    prompt = build_absolute_axis_judge_text(
                        topic=topic,
                        generation=generation,
                        dimension=dimension,
                    )
                    valid_runs: List[Dict[str, Any]] = []
                    raw_dimension_runs: List[str] = []

                    for run_index in range(self.repeats):
                        current_raw = self._run_one(
                            pixel_values=pixel_values,
                            prompt=prompt,
                            image_path=image_path,
                            run_index=(dimension_index * 100000 + run_index),
                        )
                        parsed = None
                        last_error: Optional[Exception] = None
                        for retry_index in range(self.parse_retries + 1):
                            try:
                                parsed = parse_absolute_axis_json(
                                    current_raw, dimension
                                )
                                break
                            except Exception as exc:
                                last_error = exc
                                if retry_index >= self.parse_retries:
                                    break
                                repair_prompt = prompt + f"""

Your previous response was invalid: {type(exc).__name__}: {exc}
Return ONLY the exact three-key JSON object for {dimension}:
{{"score": <integer>, "reason": "<sentence>", "confidence": <integer>}}
Do not add markdown, commentary, prefixes, suffixes, or any other metric.
"""
                                current_raw = self._run_one(
                                    pixel_values=pixel_values,
                                    prompt=repair_prompt,
                                    image_path=image_path,
                                    run_index=(
                                        dimension_index * 100000
                                        + run_index
                                        + 1000
                                        + retry_index
                                    ),
                                )
                        if parsed is None:
                            raise RuntimeError(
                                f"InternVL Judge parse failed for idx={idx}, "
                                f"dimension={dimension}: {last_error}; "
                                f"raw={current_raw!r}"
                            )
                        valid_runs.append(parsed)
                        raw_dimension_runs.append(current_raw)

                    dimensions[dimension] = majority_vote_axis(
                        valid_runs, dimension
                    )
                    raw_runs_by_dimension[dimension] = raw_dimension_runs
                    pbar.set_postfix({
                        "axis": f"{dimension_index + 1}/6",
                        "metric": dimension,
                    })
            finally:
                del pixel_values

            normalized = {
                dimension: dimensions[dimension]["normalized_0_100"]
                for dimension in PBIG_DIMENSION_ORDER
            }
            raw_scores = {
                dimension: dimensions[dimension]["score"]
                for dimension in PBIG_DIMENSION_ORDER
            }
            unweighted = statistics.mean(normalized.values())
            weighted = sum(
                TRAIN_ALIGNED_DIMENSION_WEIGHTS[dimension]
                * normalized[dimension]
                for dimension in PBIG_DIMENSION_ORDER
            )
            scored.append({
                "idx": idx,
                "domain": item["domain"],
                "image_path": image_path,
                "question": topic,
                "question_sha256": record["question_sha256"],
                "Generated_Answer": generation,
                "valid_structured_json": bool(
                    record["valid_structured_json"]
                ),
                "Scores": {
                    "pbig_dimensions": dimensions,
                    "pbig_scores_raw": raw_scores,
                    "pbig_scores_normalized_0_100": normalized,
                    "pbig_mean_0_100": unweighted,
                    "pbig_train_weighted_0_100": weighted,
                },
                "judge_model": EVAL_JUDGE_MODEL_ID,
                "judge_protocol": (
                    "six independent single-axis absolute-scoring calls"
                ),
                "judge_axis_order": list(PBIG_DIMENSION_ORDER),
                "judge_axis_call_count": (
                    len(PBIG_DIMENSION_ORDER) * self.repeats
                ),
                "judge_raw_runs_by_dimension": raw_runs_by_dimension,
            })
            pbar.set_postfix({
                "mean6": f"{unweighted:.2f}",
                "weighted6": f"{weighted:.2f}",
                "axis_calls": len(PBIG_DIMENSION_ORDER) * self.repeats,
            })
            if len(scored) % max(1, int(save_every)) == 0:
                atomic_json_dump(scored, save_path, indent=None)

        atomic_json_dump(scored, save_path, indent=2)
        return scored


def score_worker(
    worker_rank: int,
    judge_gpus: Tuple[int, ...],
    records: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    args_dict: Dict[str, Any],
    output_path: str,
) -> None:
    args = argparse.Namespace(**args_dict)
    torch.cuda.set_device(int(judge_gpus[0]))
    evaluator = InternVLAbsolutePBig6Evaluator(
        judge_gpus=judge_gpus,
        precision=args.judge_precision,
        repeats=args.judge_repeats,
        do_sample=args.judge_do_sample,
        temperature=args.judge_temperature,
        max_new_tokens=args.judge_max_new_tokens,
        parse_retries=args.judge_parse_retries,
    )
    evaluator.score_records(
        records=records,
        test_data=test_data,
        save_path=output_path,
        save_every=args.save_every,
        worker_position=worker_rank,
    )


# ---------------------------------------------------------------------------
# Summary and exactly seven human-readable result tables
# ---------------------------------------------------------------------------
def finite_values(values: Iterable[Optional[float]]) -> List[float]:
    output: List[float] = []
    for value in values:
        if value is None:
            continue
        converted = float(value)
        if math.isfinite(converted):
            output.append(converted)
    return output


def std_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = finite_values(values)
    return statistics.pstdev(clean) if clean else None


def summarize_subset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    dimension_summary: Dict[str, Any] = OrderedDict()
    for dimension in PBIG_DIMENSION_ORDER:
        raw_values = [
            record["Scores"]["pbig_scores_raw"][dimension]
            for record in records
        ]
        normalized_values = [
            record["Scores"]["pbig_scores_normalized_0_100"][dimension]
            for record in records
        ]
        dimension_summary[dimension] = {
            "display_name": PBIG_DIMENSIONS[dimension]["display_name"],
            "scale": [
                PBIG_DIMENSIONS[dimension]["scale_min"],
                PBIG_DIMENSIONS[dimension]["scale_max"],
            ],
            "mean_raw": mean_or_none(raw_values),
            "std_raw": std_or_none(raw_values),
            "mean_normalized_0_100": mean_or_none(normalized_values),
            "std_normalized_0_100": std_or_none(normalized_values),
        }

    return {
        "num_samples": len(records),
        "num_unique_images": len({
            grpo.canonical_image_key(record["image_path"])
            for record in records
        }),
        "valid_structured_json_rate": (
            sum(bool(record["valid_structured_json"]) for record in records)
            / len(records)
            if records else 0.0
        ),
        "mean_unweighted_pbig6_0_100": mean_or_none(
            record["Scores"]["pbig_mean_0_100"] for record in records
        ),
        "std_unweighted_pbig6_0_100": std_or_none(
            record["Scores"]["pbig_mean_0_100"] for record in records
        ),
        "mean_train_weighted_pbig6_0_100": mean_or_none(
            record["Scores"]["pbig_train_weighted_0_100"]
            for record in records
        ),
        "std_train_weighted_pbig6_0_100": std_or_none(
            record["Scores"]["pbig_train_weighted_0_100"]
            for record in records
        ),
        "dimension_summary": dimension_summary,
    }


def build_summary(
    args: argparse.Namespace,
    records: List[Dict[str, Any]],
    generation_file: str,
) -> Dict[str, Any]:
    by_domain = OrderedDict()
    for domain in EXPECTED_DOMAINS:
        subset = [record for record in records if record["domain"] == domain]
        by_domain[domain] = summarize_subset(subset)

    manifest_payload = [
        {
            "domain": record["domain"],
            "image_path": record["image_path"],
            "question_sha256": record["question_sha256"],
        }
        for record in sorted(records, key=lambda item: int(item["idx"]))
    ]

    return {
        "pipeline_revision": PIPELINE_REVISION,
        "training_module": GRPO_TRAIN_PATH,
        "training_pipeline_revision": getattr(grpo, "PIPELINE_REVISION", None),
        "prompt_revision": EXPECTED_INCONTEXT_PROMPT_REVISION,
        "split_revision": EXPECTED_SPLIT_REVISION,
        "test_manifest": TEST_FILE,
        "test_manifest_fingerprint": stable_sha256_text(
            json.dumps(manifest_payload, sort_keys=True, ensure_ascii=False)
        ),
        "model_type": args.model_type,
        "generator_name": (
            args.gpt_model
            if args.model_type == "gpt"
            else (resolve_local_adapter_path(args) or BASE_MODEL_ID)
        ),
        "generator_config": generation_config_payload(args),
        "generation_file": generation_file,
        "training_reward_judge_model": TRAINING_JUDGE_MODEL_ID,
        "judge_model": EVAL_JUDGE_MODEL_ID,
        "judge_same_model_as_grpo_training": False,
        "judge_family_independent": True,
        "judge_context": (
            "image + Domain Context + Image-Specific Annotation + Business Lens "
            "+ DDG Query + DDG Evidence; no Task/output instructions/GT"
        ),
        "judge_protocol": (
            "six independent InternVL single-axis absolute-scoring calls per candidate; "
            "GRPO uses six independent Qwen single-axis four-candidate rankings"
        ),
        "judge_axis_call_count_per_candidate": (
            len(PBIG_DIMENSION_ORDER) * int(args.judge_repeats)
        ),
        "judge_axis_batch_size": int(args.judge_axis_batch_size),
        "judge_precision": args.judge_precision,
        "judge_repeats": args.judge_repeats,
        "judge_do_sample": bool(args.judge_do_sample),
        "judge_temperature": (
            args.judge_temperature if args.judge_do_sample else None
        ),
        "axis_weights": dict(TRAIN_ALIGNED_DIMENSION_WEIGHTS),
        "overall": summarize_subset(records),
        "per_domain": by_domain,
    }


def _summary_number(value: Optional[float], digits: int = 4) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _ascii_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    right_align: Optional[Iterable[int]] = None,
) -> str:
    right = set(right_align or [])
    clean_rows = [
        [str(cell).replace("\n", " ") for cell in row]
        for row in rows
    ]
    widths = [
        max(
            len(str(headers[col])),
            max((len(row[col]) for row in clean_rows), default=0),
        )
        for col in range(len(headers))
    ]

    def render_row(row: Sequence[str]) -> str:
        cells = []
        for col, value in enumerate(row):
            cells.append(
                value.rjust(widths[col]) if col in right
                else value.ljust(widths[col])
            )
        return "| " + " | ".join(cells) + " |"

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    output = [
        separator,
        render_row([str(header) for header in headers]),
        separator,
    ]
    output.extend(render_row(row) for row in clean_rows)
    output.append(separator)
    return "\n".join(output)


def _model_report_slug(args: argparse.Namespace) -> str:
    return (
        safe_slug(args.gpt_model)
        if args.model_type == "gpt"
        else safe_slug(args.model_type)
    )


def build_seven_table_report(
    args: argparse.Namespace,
    summary: Dict[str, Any],
    records: List[Dict[str, Any]],
    output_file: str,
    summary_file: str,
    generation_file: str,
) -> str:
    overall = summary["overall"]
    lines: List[str] = [
        "=" * 118,
        "MBA INDEPENDENT INTERNVL PBIG-6 EVALUATION REPORT",
        "=" * 118,
        f"Generated UTC          : {datetime.now(timezone.utc).isoformat()}",
        f"Evaluation model type  : {args.model_type}",
        f"Generator model/id     : {summary['generator_name']}",
        f"Generator prompt       : exact canonical held-out question used by SFT/GRPO",
        f"Training reward Judge  : {summary['training_reward_judge_model']}",
        f"Evaluation Judge       : {summary['judge_model']}",
        "Judge relation         : different model family for independent evaluation",
        "Evaluation protocol    : six independent reference-free PBIG axis calls",
        "Training rewards       : branch-dependent and never reused by the Eval Judge",
        "Eval Judge unit        : one PBIG metric + one candidate -> one absolute score",
        "Judge context          : image + domain + annotation + image caption + business lens + DDG query/evidence",
        "Judge excluded fields  : Task, generation/output instructions, Grounding Requirements, GT/reference/creativity/market anchors",
        f"Split                   : {summary['split_revision']} / held-out test only",
        f"Test manifest           : {summary['test_manifest']}",
        f"Manifest fingerprint    : {summary['test_manifest_fingerprint']}",
        f"Samples / images        : {overall['num_samples']} / {overall['num_unique_images']}",
        f"Valid structured JSON   : {overall['valid_structured_json_rate']:.2%}",
        f"PBIG-6 mean (unweighted): {_summary_number(overall['mean_unweighted_pbig6_0_100'])}",
        f"PBIG-6 mean (weighted)  : {_summary_number(overall['mean_train_weighted_pbig6_0_100'])}",
        f"Generation JSON         : {generation_file}",
        f"Detailed result JSON    : {output_file}",
        f"Summary JSON            : {summary_file}",
        "",
    ]

    # Table 1: all six metrics over the complete test set.
    overall_rows: List[List[str]] = []
    for dimension in PBIG_DIMENSION_ORDER:
        item = overall["dimension_summary"][dimension]
        lo, hi = item["scale"]
        overall_rows.append([
            item["display_name"],
            f"{lo}-{hi}",
            str(overall["num_samples"]),
            _summary_number(item["mean_raw"]),
            _summary_number(item["std_raw"]),
            _summary_number(item["mean_normalized_0_100"]),
            _summary_number(item["std_normalized_0_100"]),
            f"{TRAIN_ALIGNED_DIMENSION_WEIGHTS[dimension]:.2f}",
        ])
    lines.extend([
        "[TABLE 1/7] Overall test-set performance across all six PBIG metrics",
        _ascii_table(
            headers=[
                "Metric", "Scale", "N", "Mean Raw", "Std Raw",
                "Mean 0-100", "Std 0-100", "Weight",
            ],
            rows=overall_rows,
            right_align={2, 3, 4, 5, 6, 7},
        ),
        "",
    ])

    # Tables 2-7: one table for each metric across all six domains.
    for table_index, dimension in enumerate(PBIG_DIMENSION_ORDER, start=2):
        meta = PBIG_DIMENSIONS[dimension]
        domain_rows: List[List[str]] = []
        for domain in EXPECTED_DOMAINS:
            subset = [
                record for record in records if record["domain"] == domain
            ]
            raw_values = [
                record["Scores"]["pbig_scores_raw"][dimension]
                for record in subset
            ]
            normalized_values = [
                record["Scores"]["pbig_scores_normalized_0_100"][dimension]
                for record in subset
            ]
            domain_rows.append([
                domain,
                str(len(subset)),
                str(len({
                    grpo.canonical_image_key(record["image_path"])
                    for record in subset
                })),
                _summary_number(mean_or_none(raw_values)),
                _summary_number(std_or_none(raw_values)),
                _summary_number(mean_or_none(normalized_values)),
                _summary_number(std_or_none(normalized_values)),
            ])

        lines.extend([
            (
                f"[TABLE {table_index}/7] {meta['display_name']} by domain "
                f"(absolute scale {meta['scale_min']}-{meta['scale_max']})"
            ),
            _ascii_table(
                headers=[
                    "Domain", "N", "Images", "Mean Raw", "Std Raw",
                    "Mean 0-100", "Std 0-100",
                ],
                rows=domain_rows,
                right_align={1, 2, 3, 4, 5, 6},
            ),
            "",
        ])

    report = "\n".join(lines).rstrip() + "\n"
    if report.count("[TABLE ") != 7:
        raise RuntimeError(
            f"Internal report error: expected exactly 7 tables, "
            f"found {report.count('[TABLE ')}"
        )
    return report


def write_and_print_report(report: str, report_file: str) -> None:
    report_path = Path(report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    os.replace(temporary, report_path)
    print("\n" + report, flush=True)
    print(f"[Report] saved={report_file}", flush=True)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def run_self_test() -> None:
    synthetic_topic = f"""[Prompt Revision]
{EXPECTED_INCONTEXT_PROMPT_REVISION}

[Task]
Generate one business idea.

[Domain Context]
domain: general_scene
domain_specific_focus: visible workflow needs

[Image-Specific Annotation]
{{"objects": ["queue", "counter"]}}

[Auxiliary Image Caption]
People wait in a line near a service counter.

[Business Lens]
lens: customer_operations_growth
lens_focus: improve throughput

[DDG Query]
Q1: queue management business market

[DDG Evidence]
{{"evidence_text": "Businesses seek lower wait times and better service throughput."}}

[Grounding Requirements]
Use the image and evidence.

{grpo.IDEA_FORMAT_INSTRUCTION}
"""
    context = grpo.build_judge_evaluation_context(synthetic_topic)
    assert "[Task]" not in context
    assert "[Grounding Requirements]" not in context
    assert "Domain Context" in context
    assert "Auxiliary Image Caption" in context
    assert "People wait in a line near a service counter." in context
    assert "DDG Evidence" in context

    idea = json.dumps({
        "title": "QueueFlow Vision",
        "description": "A B2B queue analytics service for retail operators.",
        "implementation": "Use existing cameras and computer vision to estimate waits.",
        "differentiation": "Workflow integration and site-specific operational data.",
    })
    reference_ideas = [
        json.dumps({
            "title": f"Offline Anchor {index}",
            "description": "Synthetic reference idea.",
            "implementation": "Synthetic reference implementation.",
            "differentiation": "Synthetic reference differentiation.",
        })
        for index in range(5)
    ]
    creativity_prompt = grpo.build_creativity_ranking_judge_text(
        synthetic_topic, [idea, idea], reference_ideas
    )
    assert "Offline Reference Ideas — Creativity Only" in creativity_prompt
    assert "Offline Anchor 0" in creativity_prompt
    market_reference_context = (
        grpo.MarketDBFeasibilityScorer._reference_market_context(reference_ideas)
    )
    for index in range(5):
        assert f"reference_market_anchor_{index + 1}" in market_reference_context
        assert f"Offline Anchor {index}" in market_reference_context

    axis_prompts = []
    for dimension in PBIG_DIMENSION_ORDER:
        training_pbig_prompt = grpo.build_single_dimension_ranking_judge_text(
            synthetic_topic, [idea, idea], dimension
        )
        assert "Offline Reference Ideas" not in training_pbig_prompt
        assert "Reference Idea 1" not in training_pbig_prompt
        assert "Offline Anchor 0" not in training_pbig_prompt
        assert "No GT, reference, or expert idea is provided" in training_pbig_prompt

        prompt = build_absolute_axis_judge_text(
            synthetic_topic, idea, dimension
        )
        axis_prompts.append(prompt)
        assert "[Task]" not in prompt
        assert "Reference Idea" not in prompt
        assert "Offline Anchor 0" not in prompt
        assert "training-only creativity/feasibility anchor" in prompt
        assert "Candidate Business Idea" in prompt
        assert f"Evaluate exactly ONE target dimension: {dimension}" in prompt
        for other in PBIG_DIMENSION_ORDER:
            if other != dimension:
                assert f"Evaluate exactly ONE target dimension: {other}" not in prompt

        fake = {
            "score": PBIG_DIMENSIONS[dimension]["scale_min"],
            "reason": "synthetic test reason",
            "confidence": 90,
        }
        parsed = parse_absolute_axis_json(json.dumps(fake), dimension)
        assert parsed["score"] == PBIG_DIMENSIONS[dimension]["scale_min"]

    assert len(axis_prompts) == 6
    assert len(set(axis_prompts)) == 6

    synthetic_records = []
    for idx, domain in enumerate(EXPECTED_DOMAINS):
        raw_scores = {
            dimension: int(PBIG_DIMENSIONS[dimension]["scale_min"])
            for dimension in PBIG_DIMENSION_ORDER
        }
        normalized = {
            dimension: normalize_score_0_100(
                dimension, raw_scores[dimension]
            )
            for dimension in PBIG_DIMENSION_ORDER
        }
        synthetic_records.append({
            "idx": idx,
            "domain": domain,
            "image_path": f"/synthetic/{domain}.png",
            "question_sha256": stable_sha256_text(domain),
            "Generated_Answer": idea,
            "valid_structured_json": True,
            "Scores": {
                "pbig_scores_raw": raw_scores,
                "pbig_scores_normalized_0_100": normalized,
                "pbig_mean_0_100": statistics.mean(normalized.values()),
                "pbig_train_weighted_0_100": sum(
                    TRAIN_ALIGNED_DIMENSION_WEIGHTS[dimension]
                    * normalized[dimension]
                    for dimension in PBIG_DIMENSION_ORDER
                ),
            },
        })
    synthetic_args = argparse.Namespace(
        model_type="base",
        gpt_model="gpt-4o",
        local_max_new_tokens=int(grpo.MAX_NEW_TOKENS),
        local_temperature=1.05,
        local_top_p=0.95,
        local_top_k=50,
        local_repetition_penalty=1.05,
        gpt_max_output_tokens=256,
        gpt_omit_sampling_params=False,
        gpt_temperature=1.05,
        gpt_top_p=0.95,
        gpt_reasoning_effort=None,
        gpt_image_detail="auto",
        sft_adapter=None,
        grpo_adapter=None,
        judge_precision="4bit",
        judge_repeats=1,
        judge_do_sample=False,
        judge_temperature=0.20,
    )
    synthetic_summary = {
        "generator_name": BASE_MODEL_ID,
        "training_reward_judge_model": TRAINING_JUDGE_MODEL_ID,
        "judge_model": EVAL_JUDGE_MODEL_ID,
        "split_revision": EXPECTED_SPLIT_REVISION,
        "test_manifest": TEST_FILE,
        "test_manifest_fingerprint": stable_sha256_text("synthetic"),
        "overall": summarize_subset(synthetic_records),
        "per_domain": OrderedDict(
            (
                domain,
                summarize_subset([
                    record for record in synthetic_records
                    if record["domain"] == domain
                ]),
            )
            for domain in EXPECTED_DOMAINS
        ),
    }
    report = build_seven_table_report(
        synthetic_args,
        synthetic_summary,
        synthetic_records,
        "/tmp/results.json",
        "/tmp/summary.json",
        "/tmp/generations.json",
    )
    assert report.count("[TABLE ") == 7
    print(
        "[Self-test] branch-invariant reference-free PBIG-6 prompts, separated creativity anchors, schema parsing, axis order, "
        "and exactly-seven-table report passed."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_type",
        required=False,
        default=None,
        choices=["base", "sft", "grpo", "gpt"],
    )
    parser.add_argument("--sft_adapter", default=None)
    parser.add_argument("--grpo_adapter", default=None)
    parser.add_argument(
        "--output_dir",
        default="/workspace/output",
    )
    parser.add_argument("--output_file", default=None)
    parser.add_argument(
        "--report_file",
        default=None,
        help="Human-readable debug report containing exactly seven tables.",
    )
    parser.add_argument("--generation_file", default=None)
    parser.add_argument(
        "--reuse_generation",
        action="store_true",
        help="Reuse and validate an existing generation_file.",
    )
    parser.add_argument("--generation_only", action="store_true")
    parser.add_argument("--score_only", action="store_true")
    parser.add_argument(
        "--force_regenerate",
        action="store_true",
        help=("Ignore an existing validated generation JSON and regenerate ideas. "
              "By default an existing generation JSON is automatically reused."),
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--self_test", action="store_true")

    # Exact local GRPO rollout defaults.
    parser.add_argument(
        "--local_max_new_tokens",
        type=int,
        default=int(grpo.MAX_NEW_TOKENS),
    )
    parser.add_argument("--local_temperature", type=float, default=1.05)
    parser.add_argument("--local_top_p", type=float, default=0.95)
    parser.add_argument("--local_top_k", type=int, default=50)
    parser.add_argument("--local_repetition_penalty", type=float, default=1.05)

    # Independent InternVL2.5-78B evaluation Judge.
    parser.add_argument(
        "--judge_precision",
        choices=["4bit", "8bit", "bf16"],
        default="4bit",
    )
    parser.add_argument(
        "--judge_axis_batch_size",
        type=int,
        default=1,
        help=("Must remain 1: one independent InternVL model.chat inference per "
              "metric, giving six calls per candidate when judge_repeats=1."),
    )
    parser.add_argument(
        "--judge_batch_size",
        type=int,
        default=1,
        help="Deprecated compatibility alias; must remain 1.",
    )
    parser.add_argument("--judge_repeats", type=int, default=1)
    parser.add_argument("--judge_do_sample", action="store_true")
    parser.add_argument("--judge_temperature", type=float, default=0.20)
    parser.add_argument("--judge_max_new_tokens", type=int, default=256)
    parser.add_argument("--judge_parse_retries", type=int, default=1)

    # OpenAI generator options.
    parser.add_argument("--gpt_model", default="gpt-4o")
    parser.add_argument("--gpt_max_output_tokens", type=int, default=256)
    parser.add_argument("--gpt_temperature", type=float, default=1.05)
    parser.add_argument("--gpt_top_p", type=float, default=0.95)
    parser.add_argument("--gpt_max_concurrency", type=int, default=4)
    parser.add_argument("--gpt_timeout", type=int, default=180)
    parser.add_argument("--gpt_max_retries", type=int, default=5)
    parser.add_argument("--gpt_retry_base_sleep", type=float, default=1.0)
    parser.add_argument(
        "--gpt_image_detail",
        choices=["low", "high", "auto"],
        default="auto",
    )
    parser.add_argument("--gpt_omit_sampling_params", action="store_true")
    parser.add_argument(
        "--gpt_reasoning_effort",
        choices=["minimal", "low", "medium", "high"],
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.model_type is None:
        raise RuntimeError("--model_type is required unless --self_test is used.")
    if args.generation_only and args.score_only:
        raise RuntimeError("--generation_only and --score_only cannot be combined.")
    if args.score_only and args.force_regenerate:
        raise RuntimeError("--score_only and --force_regenerate cannot be combined.")
    if args.reuse_generation and args.force_regenerate:
        raise RuntimeError("--reuse_generation and --force_regenerate cannot be combined.")
    if int(args.judge_axis_batch_size) != 1 or int(args.judge_batch_size) != 1:
        raise RuntimeError(
            "Eval axis isolation requires --judge_axis_batch_size 1 and "
            "--judge_batch_size 1: one model.chat call per metric."
        )

    args.gpt_model = normalize_gpt_model_name(args.gpt_model)

    gpu_count = torch.cuda.device_count()
    if gpu_count not in (4, 8):
        raise RuntimeError(
            f"Expose exactly 4 or 8 GPUs for evaluation; found {gpu_count}."
        )

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    test_data = load_verified_test_manifest(max_samples=args.max_samples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator_slug = (
        safe_slug(args.gpt_model) if args.model_type == "gpt"
        else args.model_type
    )
    generation_file = args.generation_file or str(
        output_dir / f"eval_{generator_slug}_generations.json"
    )
    output_file = args.output_file or str(
        output_dir / f"eval_{generator_slug}_results.json"
    )
    summary_file = str(
        output_dir / f"eval_{generator_slug}_summary.json"
    )
    report_file = args.report_file or str(
        output_dir / f"eval_{generator_slug}.txt"
    )

    tmp_dir = output_dir / f".tmp_{generator_slug}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    generation_cache_exists = os.path.isfile(generation_file)
    reuse_generation_cache = (
        args.score_only
        or args.reuse_generation
        or (generation_cache_exists and not args.force_regenerate)
    )

    if reuse_generation_cache:
        if not generation_cache_exists:
            raise RuntimeError(
                f"Requested generation reuse but file does not exist: "
                f"{generation_file}"
            )
        generated_records = read_json(generation_file)
        try:
            validate_generation_records(generated_records, test_data, args)
        except Exception as exc:
            raise RuntimeError(
                f"Existing generation cache failed validation: {generation_file}. "
                "The cache was not used with a different manifest, prompt, or model. "
                "Delete it or pass --force_regenerate to create a fresh cache. "
                f"Cause: {exc}"
            ) from exc
        print(
            f"[Generation cache] HIT -> reused={generation_file}; "
            "idea generation skipped",
            flush=True,
        )
    else:
        if args.force_regenerate and generation_cache_exists:
            print(
                f"[Generation cache] BYPASS -> regenerating={generation_file}",
                flush=True,
            )
        if args.model_type == "gpt":
            generated_records = generate_gpt_all(
                test_data, args, generation_file
            )
        else:
            generated_records = generate_local_all(
                test_data, args, generation_file, str(tmp_dir)
            )
        validate_generation_records(generated_records, test_data, args)
        print(f"[Generation] saved={generation_file}", flush=True)

    if args.generation_only:
        print(f"Generation-only complete -> {generation_file}")
        return

    # One independent InternVL2.5-78B Judge per four-GPU group.
    # With eight GPUs this creates two scoring workers and uses all GPUs.
    judge_groups = [
        tuple(range(base, base + 4)) for base in range(0, gpu_count, 4)
    ]
    score_chunks = balanced_chunks(
        generated_records,
        len(judge_groups),
        lambda record: (
            1.0
            + len(record["Generated_Answer"]) / 128.0
            + len(test_data[int(record["idx"])]["question"]) / 256.0
        ),
    )

    processes = []
    part_paths = []
    args_dict = vars(args).copy()
    for worker_rank, (judge_gpus, chunk) in enumerate(
        zip(judge_groups, score_chunks)
    ):
        shard = [generated_records[index] for index in chunk]
        part_path = str(tmp_dir / f"score_part_{worker_rank}.json")
        process = mp.Process(
            target=score_worker,
            args=(
                worker_rank,
                judge_gpus,
                shard,
                test_data,
                args_dict,
                part_path,
            ),
        )
        process.start()
        processes.append(process)
        part_paths.append(part_path)

    for process in processes:
        process.join()
    failed = [
        (idx, process.exitcode)
        for idx, process in enumerate(processes)
        if process.exitcode != 0
    ]
    if failed:
        raise RuntimeError(f"Judge worker failure(s): {failed}")

    scored_by_idx: Dict[int, Dict[str, Any]] = {}
    for part_path in part_paths:
        if not os.path.exists(part_path):
            raise RuntimeError(f"Missing score part: {part_path}")
        for record in read_json(part_path):
            idx = int(record["idx"])
            if idx in scored_by_idx:
                raise RuntimeError(f"Duplicate scored idx={idx}")
            scored_by_idx[idx] = record

    if set(scored_by_idx) != set(range(len(test_data))):
        missing = sorted(set(range(len(test_data))) - set(scored_by_idx))
        raise RuntimeError(f"Evaluation missing indices: {missing[:20]}")

    results = [scored_by_idx[idx] for idx in range(len(test_data))]
    atomic_json_dump(results, output_file, indent=2)
    summary = build_summary(args, results, generation_file)
    atomic_json_dump(summary, summary_file, indent=2)
    report = build_seven_table_report(
        args=args,
        summary=summary,
        records=results,
        output_file=output_file,
        summary_file=summary_file,
        generation_file=generation_file,
    )
    write_and_print_report(report, report_file)
    print(f"[Results] {output_file}", flush=True)
    print(f"[Summary] {summary_file}", flush=True)


if __name__ == "__main__":
    main()
