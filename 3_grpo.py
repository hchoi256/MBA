#!/usr/bin/env python3
"""GRPO trainer with branch-selectable reward objectives.

Launch with plain Python. The script auto-detects 4 or 8 visible GPUs:
  4 GPUs -> one pipeline: Qwen2-VL-72B Judge(2) + Qwen2-VL-7B Policy(1) + frozen SFT KL Reference(1)
  8 GPUs -> two independent 4-GPU pipelines with DDP-synchronized policies

Reward branches
---------------
open:
  Six reference-free PBIG ranking rewards + one reference-conditioned creativity
  ranking reward + one feasibility reward.

  Default Total Reward = 0.80 * weighted PBIG-6
                       + 0.10 * reference-conditioned creativity
                       + 0.10 * feasibility

  Pass --no_auxiliary only with --branch_type open to disable feasibility:
    Total Reward = 0.90 * weighted PBIG-6 + 0.10 * creativity

closed:
  Creativity and feasibility only. PBIG-6 Judge calls are skipped during GRPO.

  Total Reward = 0.70 * reference-conditioned creativity
               + 0.30 * feasibility

Creativity compares each candidate with the five offline GT/reference ideas from
that image and Business Lens. PBIG innovativeness and the other five PBIG rewards
are fully reference-free. Feasibility preserves the existing formulation:
  feasibility = (2/3) * transformed FActScore
                + (1/3) * transformed MarketDB evidence score
The MarketDB paired-context term also receives the five GT ideas as market anchors;
FActScore remains candidate-only.

Final Eval6M remains the same for both branches: six independent reference-free
PBIG absolute-scoring calls, with no GT/reference ideas exposed to the Judge.

Examples:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -u 3_grpo.py --branch_type open
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -u 3_grpo.py --branch_type closed
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u 3_grpo.py --branch_type open

Resume behavior:
  - This revision writes to /workspace/checkpoints/grpo unless
    MBA_GRPO_OUTPUT_DIR is set.
  - A checkpoint may resume only with the same branch_type and auxiliary setting.
  - Use separate MBA_GRPO_OUTPUT_DIR values to retain open and closed models
    simultaneously.
  - Use --resume_from_checkpoint /path/to/checkpoint-N to select one explicitly.
  - Use --no_resume to start again from the SFT adapter.

Requires transformers==4.45.2, tokenizers==0.20.1, timm==0.9.12.
"""
import os
import sys
import re
import argparse
import contextlib
import socket
import json
import random
import traceback
import logging
import types
import copy
import gc
import math
import time
import warnings
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm
from PIL import Image

import hashlib

import torch
import transformers
from packaging import version as packaging_version
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Keep production logs clean without hiding runtime/OOM errors.
warnings.filterwarnings(
    "ignore",
    message=r"`do_sample` is set to `False`\. However, `temperature` is set.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`do_sample` is set to `False`\. However, `top_p` is set.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`do_sample` is set to `False`\. However, `top_k` is set.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"MatMul8bitLt: inputs will be cast.*",
    category=UserWarning,
)

# ---------------------------------------------------------------------
# Torch compatibility patch for some transformer image processors
# ---------------------------------------------------------------------
if not hasattr(torch, "compiler"):
    torch.compiler = types.SimpleNamespace(is_compiling=lambda: False)
elif not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)
from peft import PeftConfig, get_peft_model
try:
    from peft import set_peft_model_state_dict
except ImportError:
    from peft.utils.save_and_load import set_peft_model_state_dict



os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
os.environ["DISABLE_DEEPSPEED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.disable(logging.CRITICAL)

PIPELINE_REVISION = "branch-open-closed-pbig6-creativity-feasibility-caption-v6"
# Terminology-only migration: accept checkpoints written by the immediately
# preceding identical reward implementation.
COMPATIBLE_PIPELINE_REVISIONS = {
    PIPELINE_REVISION,
    "branch-open-closed-pbig6-creativity-marketability-caption-v6",
}

# ============================================================
# Config
# ============================================================
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
SFT_LORA_DIR = "/workspace/checkpoints/sft"
OUTPUT_DIR = os.environ.get(
    "MBA_GRPO_OUTPUT_DIR",
    "/workspace/checkpoints/grpo",
)
# Reuse the canonical image-level holdout created by SFT. Keep it separate from
# the new GRPO output directory so this clean run starts fresh without changing
# the train/test split.
TEST_FILE = os.environ.get(
    "MBA_TEST_MANIFEST",
    "/workspace/checkpoints/grpo/grpo_test_prompts.json",
)

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

DOMAIN_STRATIFIED_SPLIT_REVISION = "domain_stratified_image_v1"
EXPECTED_DOMAINS = tuple(DOMAIN_QUESTIONS.keys())

BUSINESS_LENS_QUESTIONS = {
    "cost_resource_efficiency": "Focus on cost reduction, resource productivity, waste reduction, energy or material efficiency, and sustainability-linked savings.",
    "intelligent_automation": "Focus on AI automation, sensing, computer vision, predictive analytics, technical intelligence, and scalable decision support.",
    "customer_operations_growth": "Focus on customer experience, service workflow, conversion, retention, operational throughput, and business growth.",
}

# Market DB reward
MARKET_DB_DIR = "/workspace/data/mba/mbalibrary"
MARKET_RETRIEVER_DEVICE = "cpu"
MARKET_TOP_K = 10
MARKET_SCORE_TOP_N = 3
MARKET_MIN_SIM = 0.55
MARKET_MIN_UNIQUE_SOURCES = 1
MARKET_MAX_COPY_SCORE = 0.90
# Candidate-specific absolute MarketDB reward. Retrieval is driven by the generated
# idea, while the paired DDG context contributes through a separate alignment term.
# This prevents the long shared prompt from dominating every candidate embedding.
MARKET_CONTEXT_MAX_CHARS = 2200
MARKET_REFERENCE_CONTEXT_MAX_CHARS = 900
MARKET_SCORE_WEIGHTS = {
    "retrieval_support": 0.55,
    "paired_context_alignment": 0.25,
    "coverage": 0.10,
    "source_diversity": 0.10,
}
if abs(sum(MARKET_SCORE_WEIGHTS.values()) - 1.0) > 1e-8:
    raise RuntimeError(f"Market score weights must sum to 1.0: {MARKET_SCORE_WEIGHTS}")

# Branch-level reward weights. Final Eval6M remains PBIG-6 only and does not use
# these training mixtures. Feasibility preserves the existing 2:1 split between
# transformed FActScore and transformed MarketDB evidence.
OPEN_REWARD_WEIGHT_PBIG = 0.80
OPEN_REWARD_WEIGHT_CREATIVITY = 0.10
OPEN_REWARD_WEIGHT_FEASIBILITY = 0.10
OPEN_NO_AUX_REWARD_WEIGHT_PBIG = 0.90
OPEN_NO_AUX_REWARD_WEIGHT_CREATIVITY = 0.10
CLOSED_REWARD_WEIGHT_CREATIVITY = 0.70
CLOSED_REWARD_WEIGHT_FEASIBILITY = 0.30
for reward_weights in (
    (OPEN_REWARD_WEIGHT_PBIG, OPEN_REWARD_WEIGHT_CREATIVITY, OPEN_REWARD_WEIGHT_FEASIBILITY),
    (OPEN_NO_AUX_REWARD_WEIGHT_PBIG, OPEN_NO_AUX_REWARD_WEIGHT_CREATIVITY),
    (CLOSED_REWARD_WEIGHT_CREATIVITY, CLOSED_REWARD_WEIGHT_FEASIBILITY),
):
    if abs(sum(reward_weights) - 1.0) > 1e-8:
        raise RuntimeError(f"Branch reward weights must sum to 1.0: {reward_weights}")
AUXILIARY_FACTSCORE_WEIGHT = 2.0 / 3.0
AUXILIARY_MARKET_WEIGHT = 1.0 / 3.0
REWARD_MODEL_PATH = "/workspace/mistral-7b-instruct"
FACTSCORE_ROOT = "/workspace/FActScore"
FACTSCORE_DATA_DIR = "/workspace/FActScore/data/"
FACTSCORE_DEMON_DIR = "/workspace/FActScore/data/demos"
FACTSCORE_WIKI_DB = "/workspace/FActScore/data/enwiki-20260101-fts.db"
SIX_DIMENSION_WEIGHTS = {
    "specificity": 0.15,
    "technical_validity": 0.15,
    "innovativeness": 0.25,
    "competitive_advantage": 0.20,
    "need_validity": 0.15,
    "market_size": 0.10,
}
EXPECTED_SIX_DIMENSION_ORDER = (
    "specificity",
    "technical_validity",
    "innovativeness",
    "competitive_advantage",
    "need_validity",
    "market_size",
)
if tuple(SIX_DIMENSION_WEIGHTS.keys()) != EXPECTED_SIX_DIMENSION_ORDER:
    raise RuntimeError(f"Unexpected six-dimension weight order: {SIX_DIMENSION_WEIGHTS}")
if abs(sum(SIX_DIMENSION_WEIGHTS.values()) - 1.0) > 1e-8:
    raise RuntimeError(f"Six-dimension weights must sum to 1.0: {SIX_DIMENSION_WEIGHTS}")

# Reward models
JUDGE_MODEL_ID = "Qwen/Qwen2-VL-72B-Instruct"         # large Qwen-family MLLM preference judge
JUDGE_MAX_NEW_TOKENS = 320
# One dimension-specific ranking only needs a compact JSON array.
JUDGE_SINGLE_DIMENSION_MAX_NEW_TOKENS = 96
# Qwen reward-judge visual-token bounds.
JUDGE_MIN_PIXELS = 4 * 28 * 28
JUDGE_MAX_PIXELS = 128 * 28 * 28

# Each worker owns exactly four visible GPUs. With eight GPUs, two workers run
# independently and synchronize only the trainable policy gradients through DDP.
GPUS_PER_PIPELINE = 4
_RUNTIME_LAYOUT = None


def rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def is_rank0() -> bool:
    return rank() == 0


def get_runtime_layout():
    global _RUNTIME_LAYOUT
    if _RUNTIME_LAYOUT is not None:
        return _RUNTIME_LAYOUT

    count = torch.cuda.device_count()
    workers = world_size()
    if count not in (4, 8):
        raise RuntimeError(f"Expose exactly 4 or 8 GPUs; found {count}.")
    if count != workers * GPUS_PER_PIPELINE:
        raise RuntimeError(
            f"GPU/process mismatch: visible_gpus={count}, workers={workers}, "
            f"expected={workers * GPUS_PER_PIPELINE}."
        )

    base = rank() * GPUS_PER_PIPELINE
    judge_gpus = (base, base + 1)
    policy_gpu = base + 2
    reference_gpu = base + 3
    max_memory = {}
    for gpu_id in judge_gpus:
        total_gib = int(torch.cuda.get_device_properties(gpu_id).total_memory // (1024 ** 3))
        max_memory[gpu_id] = f"{max(8, total_gib - 4)}GiB"

    _RUNTIME_LAYOUT = {
        "gpu_count": count,
        "worker_count": workers,
        "worker_rank": rank(),
        "pipeline_gpus": tuple(range(base, base + GPUS_PER_PIPELINE)),
        "judge_gpus": judge_gpus,
        "policy_gpu": policy_gpu,
        "reference_gpu": reference_gpu,
        "judge_max_memory": max_memory,
    }
    return _RUNTIME_LAYOUT

def first_cuda_parameter_device(model):
    for parameter in model.parameters():
        if parameter.device.type == "cuda":
            return parameter.device
    raise RuntimeError("The sharded Judge has no CUDA parameters.")


def build_qwen2vl_device_map(model_id: str, gpu_ids):
    """Split Qwen2-VL decoder layers contiguously over all Judge GPUs.

    Vision and token embeddings remain on the first Judge GPU. Decoder layers are
    divided as evenly as possible; norm and lm_head stay with the final layer.
    """
    gpu_ids = tuple(int(x) for x in gpu_ids)
    if len(gpu_ids) < 2:
        raise ValueError("Qwen2-VL-72B model parallelism needs at least two GPUs.")

    config = AutoConfig.from_pretrained(model_id)
    text_config = getattr(config, "text_config", config)
    num_layers = int(getattr(text_config, "num_hidden_layers"))

    base, remainder = divmod(num_layers, len(gpu_ids))
    counts = [base + (1 if idx < remainder else 0) for idx in range(len(gpu_ids))]

    first_gpu = gpu_ids[0]
    last_gpu = gpu_ids[-1]
    device_map = {
        "visual": first_gpu,
        "model.embed_tokens": first_gpu,
        "model.rotary_emb": first_gpu,
        "model.norm": last_gpu,
        "lm_head": last_gpu,
    }
    layer_idx = 0
    for gpu_id, count in zip(gpu_ids, counts):
        for _ in range(count):
            device_map[f"model.layers.{layer_idx}"] = gpu_id
            layer_idx += 1

    if layer_idx != num_layers:
        raise RuntimeError(f"Device-map layer assignment failed: {layer_idx}/{num_layers}")
    return device_map


def _move_module_buffers_to_device(module, device):
    """Move only small non-parameter buffers without disturbing model shards."""
    device = torch.device(device)
    for name, buffer in list(module.named_buffers(recurse=False)):
        if buffer is not None and buffer.device != device:
            setattr(module, name, buffer.to(device=device))
    for name in ("inv_freq", "original_inv_freq"):
        value = getattr(module, name, None)
        if torch.is_tensor(value) and value.device != device:
            setattr(module, name, value.to(device=device))
    return module


def pin_qwen2vl_runtime_buffers(model, target_device):
    """Pin Qwen2-VL text rotary buffers to its first decoder GPU."""
    core = getattr(model, "model", None)
    rotary = getattr(core, "rotary_emb", None)
    if rotary is None:
        raise RuntimeError("Qwen2-VL rotary_emb module was not found.")
    _move_module_buffers_to_device(rotary, target_device)
    inv_freq = getattr(rotary, "inv_freq", None)
    if not torch.is_tensor(inv_freq) or inv_freq.device != torch.device(target_device):
        raise RuntimeError(
            f"Failed to pin Qwen2-VL rotary inv_freq to {target_device}; "
            f"found {getattr(inv_freq, 'device', None)}"
        )
    return str(inv_freq.device)

# Training
SEED = 2026
NUM_EPOCHS = 1
NUM_GENERATIONS = 4
MAX_NEW_TOKENS = 192
LEARNING_RATE = 1e-6
TARGET_GLOBAL_PROMPTS_PER_STEP = 4
MAX_GRAD_NORM = 1.0
SAVE_STEPS = 500
LOGGING_STEPS = 5
BETA_KL = 0.02
# GRPO policy and KL reference start from the same SFT LoRA checkpoint.
# They are separate full models on separate GPUs; only the policy LoRA is trainable.
POLICY_ADAPTER_NAME = "default"
REFERENCE_ADAPTER_NAME = "default"  # dedicated frozen model, not a second in-model adapter
# reward 분산이 너무 작을 때 evaluator noise를 표준화로 증폭하지 않는다.
MIN_REWARD_STD_FOR_NORMALIZATION = 0.02
# judge 파싱 실패를 중립 0.5로 숨기지 않는다.
JUDGE_FAILURE_SCORE = 0.0
LOG_JUDGE_RAW_ON_FAILURE = True
POLICY_IMAGE_SIZE = 512
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 128 * 28 * 28

# ============================================================
# Distributed runtime helpers
# ============================================================
def setup_distributed():
    layout = get_runtime_layout()
    torch.cuda.set_device(layout["policy_gpu"])
    return layout["policy_gpu"], world_size()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def local_rank():
    return int(get_runtime_layout()["policy_gpu"])


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean_scalar(x: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(x)
    value = torch.tensor(float(x), dtype=torch.float32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= dist.get_world_size()
    return float(value.item())

# ============================================================
# Text / prompt helpers
# ============================================================
def clean_qwen_vl_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)

    bad_tokens = [
        "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
        "<image>", "</image>", "<img>", "</img>", "<video>", "</video>"
    ]
    for tok in bad_tokens:
        text = text.replace(tok, "")

    text = re.sub(r"<\|[^>]*?\|>", "", text)
    text = re.sub(r"<\s*/?\s*(image|img|video)\s*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_qwen_vl_text(text).lower()).strip()


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


def attach_idea_format_instruction(question: str) -> str:
    base = clean_incontext_prompt_text(question)
    if not base:
        base = "Generate one creative and practical B2B business idea based on this image."
    lowered = base.lower()
    if all(f'"{field}"' in lowered for field in IDEA_JSON_FIELDS):
        return base
    return f"{base}\n\n{IDEA_FORMAT_INSTRUCTION}"


def _shorten_for_field(text: str, max_chars: int = 420) -> str:
    text = clean_qwen_vl_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def format_business_idea_json(topic: str, idea: str) -> str:
    title = _shorten_for_field(topic or "Image-grounded B2B Business Idea", 96)
    description = _shorten_for_field(idea or topic or "A practical business idea grounded in the image and user request.", 520)
    implementation = _shorten_for_field(
        "Build the business idea as a deployable workflow using current software, AI, data, and operational integration methods; "
        f"use the image/request context to define the target users, required inputs, core workflow, and output actions for: {description}",
        520,
    )
    differentiation = _shorten_for_field(
        "Unlike a generic AI platform or dashboard, the business idea is differentiated by a specific customer workflow, "
        f"image-grounded use case, and defensible execution focus around: {title}",
        420,
    )
    return json.dumps(
        {
            "title": title,
            "description": description,
            "implementation": implementation,
            "differentiation": differentiation,
        },
        ensure_ascii=False,
        indent=2,
    )


def _parse_json_object_maybe(value):
    if isinstance(value, dict):
        return value
    raw = str(value or "").strip()
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
        obj = _parse_json_object_maybe(candidate)
        if not isinstance(obj, dict):
            continue
        if all(clean_qwen_vl_text(obj.get(field)) for field in IDEA_JSON_FIELDS):
            return {field: clean_qwen_vl_text(obj[field]) for field in IDEA_JSON_FIELDS}
    return json.loads(format_business_idea_json(topic, str(idea or "")))


def format_structured_idea_json(item: Dict[str, Any], topic: str = "", idea: Any = "") -> str:
    obj = structured_idea_from_item(item, topic=topic, idea=idea)
    return json.dumps({field: obj[field] for field in IDEA_JSON_FIELDS}, ensure_ascii=False, indent=2)


def parse_idea_json_object(text: str):
    raw = clean_qwen_vl_text(text)
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if not all(field in obj and isinstance(obj[field], str) and clean_qwen_vl_text(obj[field]) for field in IDEA_JSON_FIELDS):
        return None
    return {field: clean_qwen_vl_text(obj[field]) for field in IDEA_JSON_FIELDS}


def format_idea_for_judge(text: str) -> str:
    obj = parse_idea_json_object(text)
    if obj is None:
        return "[INVALID_OR_UNSTRUCTURED_IDEA]\n" + clean_qwen_vl_text(text)
    return json.dumps(obj, ensure_ascii=False, indent=2)


def format_idea_for_retrieval(text: str) -> str:
    obj = parse_idea_json_object(text)
    if obj is None:
        return clean_qwen_vl_text(text)
    return (
        f"Business idea title: {obj['title']}\n"
        f"Business idea description: {obj['description']}\n"
        f"Implementation: {obj['implementation']}\n"
        f"Differentiation: {obj['differentiation']}"
    )


def clamp01(x: float) -> float:
    if x is None:
        return 0.0
    try:
        x = float(x)
        if not np.isfinite(x):
            return 0.0
    except Exception:
        return 0.0
    return max(0.0, min(1.0, x))


def count_raw_visual_tokens(obj) -> int:
    s = json.dumps(obj, ensure_ascii=False)
    patterns = [
        "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
        "<image>", "</image>", "<img>", "</img>", "<video>", "</video>"
    ]
    return sum(s.count(p) for p in patterns)


def load_image(image_path: str):
    """Apply the exact Qwen2-VL policy-image preprocessing used by SFT/GRPO/Eval."""
    image = Image.open(image_path).convert("RGB")
    if image.size[0] > POLICY_IMAGE_SIZE or image.size[1] > POLICY_IMAGE_SIZE:
        image = image.resize(
            (POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )
    return image



def make_messages(image_path: str, question: str):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question},
        ]
    }]


def make_prompt_text(processor, image_path: str, question: str):
    messages = make_messages(image_path, attach_idea_format_instruction(question))
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def processor_inputs(processor, prompt_text: str, image, device):
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt", padding=True)
    return inputs.to(device)


def parse_float_score(raw: str, default: float = 0.5) -> float:
    if raw is None:
        return default
    raw = str(raw).strip()
    m = re.search(r"Score:\s*([01](?:\.\d+)?|0?\.\d+)", raw, re.IGNORECASE)
    if not m:
        m = re.search(r"([01](?:\.\d+)?|0?\.\d+)", raw)
        
    if not m:
        return default
    try:
        return clamp01(float(m.group(1)))
    except Exception:
        return default


# ============================================================
# Duplicate/template penalties remain removed; factuality is used only inside feasibility.
# ============================================================
# ============================================================
# Dataset
# ============================================================
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
    """Resolve the canonical six-domain label stored in the SFT test manifest."""
    sample_id = _clean_scalar(data.get("sample_id"))
    source_row = _load_incontext_resources()["manifest_by_sample"].get(sample_id, {})
    domain = _clean_scalar(data.get("domain") or source_row.get("domain"))
    if domain not in EXPECTED_DOMAINS:
        raise RuntimeError(
            f"Unknown or missing domain for sample_id={sample_id or '<missing>'}: {domain!r}. "
            f"Expected one of {EXPECTED_DOMAINS}."
        )
    return domain


def extract_all_samples(data_dir, image_dir):
    samples = []
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".json")])

    for file in files:
        json_path = os.path.join(data_dir, file)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        img_path = resolve_expert_image_path(data, image_dir)
        if not img_path:
            continue
        domain = None

        for result in data.get("results", []):
            if not isinstance(result, dict):
                continue
            business_lens = _clean_scalar(result.get("business_lens"))
            for item in result.get("data", []):
                if not isinstance(item, dict):
                    continue
                topic = clean_qwen_vl_text(item.get("topic") or item.get("title") or "")
                idea = item.get("idea", "")
                if not topic and not idea:
                    continue
                if domain is None:
                    domain = resolve_sample_domain(data)

                item_lens = _clean_scalar(item.get("business_lens") or business_lens)
                if not item_lens:
                    raise RuntimeError(
                        f"Missing Business Lens in expert data: {json_path}"
                    )
                try:
                    query_index = int(item.get("query_index"))
                except Exception:
                    query_index = 0

                question = attach_idea_format_instruction(build_domain_aware_question(data, result, item, img_path))
                expert_answer = format_structured_idea_json(item, topic=topic, idea=idea)
                messages = make_messages(img_path, question)

                if count_raw_visual_tokens(messages) > 0:
                    continue

                samples.append({
                    "image_path": img_path,
                    "domain": domain,
                    "business_lens": item_lens,
                    "query_index": query_index,
                    "question": question,
                    "prompt": messages,
                    # Preserve the original singular manifest value for SFT/Eval6M
                    # compatibility. The creativity-reference list is attached separately.
                    "expert_answers": expert_answer,
                })

    # Offline data contains five structured GT ideas per image and Business Lens.
    # Attach that complete set only for the separate creativity reward without
    # changing the canonical test-manifest payload used by SFT and Eval6M.
    grouped_references: Dict[Tuple[str, str], List[str]] = {}
    grouped_seen: Dict[Tuple[str, str], set] = {}
    for sample in samples:
        key = (canonical_image_key(sample["image_path"]), sample["business_lens"])
        grouped_references.setdefault(key, [])
        grouped_seen.setdefault(key, set())
        reference = sample["expert_answers"]
        normalized = normalize_text_key(reference)
        if normalized and normalized not in grouped_seen[key]:
            grouped_seen[key].add(normalized)
            grouped_references[key].append(reference)

    invalid_groups = {
        key: len(references)
        for key, references in grouped_references.items()
        if len(references) != 5
    }
    if invalid_groups:
        preview = list(invalid_groups.items())[:10]
        raise RuntimeError(
            "Reference-conditioned creativity requires exactly five unique offline reference "
            f"ideas per image/Business-Lens group; invalid groups={preview}"
        )

    for sample in samples:
        key = (canonical_image_key(sample["image_path"]), sample["business_lens"])
        sample["creativity_reference_ideas"] = list(grouped_references[key])

    return samples


def validate_samples(samples):
    bad = []
    for i, sample in enumerate(samples):
        n = count_raw_visual_tokens(sample["prompt"])
        if n > 0:
            bad.append((i, n, sample.get("question", "")[:200]))

    if is_rank0():
        print(f"[Dataset Check] Total samples: {len(samples)}")
        print(f"[Dataset Check] Bad visual-token samples: {len(bad)}")

    if bad:
        raise ValueError("Some prompts still contain raw visual tokens.")
    if len(samples) == 0:
        raise ValueError("No valid samples found.")


def canonical_image_key(image_path: str) -> str:
    """Stable image identity used to prevent train/test image leakage."""
    return os.path.realpath(os.path.abspath(image_path))


def test_manifest_payload(samples):
    return [
        {
            "image_path": sample["image_path"],
            "domain": sample["domain"],
            "question": sample["question"],
            "expert_answers": sample["expert_answers"],
        }
        for sample in samples
    ]


def partition_samples_by_test_manifest(samples, manifest_path=TEST_FILE):
    """Partition rows into the canonical image-level train and test sets.

    SFT creates a domain-stratified 95/5 image split. GRPO trains on every row
    from the train images and excludes every row from held-out images. The manifest
    must contain all six domain labels and is verified exactly against the dataset.
    """
    if not os.path.exists(manifest_path):
        raise RuntimeError(
            f"Missing canonical test manifest: {manifest_path}. "
            "Run 2_sft.py first so the image-level split is fixed before GRPO."
        )

    with open(manifest_path, "r", encoding="utf-8") as handle:
        existing_payload = json.load(handle)
    if not isinstance(existing_payload, list) or not existing_payload:
        raise RuntimeError(f"Invalid or empty test manifest: {manifest_path}")

    required_keys = {"image_path", "domain", "question", "expert_answers"}
    for index, item in enumerate(existing_payload):
        if not isinstance(item, dict) or not required_keys.issubset(item):
            raise RuntimeError(
                f"Malformed test manifest row {index} in {manifest_path}."
            )

    manifest_image_domains = {}
    for item in existing_payload:
        domain = _clean_scalar(item.get("domain"))
        if domain not in EXPECTED_DOMAINS:
            raise RuntimeError(
                f"Test manifest contains unknown domain {domain!r}; expected {EXPECTED_DOMAINS}."
            )
        image_key = canonical_image_key(item["image_path"])
        previous = manifest_image_domains.get(image_key)
        if previous is not None and previous != domain:
            raise RuntimeError(
                f"Test manifest assigns one image to multiple domains: "
                f"{image_key} -> {previous!r}, {domain!r}."
            )
        manifest_image_domains[image_key] = domain

    test_image_keys = set(manifest_image_domains)
    all_image_keys = {canonical_image_key(item["image_path"]) for item in samples}
    unknown_images = test_image_keys - all_image_keys
    if unknown_images:
        preview = sorted(unknown_images)[:5]
        raise RuntimeError(
            f"Test manifest references images absent from the current dataset: {preview}"
        )

    train_samples = [
        sample for sample in samples
        if canonical_image_key(sample["image_path"]) not in test_image_keys
    ]
    test_samples = [
        sample for sample in samples
        if canonical_image_key(sample["image_path"]) in test_image_keys
    ]


    train_domains = {sample["domain"] for sample in train_samples}
    test_domains = {sample["domain"] for sample in test_samples}
    missing_train_domains = set(EXPECTED_DOMAINS) - train_domains
    missing_test_domains = set(EXPECTED_DOMAINS) - test_domains
    if missing_train_domains or missing_test_domains:
        raise RuntimeError(
            "Domain-stratified split must retain all six domains in both partitions: "
            f"missing_train={sorted(missing_train_domains)}, "
            f"missing_test={sorted(missing_test_domains)}"
        )

    expected_payload = test_manifest_payload(test_samples)
    if existing_payload != expected_payload:
        raise RuntimeError(
            "The test manifest is not a complete image-group holdout or no longer "
            f"matches the expert dataset: {manifest_path}. Rebuild it with "
            "2_sft.py --overwrite_test_manifest before running GRPO."
        )

    train_images = {canonical_image_key(item["image_path"]) for item in train_samples}
    test_images = {canonical_image_key(item["image_path"]) for item in test_samples}
    overlap = train_images & test_images
    if overlap:
        raise RuntimeError(
            f"Image leakage detected between GRPO train and test: {sorted(overlap)[:5]}"
        )
    if not train_samples or not test_samples:
        raise RuntimeError("Manifest partition produced an empty train or test set.")

    return train_samples, test_samples


def split_dataset(samples, world_size, global_rank, seed=SEED):
    train_samples, test_samples = partition_samples_by_test_manifest(samples, TEST_FILE)

    # The held-out image set is fixed by the manifest. Shuffle only the GRPO train rows.
    train_samples = list(train_samples)
    random.Random(seed).shuffle(train_samples)
    local_samples = train_samples[global_rank::world_size]

    local_len = torch.tensor([len(local_samples)], device=f"cuda:{local_rank()}", dtype=torch.long)
    if dist.is_available() and dist.is_initialized():
        gathered = [torch.zeros_like(local_len) for _ in range(world_size)]
        dist.all_gather(gathered, local_len)
        min_len = min(int(x.item()) for x in gathered)
    else:
        min_len = len(local_samples)

    local_samples = local_samples[:min_len]
    return train_samples, test_samples, local_samples

# ============================================================
# Market DB reward
# ============================================================
class MarketDBFeasibilityScorer:
    """Candidate-specific absolute market-evidence scorer.

    The previous implementation embedded the complete shared in-context prompt together
    with each candidate. Because that prompt is much longer than the generated idea, all
    four embeddings and FAISS results could become effectively identical. This version:
      1) retrieves MarketDB evidence from each generated idea alone;
      2) measures each idea's semantic alignment to the paired DDG query/evidence context;
      3) when supplied, adds the five same-sample GT ideas as bounded market anchors;
      4) keeps the result as an absolute score rather than forcing a ranking.
    """

    def __init__(
        self,
        market_db_dir: str = MARKET_DB_DIR,
        device: str = MARKET_RETRIEVER_DEVICE,
        top_k: int = MARKET_TOP_K,
        score_top_n: int = MARKET_SCORE_TOP_N,
        min_sim: float = MARKET_MIN_SIM,
        min_unique_sources: int = MARKET_MIN_UNIQUE_SOURCES,
        max_copy_score: float = MARKET_MAX_COPY_SCORE,
    ):
        self.market_db_dir = market_db_dir
        self.device = device
        self.top_k = int(top_k)
        self.score_top_n = int(score_top_n)
        self.min_sim = float(min_sim)
        self.min_unique_sources = max(1, int(min_unique_sources))
        self.max_copy_score = float(max_copy_score)

        self.enabled = False
        self.encoder = None
        self.index = None
        self.meta: List[Dict[str, Any]] = []
        self.config: Dict[str, Any] = {}
        self._faiss = None
        self._runtime_warning_emitted = False
        self._load()

    def _load(self):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            config_path = os.path.join(self.market_db_dir, "config.json")
            index_path = os.path.join(self.market_db_dir, "market_evidence.faiss")
            meta_path = os.path.join(self.market_db_dir, "market_evidence_meta.jsonl")

            if not (os.path.exists(config_path) and os.path.exists(index_path) and os.path.exists(meta_path)):
                if is_rank0():
                    print(f"[MarketReward] Disabled: missing files under {self.market_db_dir}")
                return

            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

            model_name = self.config.get("model_name", "BAAI/bge-base-en-v1.5")
            self.encoder = SentenceTransformer(model_name, device=self.device)
            self.index = faiss.read_index(index_path)
            self._faiss = faiss

            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.meta.append(json.loads(line))

            if int(getattr(self.index, "ntotal", len(self.meta))) != len(self.meta):
                raise RuntimeError(
                    f"MarketDB index/meta size mismatch: index={getattr(self.index, 'ntotal', None)} "
                    f"meta={len(self.meta)}"
                )

            self.enabled = True
            if is_rank0():
                metric_name = self._faiss_metric_name()
                print(
                    f"[MarketReward] Enabled | rows={len(self.meta)} | "
                    f"encoder={model_name} | top_k={self.top_k} | score_top_n={self.score_top_n} | "
                    f"mode=candidate_absolute | faiss_metric={metric_name}"
                )

        except Exception as e:
            self.enabled = False
            if is_rank0():
                print(f"[MarketReward] Disabled due to load error: {repr(e)}")

    def _faiss_metric_name(self) -> str:
        metric_type = getattr(self.index, "metric_type", None)
        if self._faiss is not None:
            if metric_type == getattr(self._faiss, "METRIC_INNER_PRODUCT", object()):
                return "inner_product"
            if metric_type == getattr(self._faiss, "METRIC_L2", object()):
                return "l2"
        return str(metric_type)

    def _index_score_to_similarity(self, raw_score: float) -> float:
        """Convert FAISS output to a higher-is-better [0, 1] similarity."""
        try:
            value = float(raw_score)
        except Exception:
            return 0.0
        metric_type = getattr(self.index, "metric_type", None)
        if self._faiss is not None and metric_type == getattr(self._faiss, "METRIC_L2", None):
            return clamp01(1.0 / (1.0 + max(0.0, value)))
        return clamp01(value)

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        a_tokens = [t for t in normalize_text_key(a).split() if len(t) >= 3]
        b_tokens = [t for t in normalize_text_key(b).split() if len(t) >= 3]
        if not a_tokens or not b_tokens:
            return 0.0

        a_set, b_set = set(a_tokens), set(b_tokens)
        inter = len(a_set & b_set)
        union = len(a_set | b_set)

        jaccard = inter / max(union, 1)
        containment = inter / max(min(len(a_set), len(b_set)), 1)
        return clamp01(0.5 * jaccard + 0.5 * containment)

    @staticmethod
    def _extract_prompt_section(topic: str, section_name: str) -> str:
        text = clean_incontext_prompt_text(topic)
        pattern = (
            rf"(?:^|\n)\[{re.escape(section_name)}\]\n"
            rf"(.*?)(?=\n\[[^\]\n]+\]\n|\Z)"
        )
        match = re.search(pattern, text, flags=re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _reference_market_context(reference_ideas: Any) -> str:
        """Bounded GT-derived market anchors used only inside feasibility."""
        if reference_ideas is None:
            return ""
        if not isinstance(reference_ideas, (list, tuple)) or len(reference_ideas) != 5:
            raise RuntimeError(
                "Feasibility requires the same five GT/reference ideas used by creativity."
            )
        lines = []
        for index, value in enumerate(reference_ideas, 1):
            idea = parse_idea_json_object(str(value))
            if idea is None:
                rendered = clean_qwen_vl_text(value)
            else:
                rendered = (
                    f"title: {idea['title']}; description: {idea['description']}; "
                    f"differentiation: {idea['differentiation']}"
                )
            rendered = _shorten_for_field(rendered, 140)
            lines.append(f"reference_market_anchor_{index}: {rendered}")
        context = "\n".join(lines).strip()
        if len(context) > MARKET_REFERENCE_CONTEXT_MAX_CHARS:
            context = context[:MARKET_REFERENCE_CONTEXT_MAX_CHARS].rstrip()
        return context

    def _paired_market_context(
        self,
        topic: str,
        reference_ideas: Any = None,
    ) -> str:
        """Extract DDG/lens context and append bounded GT market anchors."""
        query = self._extract_prompt_section(topic, "DDG Query")
        evidence_raw = self._extract_prompt_section(topic, "DDG Evidence")
        lens = self._extract_prompt_section(topic, "Business Lens")

        evidence_obj = _parse_json_object_maybe(evidence_raw)
        evidence_lines: List[str] = []
        if isinstance(evidence_obj, dict):
            for key in ("title", "source_domain", "evidence_text"):
                value = clean_qwen_vl_text(evidence_obj.get(key, ""))
                if value:
                    evidence_lines.append(f"{key}: {value}")
            facts = evidence_obj.get("extracted_market_facts") or []
            if not isinstance(facts, list):
                facts = [facts]
            facts = [clean_qwen_vl_text(value) for value in facts if clean_qwen_vl_text(value)]
            if facts:
                evidence_lines.append("market_facts: " + "; ".join(facts))
        elif evidence_raw:
            evidence_lines.append(clean_qwen_vl_text(evidence_raw))

        parts = []
        if query:
            parts.append("paired_ddg_query: " + clean_qwen_vl_text(query))
        if lens:
            parts.append("business_lens: " + clean_qwen_vl_text(lens))
        if evidence_lines:
            parts.append("paired_ddg_evidence:\n" + "\n".join(evidence_lines))

        base_context = "\n".join(parts).strip()
        if len(base_context) > MARKET_CONTEXT_MAX_CHARS:
            base_context = base_context[:MARKET_CONTEXT_MAX_CHARS].rstrip()
        reference_context = self._reference_market_context(reference_ideas)
        return "\n".join(
            value for value in (base_context, reference_context) if value
        ).strip()

    @staticmethod
    def _candidate_query_text(generation: str) -> str:
        # Intentionally exclude the long shared prompt. Candidate differences must drive
        # retrieval; paired-context relevance is scored separately below.
        candidate = format_idea_for_retrieval(generation)
        return "Generated business idea seeking market evidence:\n" + candidate

    def _encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        kwargs = dict(normalize_embeddings=True, show_progress_bar=False)
        try:
            embeddings = self.encoder.encode(texts, batch_size=max(1, min(32, len(texts))), **kwargs)
        except TypeError:
            embeddings = self.encoder.encode(texts, normalize_embeddings=True)
        array = np.asarray(embeddings, dtype="float32")
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return np.ascontiguousarray(array)

    @staticmethod
    def _meta_source(item: Dict[str, Any]) -> str:
        return clean_qwen_vl_text(
            item.get("source") or item.get("source_domain") or item.get("domain") or "Unknown"
        )

    @staticmethod
    def _meta_text(item: Dict[str, Any]) -> str:
        return clean_qwen_vl_text(
            item.get("text") or item.get("evidence_text") or item.get("snippet") or ""
        )

    def _hits_from_search_row(self, score_row, id_row) -> List[Dict[str, Any]]:
        hits = []
        seen = set()
        for raw_score, idx in zip(score_row, id_row):
            idx = int(idx)
            if not (0 <= idx < len(self.meta)):
                continue

            item = dict(self.meta[idx])
            item["score"] = self._index_score_to_similarity(raw_score)
            item["faiss_raw_score"] = float(raw_score)

            title = clean_qwen_vl_text(item.get("title", ""))
            text = self._meta_text(item)
            source = self._meta_source(item)
            key = f"{source}::{normalize_text_key(title or text[:200])}"
            if key in seen:
                continue
            seen.add(key)
            hits.append(item)
        return hits

    def _empty_detail(self) -> Dict[str, Any]:
        return {
            "market_score": 0.0,
            "retrieval_support": 0.0,
            "topn_mean": 0.0,
            "paired_context_alignment": 0.0,
            "coverage": 0.0,
            "source_diversity": 0.0,
            "copy_penalty": 0.0,
            "max_copy": 0.0,
            "top1_score": 0.0,
            "top1_source": None,
            "top1_title": None,
            "num_hits": 0,
            "num_hits_above_min_sim": 0,
            "reference_context_used": False,
            "scoring_mode": "candidate_absolute",
        }

    def _score_from_hits(
        self,
        generation: str,
        hits: List[Dict[str, Any]],
        paired_context_alignment: float,
    ) -> Tuple[float, Dict[str, Any]]:
        if not hits:
            detail = self._empty_detail()
            detail["paired_context_alignment"] = clamp01(paired_context_alignment)
            return 0.0, detail

        scores = [clamp01(h.get("score", 0.0)) for h in hits]
        topn = scores[:max(1, self.score_top_n)]
        topn_mean = sum(topn) / max(len(topn), 1)

        # Calibrate retrieval support relative to the configured evidence threshold.
        # This remains an absolute score: no candidate ranks or group statistics enter it.
        if self.min_sim < 1.0:
            retrieval_support = clamp01(
                (topn_mean - self.min_sim) / max(1e-6, 1.0 - self.min_sim)
            )
        else:
            retrieval_support = float(topn_mean >= self.min_sim)

        above = [h for h in hits if float(h.get("score", 0.0)) >= self.min_sim]
        coverage = clamp01(len(above) / max(self.score_top_n, 1))

        sources = {self._meta_source(h) for h in above if self._meta_source(h)}
        diversity_target = max(
            self.min_unique_sources,
            min(self.score_top_n, max(len(above), 1)),
        )
        source_diversity = clamp01(len(sources) / max(diversity_target, 1))

        max_copy = 0.0
        for h in hits[: self.score_top_n]:
            title = clean_qwen_vl_text(h.get("title", ""))
            evidence_text = f"{title}. {self._meta_text(h)[:500]}"
            max_copy = max(max_copy, self._text_similarity(generation, evidence_text))

        copy_penalty = 0.0
        if max_copy > self.max_copy_score:
            copy_penalty = clamp01(
                (max_copy - self.max_copy_score)
                / max(1e-6, 1.0 - self.max_copy_score)
            )

        paired_context_alignment = clamp01(paired_context_alignment)
        market_score = (
            MARKET_SCORE_WEIGHTS["retrieval_support"] * retrieval_support
            + MARKET_SCORE_WEIGHTS["paired_context_alignment"] * paired_context_alignment
            + MARKET_SCORE_WEIGHTS["coverage"] * coverage
            + MARKET_SCORE_WEIGHTS["source_diversity"] * source_diversity
            - 0.25 * copy_penalty
        )
        market_score = clamp01(market_score)

        top1 = hits[0]
        detail = {
            "market_score": market_score,
            "retrieval_support": retrieval_support,
            "topn_mean": topn_mean,
            "paired_context_alignment": paired_context_alignment,
            "coverage": coverage,
            "source_diversity": source_diversity,
            "copy_penalty": copy_penalty,
            "max_copy": max_copy,
            "top1_score": float(top1.get("score", 0.0)),
            "top1_source": self._meta_source(top1),
            "top1_title": top1.get("title"),
            "num_hits": len(hits),
            "num_hits_above_min_sim": len(above),
            "scoring_mode": "candidate_absolute",
        }
        return market_score, detail

    def score_one(
        self,
        topic: str,
        generation: str,
        reference_ideas: Any = None,
    ) -> Tuple[float, Dict[str, Any]]:
        scores, details = self.score_batched(
            topic, [generation], reference_ideas=reference_ideas
        )
        return scores[0], details[0]

    @torch.no_grad()
    def score_batched(
        self,
        topic: str,
        generations: List[str],
        reference_ideas: Any = None,
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Score every candidate independently without converting scores to ranks."""
        if not generations:
            return [], []
        if not self.enabled or self.encoder is None or self.index is None:
            return [0.0] * len(generations), [self._empty_detail() for _ in generations]

        candidate_texts = [self._candidate_query_text(g) for g in generations]
        valid = [bool(clean_qwen_vl_text(g)) for g in generations]
        paired_context = self._paired_market_context(topic, reference_ideas)

        try:
            encode_texts = list(candidate_texts)
            if paired_context:
                encode_texts.append(paired_context)
            embeddings = self._encode(encode_texts)
            candidate_embeddings = embeddings[:len(generations)]
            context_embedding = embeddings[-1] if paired_context else None

            raw_scores, raw_ids = self.index.search(
                np.ascontiguousarray(candidate_embeddings, dtype="float32"),
                self.top_k,
            )

            scores: List[float] = []
            details: List[Dict[str, Any]] = []
            for index, generation in enumerate(generations):
                if not valid[index]:
                    scores.append(0.0)
                    details.append(self._empty_detail())
                    continue

                context_alignment = 0.0
                if context_embedding is not None:
                    context_alignment = clamp01(float(np.dot(
                        candidate_embeddings[index], context_embedding
                    )))

                hits = self._hits_from_search_row(raw_scores[index], raw_ids[index])
                score, detail = self._score_from_hits(
                    generation=generation,
                    hits=hits,
                    paired_context_alignment=context_alignment,
                )
                detail["reference_context_used"] = bool(reference_ideas)
                scores.append(score)
                details.append(detail)

            return scores, details

        except Exception as exc:
            if is_rank0() and not self._runtime_warning_emitted:
                print(
                    f"[MarketReward] Runtime scoring failure; returning zero absolute "
                    f"market rewards: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                self._runtime_warning_emitted = True
            return [0.0] * len(generations), [self._empty_detail() for _ in generations]


SIX_DIMENSION_RUBRICS = {
    "specificity": (
        "Rank only clarity and concreteness. Prefer one clearly defined offering with a specific target user, "
        "business need, benefit, and operating mechanism. Penalize vague or multi-idea answers."
    ),
    "technical_validity": (
        "Rank only present-day technical and operational feasibility. Prefer implementable sensing, software, AI, "
        "data, integration, deployment, and workflow assumptions. Penalize impossible or unsupported claims."
    ),
    "innovativeness": (
        "Rank only novelty and non-obvious value creation. Prefer a genuinely new mechanism, combination, workflow, "
        "or application. Do not reward generic AI, IoT, dashboard, platform, marketplace, monitoring, or optimization wrappers."
    ),
    "competitive_advantage": (
        "Rank only defensibility and differentiated advantage. Prefer hard-to-copy data, process integration, switching costs, "
        "workflow lock-in, specialized models, operational know-how, network effects, or essential core technology."
    ),
    "need_validity": (
        "Rank only the credibility and urgency of the user/business need. Prefer clear cost, revenue, risk, compliance, "
        "productivity, safety, sustainability, service, or customer-experience impact for a defined buyer."
    ),
    "market_size": (
        "Rank only reachable adoption scale. Prefer a meaningful B2B market with a clear beachhead segment and plausible "
        "expansion path. Penalize one-off or extremely narrow use cases and consumer-only ideas."
    ),
}

PBIG_DIMENSION_ORDER = tuple(SIX_DIMENSION_WEIGHTS.keys())


def _format_six_dimension_weights() -> str:
    return "\n".join(
        f"- {name}: aggregation weight {weight:.2f}"
        for name, weight in SIX_DIMENSION_WEIGHTS.items()
    )


JUDGE_EVALUATION_CONTEXT_SECTIONS = (
    "Domain Context",
    "Image-Specific Annotation",
    CAPTION_SECTION_NAME,
    "Business Lens",
    "DDG Query",
    "DDG Evidence",
)


def extract_prompt_section_for_judge(topic: str, section_name: str) -> str:
    """Extract one canonical context section without carrying generation commands."""
    text = clean_incontext_prompt_text(topic)
    pattern = (
        rf"(?:^|\n)\[{re.escape(section_name)}\]\n"
        rf"(.*?)(?=\n\[[^\]\n]+\]\n|\Z)"
    )
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def build_judge_evaluation_context(topic: str) -> str:
    """Build neutral Judge context from evidence sections only.

    Included evidence context:
      - domain context and image-specific annotation
      - the same auxiliary image caption seen by the policy
      - business lens and paired DDG query/evidence

    Deliberately excluded:
      - [Task] and all idea-generation orders
      - JSON/output-format generation instructions
      - [Grounding Requirements] written for the policy
      - GT/reference/expert idea
    """
    blocks = []
    missing = []
    for section_name in JUDGE_EVALUATION_CONTEXT_SECTIONS:
        value = extract_prompt_section_for_judge(topic, section_name)
        if value:
            blocks.append(f"[{section_name}]\n{value}")
        else:
            missing.append(section_name)

    if missing:
        raise RuntimeError(
            "Cannot construct leakage-free Judge evaluation context; missing canonical "
            f"sections: {missing}. The full generation prompt is intentionally not used "
            "as a fallback."
        )
    return "\n\n".join(blocks)


def _format_creativity_reference_block(reference_ideas: Any) -> str:
    if not isinstance(reference_ideas, (list, tuple)):
        raise RuntimeError(
            "Creativity Judge requires five offline GT/reference ideas from the "
            "same image and Business Lens."
        )
    formatted = [format_idea_for_judge(value) for value in reference_ideas]
    if len(formatted) != 5:
        raise RuntimeError(
            f"Creativity Judge expected exactly five reference ideas; got {len(formatted)}."
        )
    return "\n\n".join(
        f"[Reference Idea {index + 1}]\n{value}"
        for index, value in enumerate(formatted)
    )


CREATIVITY_RUBRIC = (
    "Rank reference-conditioned creativity: meaningful conceptual novelty relative "
    "to the closest of the five supplied offline ideas, while remaining grounded, "
    "specific, and technically plausible."
)


def build_single_dimension_ranking_judge_text(
    topic: str,
    generations: List[str],
    dimension: str,
) -> str:
    """Build one fully reference-free Judge prompt for one PBIG dimension."""
    if dimension not in PBIG_DIMENSION_ORDER:
        raise ValueError(
            f"Unknown PBIG dimension={dimension!r}; expected one of "
            f"{PBIG_DIMENSION_ORDER}."
        )
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(generations)])
    candidates = "\n\n".join(
        f"[Candidate {labels[i]}]\n{format_idea_for_judge(g)}"
        for i, g in enumerate(generations)
    )
    ranking_shape = ", ".join(repr(label) for label in labels).replace("'", '"')
    evaluation_context = build_judge_evaluation_context(topic)
    rubric = SIX_DIMENSION_RUBRICS[dimension]

    return f"""You are a strict comparative multimodal reward judge for GRPO training.
Your only task is to EVALUATE and RANK the supplied candidate ideas. Do not generate,
rewrite, complete, or improve any business idea.

Evaluate exactly ONE target dimension: {dimension}
Target-dimension rubric:
{rubric}

Critical independence rule:
- Judge ONLY the target dimension above.
- Do not score, rank, aggregate, or reason about any other PBIG dimension.
- Do not infer a holistic overall ranking.

Global validity rules relevant to this target dimension:
- Prefer valid JSON with exactly four non-empty string keys: title, description, implementation, differentiation.
- Penalize invalid JSON, missing keys, extra prose outside JSON, or a one-line/unstructured answer when it weakens the target dimension.
- Use the visible image and the neutral evaluation context below, including its domain annotation, auxiliary image caption, business lens, DDG query, and paired DDG evidence.
- Treat the visible image as primary evidence; use the generated caption only as auxiliary semantic context and do not let it override the image.
- Penalize ideas that ignore or contradict the visible image/evaluation context, hallucinate unsupported visual details, or are not business ideas.
- No GT, reference, or expert idea is provided. Never infer or compare with one.
- Use every candidate label exactly once. Ties are not allowed.

[Evaluation Context]
{evaluation_context}

{candidates}

Return ONLY one valid JSON object, with no markdown, explanation, scores, or extra keys:
{{"ranking": [{ranking_shape}]}}
The displayed label order is only the required shape. Replace it with the actual strongest-to-weakest order for the target dimension.
"""


def build_creativity_ranking_judge_text(
    topic: str,
    generations: List[str],
    creativity_reference_ideas: Any,
) -> str:
    """Build the separate GT/reference-conditioned creativity ranking prompt."""
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(generations)])
    candidates = "\n\n".join(
        f"[Candidate {labels[i]}]\n{format_idea_for_judge(g)}"
        for i, g in enumerate(generations)
    )
    ranking_shape = ", ".join(repr(label) for label in labels).replace("'", '"')
    evaluation_context = build_judge_evaluation_context(topic)
    reference_block = _format_creativity_reference_block(
        creativity_reference_ideas
    )

    return f"""You are a strict comparative multimodal creativity reward judge for GRPO training.
Your only task is to EVALUATE and RANK the supplied candidate ideas. Do not generate,
rewrite, complete, or improve any business idea.

Evaluate exactly ONE target metric: reference_conditioned_creativity
Metric rubric:
{CREATIVITY_RUBRIC}

Reference-conditioned creativity rules:
- The five offline reference ideas below come from the same image and Business Lens.
- Compare each complete candidate with its closest reference idea across description, implementation, and differentiation; use title only as a weak topical signal.
- Reward substantive novelty in the core mechanism, target user or problem, workflow, technology or data use, value-creation logic, or application.
- Paraphrasing, title changes, reordered fields, industry substitution, generic technology additions, or cosmetic feature changes do not count as creativity.
- Dissimilarity alone is insufficient: an irrelevant, vague, image-ungrounded, non-business, or technically implausible candidate cannot rank highly.
- The references are comparison anchors only, never answers to copy.
- Do not score or infer any PBIG dimension or feasibility component.

Global validity and grounding rules:
- Prefer valid JSON with exactly four non-empty string keys: title, description, implementation, differentiation.
- Use the visible image and the neutral evaluation context below, including its domain annotation, auxiliary image caption, business lens, DDG query, and paired DDG evidence.
- Treat the visible image as primary evidence; use the generated caption only as auxiliary semantic context and do not let it override the image.
- Use every candidate label exactly once. Ties are not allowed.

[Evaluation Context]
{evaluation_context}

[Offline Reference Ideas — Creativity Only]
{reference_block}

{candidates}

Return ONLY one valid JSON object, with no markdown, explanation, scores, or extra keys:
{{"ranking": [{ranking_shape}]}}
The displayed label order is only the required shape. Replace it with the actual strongest-to-weakest creativity order.
"""

def parse_single_dimension_ranking(raw: str, num_candidates: int) -> List[int]:
    """Parse one dimension-specific strongest-to-weakest candidate ranking."""
    if raw is None:
        raise RuntimeError("Qwen reward judge returned no output.")
    raw_text = str(raw).strip()
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        raise RuntimeError(
            f"Qwen reward judge returned no JSON object. Raw output: {raw_text!r}"
        )
    try:
        obj = json.loads(match.group(0))
    except Exception as exc:
        raise RuntimeError(
            f"Qwen reward judge returned invalid JSON. Raw output: {raw_text!r}"
        ) from exc
    if not isinstance(obj, dict) or set(obj.keys()) != {"ranking"}:
        raise RuntimeError(
            "Dimension-specific reward judge JSON must contain exactly the key "
            f"'ranking'. Raw output: {raw_text!r}"
        )

    ranking = obj.get("ranking")
    valid_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:num_candidates])
    if not isinstance(ranking, list):
        raise RuntimeError(
            f"Dimension-specific reward judge lacks a ranking list. Raw output: {raw_text!r}"
        )
    normalized = [str(value).strip().upper() for value in ranking]
    if len(normalized) != num_candidates:
        raise RuntimeError(
            f"Dimension-specific ranking length={len(normalized)}; expected "
            f"{num_candidates}. Raw output: {raw_text!r}"
        )
    if len(set(normalized)) != num_candidates or set(normalized) != set(valid_labels):
        raise RuntimeError(
            "Dimension-specific ranking must contain every candidate label exactly "
            f"once: {valid_labels}. Raw output: {raw_text!r}"
        )
    return [valid_labels.index(label) for label in normalized]


def rank_positions_to_preference_scores(ranked_indices: List[int], num_candidates: int) -> List[float]:
    """Convert strongest-to-weakest order into normalized 0-1 preference rewards."""
    if num_candidates <= 1:
        return [0.0] * max(num_candidates, 0)
    scores = [0.0] * num_candidates
    for position, original_idx in enumerate(ranked_indices):
        scores[original_idx] = 1.0 - (float(position) / float(num_candidates - 1))
    return [clamp01(x) for x in scores]


def aggregate_six_dimension_scores(
    dimension_scores: Dict[str, List[float]],
    num_candidates: int,
) -> List[float]:
    outputs = []
    for candidate_index in range(num_candidates):
        value = sum(
            SIX_DIMENSION_WEIGHTS[dimension]
            * clamp01(dimension_scores[dimension][candidate_index])
            for dimension in PBIG_DIMENSION_ORDER
        )
        outputs.append(clamp01(value))
    return outputs


def transform_reward_components(
    factscore_raw: float,
    market_raw: float,
    creativity_raw: float,
    dimension_raw: Dict[str, float],
    branch_type: str = "open",
    use_auxiliary: bool = True,
):
    """Combine branch-specific rewards without changing component definitions."""
    branch_type = str(branch_type).strip().lower()
    if branch_type not in {"open", "closed"}:
        raise ValueError(f"Unknown branch_type={branch_type!r}")
    if branch_type == "closed" and not use_auxiliary:
        raise ValueError("closed branch requires feasibility; --no_auxiliary is open-only")

    factuality_raw = clamp01(factscore_raw)
    market_raw = clamp01(market_raw)
    creativity_reward = clamp01(creativity_raw)
    dimension_rewards = {
        dimension: clamp01(dimension_raw.get(dimension, 0.0))
        for dimension in PBIG_DIMENSION_ORDER
    }
    six_axis_reward = clamp01(sum(
        SIX_DIMENSION_WEIGHTS[dimension] * dimension_rewards[dimension]
        for dimension in PBIG_DIMENSION_ORDER
    ))

    # Preserve the existing nonlinear feasibility definition.
    factuality_reward = factuality_raw ** 2
    market_reward = market_raw ** 3
    feasibility_reward = clamp01(
        AUXILIARY_FACTSCORE_WEIGHT * factuality_reward
        + AUXILIARY_MARKET_WEIGHT * market_reward
    ) if use_auxiliary else 0.0

    if branch_type == "open":
        if use_auxiliary:
            weighted_reward = clamp01(
                OPEN_REWARD_WEIGHT_PBIG * six_axis_reward
                + OPEN_REWARD_WEIGHT_CREATIVITY * creativity_reward
                + OPEN_REWARD_WEIGHT_FEASIBILITY * feasibility_reward
            )
        else:
            weighted_reward = clamp01(
                OPEN_NO_AUX_REWARD_WEIGHT_PBIG * six_axis_reward
                + OPEN_NO_AUX_REWARD_WEIGHT_CREATIVITY * creativity_reward
            )
    else:
        weighted_reward = clamp01(
            CLOSED_REWARD_WEIGHT_CREATIVITY * creativity_reward
            + CLOSED_REWARD_WEIGHT_FEASIBILITY * feasibility_reward
        )

    return {
        "factscore_raw": factuality_raw,
        "factscore_reward": factuality_reward,
        "market_raw": market_raw,
        "market_reward": market_reward,
        "feasibility_reward": feasibility_reward,
        # Legacy key retained for existing log/debug consumers.
        "auxiliary_reward": feasibility_reward,
        "creativity_reward": creativity_reward,
        "dimension_raw": dict(dimension_rewards),
        "dimension_rewards": dict(dimension_rewards),
        "six_axis_reward": six_axis_reward,
        "weighted_reward": weighted_reward,
    }

# ============================================================
# Reward manager
# ============================================================
class RewardModelManager:
    def __init__(
        self,
        model_path=REWARD_MODEL_PATH,
        judge_model_id=JUDGE_MODEL_ID,
        judge_axis_batch_size: int = 0,
        factscore_verifier_batch_size: int = 0,
        rollout_superbatch_size: int = 4,
        factscore_precision: str = "auto",
        judge_precision: str = "auto",
        use_auxiliary: bool = True,
        branch_type: str = "open",
    ):
        layout = get_runtime_layout()
        self.local_rank = 0
        self.judge_gpus = tuple(layout["judge_gpus"])
        self.fact_gpu = int(layout["reference_gpu"])
        self.fact_device = f"cuda:{self.fact_gpu}"
        self.branch_type = str(branch_type).strip().lower()
        if self.branch_type not in {"open", "closed"}:
            raise ValueError("branch_type must be one of: open, closed")
        self.use_pbig = self.branch_type == "open"
        self.use_creativity = True
        self.use_auxiliary = bool(use_auxiliary)
        if self.branch_type == "closed" and not self.use_auxiliary:
            raise ValueError("closed branch requires feasibility; --no_auxiliary is open-only")
        self.last_reward_breakdown: Dict[str, Any] = {}
        self.rollout_superbatch_size = max(1, int(rollout_superbatch_size))

        min_judge_gib = min(
            int(torch.cuda.get_device_properties(gpu).total_memory // (1024 ** 3))
            for gpu in self.judge_gpus
        )
        if int(judge_axis_batch_size) > 0:
            self.judge_axis_batch_size = int(judge_axis_batch_size)
        elif min_judge_gib >= 40:
            self.judge_axis_batch_size = 2 * self.rollout_superbatch_size
        else:
            self.judge_axis_batch_size = min(2, 2 * self.rollout_superbatch_size)

        requested_judge_precision = str(judge_precision).lower().strip()
        if requested_judge_precision not in {"auto", "4bit", "8bit"}:
            raise ValueError("judge_precision must be one of: auto, 4bit, 8bit")
        self._judge_precision_requested = requested_judge_precision
        self.judge_precision = (
            "8bit" if requested_judge_precision == "auto" and min_judge_gib >= 46
            else "4bit" if requested_judge_precision == "auto"
            else requested_judge_precision
        )

        requested_fact_precision = str(factscore_precision).lower().strip()
        if requested_fact_precision not in {"auto", "4bit", "bf16"}:
            raise ValueError("factscore_precision must be one of: auto, 4bit, bf16")
        # FActScore shares the fourth pipeline GPU with the frozen 7B KL reference.
        # Use 4-bit by default on A6000-class hardware to keep both resident safely.
        self.factscore_precision = (
            "4bit" if requested_fact_precision == "auto"
            else requested_fact_precision
        ) if self.use_auxiliary else "disabled"
        if self.use_auxiliary:
            if int(factscore_verifier_batch_size) > 0:
                self.factscore_verifier_batch_size = int(factscore_verifier_batch_size)
            else:
                self.factscore_verifier_batch_size = max(
                    8, 8 * self.rollout_superbatch_size
                )
        else:
            self.factscore_verifier_batch_size = 0

        self._reward_executor = ThreadPoolExecutor(
            max_workers=3 if self.use_auxiliary else 1,
            thread_name_prefix=f"reward-r{rank()}",
        )

        quant_cfg_4bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        quant_cfg_8bit = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        self.market_scorer = None
        self.fact_model = None
        self.wiki_fact_scorer = None
        if self.use_auxiliary:
            self.market_scorer = MarketDBFeasibilityScorer()
            self._initialize_factscore(model_path, quant_cfg_4bit)

        # -------- Qwen2-VL-72B Judge sharded over two A6000s --------
        self.judge_processor = Qwen2VLProcessor.from_pretrained(
            judge_model_id, use_fast=False
        )
        self.judge_processor.image_processor.min_pixels = JUDGE_MIN_PIXELS
        self.judge_processor.image_processor.max_pixels = JUDGE_MAX_PIXELS

        def load_judge_with_precision(precision: str):
            quant_cfg = quant_cfg_8bit if precision == "8bit" else quant_cfg_4bit
            reserve_gib = 2 if precision == "8bit" else 4
            max_memory = {}
            for gpu_id in self.judge_gpus:
                total_gib = int(
                    torch.cuda.get_device_properties(gpu_id).total_memory // (1024 ** 3)
                )
                max_memory[gpu_id] = f"{max(8, total_gib - reserve_gib)}GiB"
            print(
                f"[Reward] Loading Qwen Judge ({judge_model_id}) in {precision} across "
                f"visible GPUs {self.judge_gpus} | max_memory={max_memory}...",
                flush=True,
            )
            return Qwen2VLForConditionalGeneration.from_pretrained(
                judge_model_id,
                torch_dtype=torch.bfloat16,
                quantization_config=quant_cfg,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
                device_map=build_qwen2vl_device_map(judge_model_id, self.judge_gpus),
                max_memory=max_memory,
                offload_state_dict=True,
            ).eval()

        try:
            self.judge_model = load_judge_with_precision(self.judge_precision)
        except Exception as exc:
            can_fallback = (
                self.judge_precision == "8bit"
                and self._judge_precision_requested == "auto"
            )
            if not can_fallback:
                raise
            print(
                f"[Rank {rank()}] 8-bit Judge load failed ({type(exc).__name__}: {exc}); "
                "falling back to 4-bit.",
                flush=True,
            )
            gc.collect()
            for gpu_id in self.judge_gpus:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
            self.judge_precision = "4bit"
            self.judge_model = load_judge_with_precision("4bit")
        self.judge_model.generation_config.do_sample = False
        self.judge_model.generation_config.num_beams = 1
        self.judge_model.generation_config.temperature = None
        self.judge_model.generation_config.top_p = None
        self.judge_model.generation_config.top_k = None
        self.judge_input_device = torch.device(f"cuda:{self.judge_gpus[0]}")
        rotary_device = pin_qwen2vl_runtime_buffers(
            self.judge_model, self.judge_input_device
        )
        used_devices = sorted({
            str(v) for v in getattr(self.judge_model, "hf_device_map", {}).values()
        })
        if self.branch_type == "open" and self.use_auxiliary:
            formula_text = (
                f"{OPEN_REWARD_WEIGHT_PBIG:.2f}*WeightedPBIG6+"
                f"{OPEN_REWARD_WEIGHT_CREATIVITY:.2f}*Creativity+"
                f"{OPEN_REWARD_WEIGHT_FEASIBILITY:.2f}*Feasibility"
            )
        elif self.branch_type == "open":
            formula_text = (
                f"{OPEN_NO_AUX_REWARD_WEIGHT_PBIG:.2f}*WeightedPBIG6+"
                f"{OPEN_NO_AUX_REWARD_WEIGHT_CREATIVITY:.2f}*Creativity"
            )
        else:
            formula_text = (
                f"{CLOSED_REWARD_WEIGHT_CREATIVITY:.2f}*Creativity+"
                f"{CLOSED_REWARD_WEIGHT_FEASIBILITY:.2f}*Feasibility"
            )
        print(
            f"[Reward] branch={self.branch_type} | formula={formula_text} | "
            f"Feasibility={AUXILIARY_FACTSCORE_WEIGHT:.3f}*FActScore^2+"
            f"{AUXILIARY_MARKET_WEIGHT:.3f}*MarketDB^3"
            + ("" if self.use_auxiliary else " | feasibility=disabled"),
            flush=True,
        )
        print(
            "[Judge Context] image + neutral domain/annotation/caption/business/DDG context | "
            "PBIG-6 fully reference-free | five offline references exposed to creativity "
            "and bounded MarketDB market context only",
            flush=True,
        )
        print(
            f"[Judge Protocol] PBIG prompts/group={len(PBIG_DIMENSION_ORDER) if self.use_pbig else 0} | "
            "creativity prompts/group=1 | one isolated target per prompt",
            flush=True,
        )
        print(
            f"[Qwen reward judge] ready | revision={PIPELINE_REVISION} | precision={self.judge_precision} | "
            f"input_device={self.judge_input_device} | rotary_device={rotary_device} | "
            f"device_map_targets={used_devices} | transformers={transformers.__version__}",
            flush=True,
        )

    def _initialize_factscore(self, model_path: str, quant_cfg_4bit) -> None:
        if FACTSCORE_ROOT not in sys.path:
            sys.path.append(FACTSCORE_ROOT)

        # Some FActScore modules call nltk.download at import time.
        try:
            import nltk
            original_download = nltk.download

            def quiet_download(*args, **kwargs):
                kwargs["quiet"] = True
                return original_download(*args, **kwargs)

            nltk.download = quiet_download
        except Exception:
            pass

        try:
            from wiki_fact_scorer import WikiFactScorer
            from factscore.factscorer import FactScorer
            from factscore.atomic_facts import AtomicFactGenerator
            from factscore.llama_lm import LlamaModel
        except Exception as exc:
            raise RuntimeError(
                "Auxiliary reward requires the existing /workspace/FActScore setup. "
                "Use --no_auxiliary only for the open branch when intentionally disabling feasibility."
            ) from exc

        print(
            f"[Auxiliary] Loading FActScore verifier on {self.fact_device} "
            f"with precision={self.factscore_precision} alongside the frozen KL reference...",
            flush=True,
        )
        self.fact_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.fact_tokenizer.padding_side = "left"
        if self.fact_tokenizer.pad_token is None:
            self.fact_tokenizer.pad_token = self.fact_tokenizer.eos_token

        if self.factscore_precision == "bf16":
            fact_kwargs = dict(
                torch_dtype=torch.bfloat16,
                device_map={"": self.fact_gpu},
                low_cpu_mem_usage=True,
            )
            try:
                self.fact_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    attn_implementation="flash_attention_2",
                    **fact_kwargs,
                ).eval()
            except (TypeError, ValueError, ImportError):
                self.fact_model = AutoModelForCausalLM.from_pretrained(
                    model_path, **fact_kwargs
                ).eval()
        else:
            self.fact_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quant_cfg_4bit,
                device_map={"": self.fact_gpu},
                low_cpu_mem_usage=True,
            ).eval()

        self.shared_llama = LlamaModel(model_path=model_path)
        self.shared_llama.model = self.fact_model
        self.shared_llama.tokenizer = self.fact_tokenizer
        self.fact_scorer = FactScorer(
            data_dir=FACTSCORE_DATA_DIR,
            model_name="retrieval+llama",
        )
        if self.fact_scorer.af_generator is None:
            self.fact_scorer.af_generator = AtomicFactGenerator(
                key_path="dummy",
                demon_dir=FACTSCORE_DEMON_DIR,
                shared_model=self.shared_llama,
            )
        self.fact_scorer.lm = self.shared_llama
        self.wiki_fact_scorer = WikiFactScorer(
            fts_db_path=FACTSCORE_WIKI_DB,
            af_generator=self.fact_scorer.af_generator,
            model=self.fact_model,
            tokenizer=self.fact_tokenizer,
            device=self.fact_device,
            retrieval_k=4,
            verifier_batch_size=self.factscore_verifier_batch_size,
            max_checkable_atoms=12,
            scoring_mode="factscore",
            debug_prob=0.0,
        )
        print(
            f"[Auxiliary] FActScore ready | verifier_batch_size="
            f"{self.factscore_verifier_batch_size} | MarketDB=enabled",
            flush=True,
        )

    def _factscore_batched(self, generations: List[str]) -> List[float]:
        if not self.use_auxiliary or self.wiki_fact_scorer is None:
            return [0.0] * len(generations)
        torch.cuda.set_device(self.fact_gpu)
        while True:
            try:
                return [
                    clamp01(value)
                    for value in self.wiki_fact_scorer.score_batched(generations)
                ]
            except torch.cuda.OutOfMemoryError:
                old_batch = int(self.factscore_verifier_batch_size)
                if old_batch <= 8:
                    raise
                new_batch = max(8, old_batch // 2)
                self.factscore_verifier_batch_size = new_batch
                if hasattr(self.wiki_fact_scorer, "verifier_batch_size"):
                    self.wiki_fact_scorer.verifier_batch_size = new_batch
                with torch.cuda.device(self.fact_gpu):
                    torch.cuda.empty_cache()
                print(
                    f"[Rank {rank()}] FActScore verifier OOM; retrying "
                    f"batch_size={new_batch} (was {old_batch}).",
                    flush=True,
                )

    def _run_qwen_judge_multi_batch(
        self,
        image_paths: List[str],
        prompts: List[str],
        max_new_tokens: int,
    ) -> List[str]:
        if len(image_paths) != len(prompts):
            raise ValueError("image_paths and prompts must have identical lengths")
        images = [Image.open(path).convert("RGB") for path in image_paths]

        def run_with_batch_size(batch_size: int) -> List[str]:
            outputs: List[str] = []
            for offset in range(0, len(prompts), batch_size):
                prompt_chunk = prompts[offset: offset + batch_size]
                image_chunk = images[offset: offset + batch_size]
                messages_batch = [[{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }] for image, prompt in zip(image_chunk, prompt_chunk)]
                prompt_texts = [
                    self.judge_processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    for messages in messages_batch
                ]
                inputs = self.judge_processor(
                    text=prompt_texts,
                    images=image_chunk,
                    padding=True,
                    return_tensors="pt",
                ).to(self.judge_input_device)
                generated_ids = self.judge_model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    num_beams=1,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self.judge_processor.tokenizer.pad_token_id,
                    eos_token_id=self.judge_processor.tokenizer.eos_token_id,
                )
                input_width = inputs.input_ids.shape[1]
                trimmed = [row[input_width:] for row in generated_ids]
                outputs.extend(self.judge_processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ))
                del inputs, generated_ids, trimmed
            return [str(value).strip() for value in outputs]

        batch_size = max(1, min(self.judge_axis_batch_size, len(prompts)))
        while True:
            try:
                return run_with_batch_size(batch_size)
            except torch.cuda.OutOfMemoryError:
                if batch_size <= 1:
                    raise
                next_batch = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                print(
                    f"[Rank {rank()}] Judge request batch OOM; retrying "
                    f"batch_size={next_batch} (was {batch_size}).",
                    flush=True,
                )
                batch_size = next_batch
                self.judge_axis_batch_size = next_batch

    def _run_qwen_judge_batch(
        self, image_path: str, prompts: List[str], max_new_tokens: int
    ) -> List[str]:
        return self._run_qwen_judge_multi_batch(
            [image_path] * len(prompts), prompts, max_new_tokens
        )

    def _run_qwen_judge(self, image_path: str, prompt: str, max_new_tokens: int, **_) -> str:
        return self._run_qwen_judge_batch(image_path, [prompt], max_new_tokens)[0]

    @torch.no_grad()
    def axiswise_six_metric_preference_multi_batched(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run six independent dimension-specific rankings per candidate group.

        For every item, this creates exactly six independent Judge prompt sequences.
        Each sequence contains the same candidate group but exactly one PBIG rubric and
        returns exactly one ranking. Sequences may be processed together as a GPU batch,
        but they never share prompt context or a joint six-dimension output.
        """
        all_requests: List[Dict[str, Any]] = []

        for item_index, item in enumerate(items):
            generations = item["generations"]
            num_candidates = len(generations)
            if num_candidates < 2:
                raise RuntimeError(
                    "Axis-wise PBIG preference reward requires at least two candidates."
                )
            stable_text = "||".join(clean_qwen_vl_text(g) for g in generations)

            for dimension in PBIG_DIMENSION_ORDER:
                # Use an independent deterministic candidate permutation for each axis
                # to reduce shared position bias while preserving reproducibility.
                order_seed = int(hashlib.sha256(
                    f"independent_dimension||{dimension}||{stable_text}".encode("utf-8")
                ).hexdigest()[:8], 16)
                shown_order = list(range(num_candidates))
                random.Random(order_seed).shuffle(shown_order)
                prompt = build_single_dimension_ranking_judge_text(
                    topic=item["topic"],
                    generations=[generations[index] for index in shown_order],
                    dimension=dimension,
                )
                all_requests.append({
                    "item_index": item_index,
                    "dimension": dimension,
                    "shown_order": shown_order,
                    "image_path": item["image_path"],
                    "prompt": prompt,
                })

        # These are independent prompt sequences. Batching only parallelizes tensor
        # execution; no sequence can attend to another dimension's prompt or output.
        raw_outputs = self._run_qwen_judge_multi_batch(
            image_paths=[request["image_path"] for request in all_requests],
            prompts=[request["prompt"] for request in all_requests],
            max_new_tokens=JUDGE_SINGLE_DIMENSION_MAX_NEW_TOKENS,
        )
        if len(raw_outputs) != len(all_requests):
            raise RuntimeError(
                "Dimension-independent Judge output count mismatch: "
                f"requests={len(all_requests)}, outputs={len(raw_outputs)}"
            )

        partial: List[Dict[str, Any]] = [
            {
                "rankings": {},
                "dimension_scores": {},
                "raw_judge_outputs": {},
            }
            for _ in items
        ]

        for request, raw in zip(all_requests, raw_outputs):
            item_index = request["item_index"]
            dimension = request["dimension"]
            num_candidates = len(items[item_index]["generations"])
            try:
                ranked_shown = parse_single_dimension_ranking(raw, num_candidates)
            except RuntimeError as first_error:
                if is_rank0():
                    print(
                        f"[PBIG Dimension Parse Retry p{item_index} {dimension}] "
                        f"{first_error}",
                        flush=True,
                    )
                retry_prompt = request["prompt"] + f"""

Your previous response did not satisfy the required JSON schema.
Evaluate only {dimension}. Return only {{"ranking": [...]}} with every candidate label appearing exactly once. No explanation.
"""
                retry_raw = self._run_qwen_judge(
                    request["image_path"],
                    retry_prompt,
                    JUDGE_SINGLE_DIMENSION_MAX_NEW_TOKENS,
                )
                try:
                    ranked_shown = parse_single_dimension_ranking(
                        retry_raw, num_candidates
                    )
                    raw = retry_raw
                except RuntimeError as second_error:
                    raise RuntimeError(
                        "Dimension-independent PBIG Judge parse failed twice for "
                        f"item={item_index}, dimension={dimension}. "
                        f"First={first_error}; Second={second_error}"
                    ) from second_error

            ranked_original = [
                request["shown_order"][shown_index]
                for shown_index in ranked_shown
            ]
            partial[item_index]["rankings"][dimension] = ranked_original
            partial[item_index]["dimension_scores"][dimension] = (
                rank_positions_to_preference_scores(
                    ranked_original, num_candidates
                )
            )
            partial[item_index]["raw_judge_outputs"][dimension] = str(raw)

        outputs: List[Dict[str, Any]] = []
        for item_index, state in enumerate(partial):
            missing_rankings = [
                dimension for dimension in PBIG_DIMENSION_ORDER
                if dimension not in state["rankings"]
            ]
            missing_scores = [
                dimension for dimension in PBIG_DIMENSION_ORDER
                if dimension not in state["dimension_scores"]
            ]
            if missing_rankings or missing_scores:
                raise RuntimeError(
                    "Incomplete dimension-independent Judge results for "
                    f"item={item_index}: missing_rankings={missing_rankings}, "
                    f"missing_scores={missing_scores}"
                )
            num_candidates = len(items[item_index]["generations"])
            aggregate_scores = aggregate_six_dimension_scores(
                state["dimension_scores"], num_candidates
            )
            outputs.append({
                "rankings": state["rankings"],
                "dimension_scores": state["dimension_scores"],
                "aggregate_scores": aggregate_scores,
                "raw_judge_outputs": state["raw_judge_outputs"],
                # Keep the legacy key for checkpoint/debug consumers, but its value
                # now records six independent outputs rather than one joint response.
                "raw_judge_output": json.dumps(
                    state["raw_judge_outputs"], ensure_ascii=False
                ),
            })

        return outputs

    @torch.no_grad()
    def creativity_preference_multi_batched(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run one isolated reference-conditioned creativity ranking per group."""
        requests: List[Dict[str, Any]] = []
        for item_index, item in enumerate(items):
            generations = item["generations"]
            num_candidates = len(generations)
            if num_candidates < 2:
                raise RuntimeError(
                    "Creativity preference reward requires at least two candidates."
                )
            stable_text = "||".join(clean_qwen_vl_text(g) for g in generations)
            order_seed = int(hashlib.sha256(
                f"reference_conditioned_creativity||{stable_text}".encode("utf-8")
            ).hexdigest()[:8], 16)
            shown_order = list(range(num_candidates))
            random.Random(order_seed).shuffle(shown_order)
            prompt = build_creativity_ranking_judge_text(
                topic=item["topic"],
                generations=[generations[index] for index in shown_order],
                creativity_reference_ideas=item.get("creativity_reference_ideas"),
            )
            requests.append({
                "item_index": item_index,
                "shown_order": shown_order,
                "image_path": item["image_path"],
                "prompt": prompt,
            })

        raw_outputs = self._run_qwen_judge_multi_batch(
            image_paths=[request["image_path"] for request in requests],
            prompts=[request["prompt"] for request in requests],
            max_new_tokens=JUDGE_SINGLE_DIMENSION_MAX_NEW_TOKENS,
        )
        if len(raw_outputs) != len(requests):
            raise RuntimeError(
                "Creativity Judge output count mismatch: "
                f"requests={len(requests)}, outputs={len(raw_outputs)}"
            )

        outputs: List[Dict[str, Any]] = []
        for request, raw in zip(requests, raw_outputs):
            item_index = request["item_index"]
            num_candidates = len(items[item_index]["generations"])
            try:
                ranked_shown = parse_single_dimension_ranking(raw, num_candidates)
            except RuntimeError as first_error:
                if is_rank0():
                    print(
                        f"[Creativity Parse Retry p{item_index}] {first_error}",
                        flush=True,
                    )
                retry_prompt = request["prompt"] + """

Your previous response did not satisfy the required JSON schema.
Evaluate only reference_conditioned_creativity. Return only {"ranking": [...]} with every candidate label appearing exactly once. No explanation.
"""
                retry_raw = self._run_qwen_judge(
                    request["image_path"],
                    retry_prompt,
                    JUDGE_SINGLE_DIMENSION_MAX_NEW_TOKENS,
                )
                try:
                    ranked_shown = parse_single_dimension_ranking(
                        retry_raw, num_candidates
                    )
                    raw = retry_raw
                except RuntimeError as second_error:
                    raise RuntimeError(
                        "Creativity Judge parse failed twice for "
                        f"item={item_index}. First={first_error}; Second={second_error}"
                    ) from second_error

            ranked_original = [
                request["shown_order"][shown_index]
                for shown_index in ranked_shown
            ]
            outputs.append({
                "creativity_ranking": ranked_original,
                "creativity_scores": rank_positions_to_preference_scores(
                    ranked_original, num_candidates
                ),
                "raw_creativity_output": str(raw),
            })
        return outputs


    def training_preference_multi_batched(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run only the Judge rewards enabled by the selected branch."""
        if self.use_pbig:
            pbig_outputs = self.axiswise_six_metric_preference_multi_batched(items)
        else:
            pbig_outputs = [
                {
                    "rankings": {},
                    "dimension_scores": {
                        dimension: [0.0] * len(item["generations"])
                        for dimension in PBIG_DIMENSION_ORDER
                    },
                    "aggregate_scores": [0.0] * len(item["generations"]),
                    "raw_judge_outputs": {},
                    "raw_judge_output": "{}",
                }
                for item in items
            ]
        creativity_outputs = self.creativity_preference_multi_batched(items)
        merged = []
        for pbig_output, creativity_output in zip(
            pbig_outputs, creativity_outputs
        ):
            merged.append({**pbig_output, **creativity_output})
        return merged


    # Backward-compatible name for any external caller. It returns the weighted
    # six-axis preference values, while the training path keeps all components.
    @torch.no_grad()
    def six_metric_preference_multi_batched(
        self,
        items: List[Dict[str, Any]],
    ) -> List[List[float]]:
        return [
            item["aggregate_scores"]
            for item in self.axiswise_six_metric_preference_multi_batched(items)
        ]

    @torch.no_grad()
    def six_metric_preference_batched(
        self,
        image_path: str,
        topic: str,
        generations: List[str],
    ) -> List[float]:
        return self.axiswise_six_metric_preference_multi_batched([{
            "image_path": image_path,
            "topic": topic,
            "generations": generations,
        }])[0]["aggregate_scores"]

    def _finalize_reward_one(
        self,
        generations: List[str],
        f_scores: List[float],
        m_scores: List[float],
        m_details: List[Dict[str, Any]],
        preference_output: Dict[str, Any],
        item_index: int,
    ) -> Dict[str, Any]:
        dimension_scores: Dict[str, List[float]] = preference_output["dimension_scores"]
        aggregate_scores: List[float] = preference_output["aggregate_scores"]
        creativity_scores: List[float] = preference_output["creativity_scores"]

        rewards: List[float] = []
        feasibility_rewards: List[float] = []
        creativity_rewards: List[float] = []
        fact_rewards: List[float] = []
        market_rewards: List[float] = []
        six_axis_rewards: List[float] = []
        component_rewards: List[Dict[str, float]] = []

        for candidate_index in range(len(generations)):
            candidate_dimensions = {
                dimension: dimension_scores[dimension][candidate_index]
                for dimension in PBIG_DIMENSION_ORDER
            }
            component = transform_reward_components(
                factscore_raw=f_scores[candidate_index],
                market_raw=m_scores[candidate_index],
                creativity_raw=creativity_scores[candidate_index],
                dimension_raw=candidate_dimensions,
                branch_type=self.branch_type,
                use_auxiliary=self.use_auxiliary,
            )
            fact_rewards.append(component["factscore_reward"])
            market_rewards.append(component["market_reward"])
            feasibility_rewards.append(component["feasibility_reward"])
            creativity_rewards.append(component["creativity_reward"])
            six_axis_rewards.append(component["six_axis_reward"])
            rewards.append(component["weighted_reward"])

            candidate_components: Dict[str, float] = {
                "reference_conditioned_creativity": component["creativity_reward"],
                "total": component["weighted_reward"],
            }
            if self.use_pbig:
                candidate_components.update({
                    **component["dimension_rewards"],
                    "weighted_six_axis": component["six_axis_reward"],
                })
            if self.use_auxiliary:
                candidate_components.update({
                    "feasibility": component["feasibility_reward"],
                    # Legacy key retained for downstream log readers.
                    "auxiliary_feasibility": component["feasibility_reward"],
                    "factscore_raw": component["factscore_raw"],
                    "factscore_transformed": component["factscore_reward"],
                    "market_raw": component["market_raw"],
                    "market_transformed": component["market_reward"],
                })
            component_rewards.append(candidate_components)

        count = max(len(generations), 1)
        breakdown: Dict[str, Any] = {
            "branch_type": self.branch_type,
            "fact": sum(f_scores) / count,
            "fact_reward": sum(fact_rewards) / count,
            "market": sum(m_scores) / count,
            "market_reward": sum(market_rewards) / count,
            "feasibility": sum(feasibility_rewards) / count,
            # Legacy key retained for existing training dashboards.
            "auxiliary": sum(feasibility_rewards) / count,
            "creativity": sum(creativity_rewards) / count,
            "creative": sum(creativity_rewards) / count,
            "creative_raw": sum(creativity_scores) / count,
            "vlm": sum(aggregate_scores) / count,
            "six_metric": sum(six_axis_rewards) / count,
            "six_axis_weighted": sum(six_axis_rewards) / count,
            "raw_reward": sum(rewards) / count,
            "final_reward": sum(rewards) / count,
            "market_top1": (
                sum(float(d.get("top1_score", 0.0)) for d in m_details) / count
                if self.use_auxiliary else 0.0
            ),
            "reward_std": float(np.std(rewards)),
            "fact_std": float(np.std(f_scores)),
            "market_std": float(np.std(m_scores)),
            "feasibility_std": float(np.std(feasibility_rewards)),
            "auxiliary_std": float(np.std(feasibility_rewards)),
            "creativity_std": float(np.std(creativity_rewards)),
            "vlm_std": float(np.std(aggregate_scores)),
            "dup_penalty": 0.0,
            "template_penalty": 0.0,
        }
        for dimension in PBIG_DIMENSION_ORDER:
            values = dimension_scores[dimension]
            breakdown[dimension] = sum(values) / count
            breakdown[f"{dimension}_std"] = float(np.std(values))

        if is_rank0():
            axis_summary = " | ".join(
                f"{dimension}={breakdown[dimension]:.3f}"
                for dimension in PBIG_DIMENSION_ORDER
            ) if self.use_pbig else "PBIG6=skipped"
            if self.branch_type == "open" and self.use_auxiliary:
                formula = (
                    f"{OPEN_REWARD_WEIGHT_PBIG:.2f}*Weighted6+"
                    f"{OPEN_REWARD_WEIGHT_CREATIVITY:.2f}*Creativity+"
                    f"{OPEN_REWARD_WEIGHT_FEASIBILITY:.2f}*Feasibility"
                )
            elif self.branch_type == "open":
                formula = (
                    f"{OPEN_NO_AUX_REWARD_WEIGHT_PBIG:.2f}*Weighted6+"
                    f"{OPEN_NO_AUX_REWARD_WEIGHT_CREATIVITY:.2f}*Creativity"
                )
            else:
                formula = (
                    f"{CLOSED_REWARD_WEIGHT_CREATIVITY:.2f}*Creativity+"
                    f"{CLOSED_REWARD_WEIGHT_FEASIBILITY:.2f}*Feasibility"
                )
            feasibility_summary = (
                f"FAct={breakdown['fact']:.3f} | "
                f"Market={breakdown['market']:.3f} | "
                f"Feasibility={breakdown['feasibility']:.3f} | "
                if self.use_auxiliary else "Feasibility=disabled | "
            )
            print(
                f"[Reward Breakdown p{item_index}] branch={self.branch_type} | "
                f"{axis_summary} | Creativity={breakdown['creativity']:.3f} | "
                + feasibility_summary
                + f"Weighted6={breakdown['six_axis_weighted']:.3f} | "
                f"R={breakdown['final_reward']:.3f} | "
                f"Rstd={breakdown['reward_std']:.4f} | formula={formula}",
                flush=True,
            )
            component_count = (
                8 if self.branch_type == "open" and self.use_auxiliary
                else 7 if self.branch_type == "open"
                else 2
            )
            signal = (
                f"[GRPO {component_count}-Component Signal p{item_index}] "
                f"Creativity={[round(float(x), 3) for x in creativity_scores]}"
            )
            if self.use_pbig:
                signal += " | " + " | ".join(
                    f"{dimension}={[round(float(x), 3) for x in dimension_scores[dimension]]}"
                    for dimension in PBIG_DIMENSION_ORDER
                )
            if self.use_auxiliary:
                signal += (
                    f" | FActRaw={[round(float(x), 4) for x in f_scores]}"
                    f" | MarketRaw={[round(float(x), 4) for x in m_scores]}"
                    f" | Feasibility={[round(float(x), 4) for x in feasibility_rewards]}"
                )
            if self.use_pbig:
                signal += f" | Weighted6={[round(float(x), 3) for x in aggregate_scores]}"
            signal += f" | Total={[round(float(x), 3) for x in rewards]}"
            print(signal, flush=True)
            if self.use_auxiliary:
                print(
                    f"[Feasibility Detail p{item_index}] "
                    f"support={[round(float(d.get('retrieval_support', 0.0)), 4) for d in m_details]} | "
                    f"paired_context={[round(float(d.get('paired_context_alignment', 0.0)), 4) for d in m_details]} | "
                    f"top1={[round(float(d.get('top1_score', 0.0)), 4) for d in m_details]} | "
                    f"GTctx={[bool(d.get('reference_context_used', False)) for d in m_details]}",
                    flush=True,
                )

        rankings = dict(preference_output.get("rankings", {}))
        rankings["reference_conditioned_creativity"] = preference_output.get(
            "creativity_ranking", []
        )
        return {
            "rewards": rewards,
            "component_rewards": component_rewards,
            "rankings": rankings,
            "breakdown": breakdown,
        }

    def total_reward_multi_batched(
        self, items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        def preference_job():
            torch.cuda.set_device(self.judge_gpus[0])
            return self.training_preference_multi_batched(items)

        preference_future = self._reward_executor.submit(preference_job)

        if self.use_auxiliary:
            flat_generations: List[str] = []
            spans: List[Tuple[int, int]] = []
            for item in items:
                start = len(flat_generations)
                flat_generations.extend(item["generations"])
                spans.append((start, len(flat_generations)))

            def fact_job():
                return self._factscore_batched(flat_generations)

            def market_job():
                return [
                    self.market_scorer.score_batched(
                        item["topic"],
                        item["generations"],
                        reference_ideas=item.get("creativity_reference_ideas"),
                    )
                    for item in items
                ]

            fact_future = self._reward_executor.submit(fact_job)
            market_future = self._reward_executor.submit(market_job)
            flat_f_scores = fact_future.result()
            market_outputs = market_future.result()
        else:
            spans = []
            flat_f_scores = []
            market_outputs = [
                (
                    [0.0] * len(item["generations"]),
                    [{} for _ in item["generations"]],
                )
                for item in items
            ]

        preference_outputs = preference_future.result()
        results = []
        for item_index, (item, market_output, preference_output) in enumerate(zip(
            items, market_outputs, preference_outputs
        )):
            m_scores, m_details = market_output
            if self.use_auxiliary:
                start, end = spans[item_index]
                f_scores = list(flat_f_scores[start:end])
            else:
                f_scores = [0.0] * len(item["generations"])
            results.append(self._finalize_reward_one(
                generations=item["generations"],
                f_scores=f_scores,
                m_scores=list(m_scores),
                m_details=list(m_details),
                preference_output=preference_output,
                item_index=item_index,
            ))
        self.last_reward_breakdown = dict(results[-1]["breakdown"]) if results else {}
        return results

    def total_reward_batched(
        self,
        image_path: str,
        topic: str,
        generations: List[str],
        creativity_reference_ideas: Any,
    ) -> List[float]:
        result = self.total_reward_multi_batched([{
            "image_path": image_path,
            "topic": topic,
            "generations": generations,
            "creativity_reference_ideas": creativity_reference_ideas,
        }])[0]
        self.last_reward_breakdown = dict(result["breakdown"])
        return list(result["rewards"])

    def close(self):
        """Release the internal reward executor after all prefetched jobs finish."""
        self._reward_executor.shutdown(wait=True, cancel_futures=False)

# ============================================================
# PEFT adapter compatibility loader
# ============================================================
def _load_local_adapter_state_dict(adapter_dir: str):
    """Load a local PEFT adapter checkpoint without relying on PEFT key-prefix heuristics."""
    candidates = [
        os.path.join(adapter_dir, "adapter_model.safetensors"),
        os.path.join(adapter_dir, "adapter_model.bin"),
        os.path.join(adapter_dir, POLICY_ADAPTER_NAME, "adapter_model.safetensors"),
        os.path.join(adapter_dir, POLICY_ADAPTER_NAME, "adapter_model.bin"),
    ]

    checkpoint_path = next((path for path in candidates if os.path.isfile(path)), None)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No PEFT adapter weights found under {adapter_dir}. "
            "Expected adapter_model.safetensors or adapter_model.bin."
        )

    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path, device="cpu")
    else:
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(f"Invalid or empty PEFT state dict: {checkpoint_path}")
    return state_dict, checkpoint_path


def _strip_lora_adapter_name(key: str) -> str:
    """Convert ...lora_A.<adapter>.weight into the adapter-name-free PEFT save format."""
    return re.sub(
        r"(\.lora_(?:A|B|embedding_A|embedding_B))\.[^.]+(?=\.(?:weight|bias)$)",
        r"\1",
        key,
    )


def _lora_key_tail(key: str) -> str:
    """Architecture-independent suffix used across old/new Qwen2-VL module layouts."""
    key = _strip_lora_adapter_name(key)
    match = re.search(r"(layers\.\d+\..+)$", key)
    if match:
        return match.group(1)
    parts = key.split(".")
    return ".".join(parts[-7:])


def _remap_adapter_state_dict_for_model(
    model,
    raw_state_dict: Dict[str, torch.Tensor],
    adapter_name: str,
):
    """Map a saved LoRA checkpoint onto the current Transformers Qwen2-VL module layout."""
    expected_full = {
        key: tensor.shape
        for key, tensor in model.state_dict().items()
        if "lora_" in key and f".{adapter_name}." in key
    }
    if not expected_full:
        raise RuntimeError(f"No LoRA parameters were created for adapter {adapter_name!r}.")

    expected_no_name = {
        _strip_lora_adapter_name(key): (key, shape)
        for key, shape in expected_full.items()
    }

    tail_index = {}
    for no_name_key, (full_key, shape) in expected_no_name.items():
        tail_index.setdefault((_lora_key_tail(no_name_key), tuple(shape)), []).append(
            (no_name_key, full_key)
        )

    raw_lora = {
        key: tensor
        for key, tensor in raw_state_dict.items()
        if "lora_" in key
    }
    if not raw_lora:
        raise RuntimeError("The SFT checkpoint contains no LoRA tensors.")

    remapped = {}
    source_to_full = {}
    unmatched = []

    for source_key, tensor in raw_lora.items():
        normalized_source = _strip_lora_adapter_name(source_key)
        target = None

        direct = expected_no_name.get(normalized_source)
        if direct is not None and tuple(direct[1]) == tuple(tensor.shape):
            target = (normalized_source, direct[0])
        else:
            candidates = tail_index.get(
                (_lora_key_tail(normalized_source), tuple(tensor.shape)),
                [],
            )
            if len(candidates) == 1:
                target = candidates[0]

        if target is None:
            unmatched.append(source_key)
            continue

        no_name_key, full_key = target
        if no_name_key in remapped:
            raise RuntimeError(
                f"Multiple checkpoint tensors map to the same LoRA parameter: {full_key}"
            )
        remapped[no_name_key] = tensor
        source_to_full[source_key] = full_key

    mapped_full = set(source_to_full.values())
    missing_expected = [key for key in expected_full if key not in mapped_full]

    if unmatched or missing_expected:
        raise RuntimeError(
            "SFT LoRA checkpoint is incompatible after automatic Qwen2-VL key remapping. "
            f"mapped={len(remapped)}/{len(raw_lora)}, "
            f"unmatched_sample={unmatched[:5]}, "
            f"missing_sample={missing_expected[:5]}"
        )

    return remapped, source_to_full


def _load_adapter_weights_compat(model, adapter_dir: str, adapter_name: str):
    raw_state_dict, checkpoint_path = _load_local_adapter_state_dict(adapter_dir)
    remapped, source_to_full = _remap_adapter_state_dict_for_model(
        model=model,
        raw_state_dict=raw_state_dict,
        adapter_name=adapter_name,
    )

    load_result = set_peft_model_state_dict(
        model,
        remapped,
        adapter_name=adapter_name,
    )
    missing_adapter_keys = [
        key for key in getattr(load_result, "missing_keys", [])
        if "lora_" in key and f".{adapter_name}." in key
    ]
    unexpected_adapter_keys = [
        key for key in getattr(load_result, "unexpected_keys", [])
        if "lora_" in key
    ]
    if missing_adapter_keys or unexpected_adapter_keys:
        raise RuntimeError(
            f"Failed to load adapter {adapter_name!r}: "
            f"missing={missing_adapter_keys[:5]}, "
            f"unexpected={unexpected_adapter_keys[:5]}"
        )

    remapped_count = sum(
        _strip_lora_adapter_name(source_key)
        != _strip_lora_adapter_name(full_key)
        for source_key, full_key in source_to_full.items()
    )
    print(
        f"[Rank {rank()}] Loaded adapter {adapter_name!r} from {checkpoint_path} | "
        f"tensors={len(remapped)} | architecture_key_remaps={remapped_count}",
        flush=True,
    )


def load_lora_adapter_compat(
    base_model,
    adapter_dir: str,
    adapter_name: str = "default",
    is_trainable: bool = False,
):
    """Load one adapter with strict key validation; also used by 4_eval_rubric.py."""
    config = PeftConfig.from_pretrained(adapter_dir)
    config.inference_mode = not is_trainable

    try:
        model = get_peft_model(
            base_model,
            config,
            adapter_name=adapter_name,
        )
    except TypeError:
        if adapter_name != "default":
            raise
        model = get_peft_model(base_model, config)

    _load_adapter_weights_compat(
        model=model,
        adapter_dir=adapter_dir,
        adapter_name=adapter_name,
    )

    model.set_adapter(adapter_name)
    for name, parameter in model.named_parameters():
        if f".{adapter_name}." in name:
            parameter.requires_grad = bool(is_trainable)
    return model

# ============================================================
# Resume helpers
# ============================================================
_CHECKPOINT_NAME_RE = re.compile(r"^checkpoint-(\d+)$")
_RESUME_STATE_VERSION = 1


def _checkpoint_step(checkpoint_dir: str) -> int:
    match = _CHECKPOINT_NAME_RE.fullmatch(os.path.basename(os.path.normpath(checkpoint_dir)))
    if not match:
        raise ValueError(
            f"Resume checkpoint directory must be named checkpoint-N: {checkpoint_dir}"
        )
    return int(match.group(1))


def _is_valid_adapter_checkpoint(checkpoint_dir: str) -> bool:
    if not os.path.isdir(checkpoint_dir):
        return False
    config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    weight_candidates = (
        os.path.join(checkpoint_dir, "adapter_model.safetensors"),
        os.path.join(checkpoint_dir, "adapter_model.bin"),
    )
    return os.path.isfile(config_path) and any(
        os.path.isfile(path) for path in weight_candidates
    )


def find_latest_resume_checkpoint(output_dir: str = OUTPUT_DIR):
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        match = _CHECKPOINT_NAME_RE.fullmatch(name)
        if not match:
            continue
        checkpoint_dir = os.path.join(output_dir, name)
        if _is_valid_adapter_checkpoint(checkpoint_dir):
            candidates.append((int(match.group(1)), checkpoint_dir))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def resolve_resume_checkpoint(args: argparse.Namespace):
    if bool(getattr(args, "no_resume", False)):
        return None

    requested = str(getattr(args, "resume_from_checkpoint", "auto") or "auto").strip()
    if requested.lower() in {"", "auto", "latest"}:
        checkpoint_dir = find_latest_resume_checkpoint(OUTPUT_DIR)
        if checkpoint_dir and is_rank0():
            print(
                f"[Resume] Auto-selected latest checkpoint: {checkpoint_dir}",
                flush=True,
            )
        return checkpoint_dir
    if requested.lower() in {"none", "off", "false", "no"}:
        return None

    checkpoint_dir = os.path.realpath(os.path.abspath(requested))
    if not _is_valid_adapter_checkpoint(checkpoint_dir):
        raise RuntimeError(
            f"Invalid resume checkpoint or missing adapter files: {checkpoint_dir}"
        )
    if is_rank0():
        print(f"[Resume] Using explicit checkpoint: {checkpoint_dir}", flush=True)
    return checkpoint_dir


def _torch_load_compat(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _atomic_torch_save(payload, path: str):
    temp_path = f"{path}.tmp-r{rank()}-{os.getpid()}"
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def _move_optimizer_state_to_device(optimizer, device):
    device = torch.device(device)
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def _capture_rng_state() -> Dict[str, Any]:
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state_from_resume(state: Dict[str, Any]):
    if not state:
        return
    if state.get("python_random_state") is not None:
        random.setstate(state["python_random_state"])
    if state.get("numpy_random_state") is not None:
        np.random.set_state(state["numpy_random_state"])
    if state.get("torch_cpu_rng_state") is not None:
        torch.set_rng_state(state["torch_cpu_rng_state"])
    cuda_states = state.get("torch_cuda_rng_state_all")
    if cuda_states is not None and torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "Resume CUDA RNG state count does not match the current visible GPU count: "
                f"saved={len(cuda_states)} current={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(cuda_states)


def _legacy_resume_cursor(
    checkpoint_step: int,
    local_sample_count: int,
    local_prompts_per_update: int,
):
    updates_per_epoch = math.ceil(
        local_sample_count / max(local_prompts_per_update, 1)
    )
    if updates_per_epoch <= 0:
        return 0, 0
    start_epoch = checkpoint_step // updates_per_epoch
    step_in_epoch = checkpoint_step % updates_per_epoch
    prompt_offset = min(
        step_in_epoch * local_prompts_per_update,
        local_sample_count,
    )
    return int(start_epoch), int(prompt_offset)


def load_resume_training_state(
    checkpoint_dir: str,
    optimizer,
    device,
    local_sample_count: int,
    local_prompts_per_update: int,
    workers: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    default_state = {
        "global_step": 0,
        "start_epoch": 0,
        "start_prompt_offset": 0,
        "selected_generation_microbatch": None,
        "saved_logprob_microbatch": None,
        "rng_state": {},
        "optimizer_loaded": False,
        "legacy_checkpoint": False,
    }
    if not checkpoint_dir:
        return default_state

    checkpoint_step = _checkpoint_step(checkpoint_dir)
    rank_state_path = os.path.join(
        checkpoint_dir, f"training_state_rank{rank()}.pt"
    )
    shared_state_path = os.path.join(checkpoint_dir, "training_state.pt")
    metadata_path = (
        rank_state_path if os.path.isfile(rank_state_path)
        else shared_state_path if os.path.isfile(shared_state_path)
        else None
    )
    optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")

    metadata = None
    if metadata_path:
        metadata = _torch_load_compat(metadata_path, map_location="cpu")
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Malformed resume state: {metadata_path}")
        saved_revision = str(metadata.get("pipeline_revision") or "").strip()
        if saved_revision and saved_revision not in COMPATIBLE_PIPELINE_REVISIONS:
            raise RuntimeError(
                "Resume reward-pipeline mismatch: "
                f"saved={saved_revision}, current={PIPELINE_REVISION}. "
                "Use --no_resume or a new MBA_GRPO_OUTPUT_DIR so the new "
                "branch-specific reward objective starts from the SFT adapter."
            )
        saved_run_config = metadata.get("run_config") or {}
        saved_branch_type = str(saved_run_config.get("branch_type") or "").strip().lower()
        current_branch_type = str(args.branch_type).strip().lower()
        if saved_branch_type and saved_branch_type != current_branch_type:
            raise RuntimeError(
                "Resume branch mismatch: "
                f"saved={saved_branch_type}, current={current_branch_type}. "
                "Use --no_resume or a separate MBA_GRPO_OUTPUT_DIR."
            )
        saved_no_auxiliary = saved_run_config.get("no_auxiliary")
        if saved_no_auxiliary is not None and bool(saved_no_auxiliary) != bool(args.no_auxiliary):
            raise RuntimeError(
                "Resume auxiliary-setting mismatch: "
                f"saved_no_auxiliary={bool(saved_no_auxiliary)}, "
                f"current_no_auxiliary={bool(args.no_auxiliary)}."
            )
        saved_workers = int(metadata.get("world_size", workers))
        if saved_workers != workers:
            raise RuntimeError(
                f"Resume world-size mismatch: saved={saved_workers}, current={workers}"
            )
        saved_count = int(metadata.get("local_sample_count", local_sample_count))
        if saved_count != local_sample_count:
            raise RuntimeError(
                "Resume dataset partition size mismatch: "
                f"saved={saved_count}, current={local_sample_count}"
            )
        saved_local_prompts = int(
            metadata.get("local_prompts_per_update", local_prompts_per_update)
        )
        if saved_local_prompts != local_prompts_per_update:
            raise RuntimeError(
                "Resume local-prompts/update mismatch: "
                f"saved={saved_local_prompts}, current={local_prompts_per_update}"
            )

    optimizer_loaded = False
    if os.path.isfile(optimizer_path):
        optimizer_state = _torch_load_compat(optimizer_path, map_location="cpu")
        optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state_to_device(optimizer, device)
        optimizer_loaded = True
    elif is_rank0():
        print(
            f"[Resume] No optimizer.pt in {checkpoint_dir}; optimizer starts fresh.",
            flush=True,
        )

    if metadata is None:
        start_epoch, start_prompt_offset = _legacy_resume_cursor(
            checkpoint_step=checkpoint_step,
            local_sample_count=local_sample_count,
            local_prompts_per_update=local_prompts_per_update,
        )
        if is_rank0():
            print(
                "[Resume] Legacy adapter-only checkpoint detected. "
                f"Loaded policy at step={checkpoint_step}; inferred "
                f"epoch={start_epoch}, prompt_offset={start_prompt_offset}. "
                "Optimizer and exact RNG history are unavailable for this first resume.",
                flush=True,
            )
        return {
            **default_state,
            "global_step": checkpoint_step,
            "start_epoch": start_epoch,
            "start_prompt_offset": start_prompt_offset,
            "optimizer_loaded": optimizer_loaded,
            "legacy_checkpoint": True,
        }

    global_step = int(metadata.get("global_step", checkpoint_step))
    if global_step != checkpoint_step:
        raise RuntimeError(
            f"Checkpoint directory/state step mismatch: dir={checkpoint_step}, "
            f"state={global_step}"
        )
    start_epoch = int(metadata.get("next_epoch", 0))
    start_prompt_offset = int(metadata.get("next_prompt_offset", 0))
    if start_epoch < 0 or start_prompt_offset < 0:
        raise RuntimeError("Resume cursor must be non-negative.")
    if start_prompt_offset > local_sample_count:
        raise RuntimeError(
            f"Resume prompt offset exceeds local dataset: "
            f"offset={start_prompt_offset}, size={local_sample_count}"
        )

    if is_rank0():
        print(
            f"[Resume] Restored checkpoint step={global_step} | "
            f"next_epoch={start_epoch} | next_prompt_offset={start_prompt_offset} | "
            f"optimizer={'restored' if optimizer_loaded else 'fresh'}",
            flush=True,
        )

    return {
        **default_state,
        "global_step": global_step,
        "start_epoch": start_epoch,
        "start_prompt_offset": start_prompt_offset,
        "selected_generation_microbatch": metadata.get(
            "selected_generation_microbatch"
        ),
        "saved_logprob_microbatch": metadata.get("logprob_microbatch"),
        "rng_state": metadata.get("rng_state") or {},
        "optimizer_loaded": optimizer_loaded,
        "legacy_checkpoint": False,
    }


def save_resume_training_state(
    checkpoint_dir: str,
    optimizer,
    global_step: int,
    next_epoch: int,
    next_prompt_offset: int,
    local_sample_count: int,
    local_prompts_per_update: int,
    selected_generation_microbatch: int,
    logprob_microbatch: int,
    args: argparse.Namespace,
):
    # DDP workers have synchronized optimizer states, so rank 0 stores one shared
    # optimizer file while every rank stores its own RNG/cursor metadata.
    if is_rank0():
        _atomic_torch_save(
            optimizer.state_dict(),
            os.path.join(checkpoint_dir, "optimizer.pt"),
        )
    barrier()

    metadata = {
        "resume_state_version": _RESUME_STATE_VERSION,
        "pipeline_revision": PIPELINE_REVISION,
        "global_step": int(global_step),
        "next_epoch": int(next_epoch),
        "next_prompt_offset": int(next_prompt_offset),
        "local_sample_count": int(local_sample_count),
        "local_prompts_per_update": int(local_prompts_per_update),
        "world_size": int(world_size()),
        "rank": int(rank()),
        "selected_generation_microbatch": int(selected_generation_microbatch),
        "logprob_microbatch": int(logprob_microbatch),
        "rng_state": _capture_rng_state(),
        "run_config": {
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "num_generations": int(args.num_generations),
            "learning_rate": float(args.learning_rate),
            "beta_kl": float(args.beta_kl),
            "target_global_prompts_per_step": int(
                args.target_global_prompts_per_step
            ),
            "rollout_superbatch_size": int(args.rollout_superbatch_size),
            "train_prompt_minibatch_size": int(
                args.train_prompt_minibatch_size
            ),
            "creativity_reference_count": 5,
            "branch_type": str(args.branch_type),
            "no_auxiliary": bool(args.no_auxiliary),
            "factscore_precision": str(args.factscore_precision),
            "factscore_verifier_batch_size": int(
                args.factscore_verifier_batch_size
            ),
        },
    }
    _atomic_torch_save(
        metadata,
        os.path.join(checkpoint_dir, f"training_state_rank{rank()}.pt"),
    )
    barrier()


# ============================================================
# Policy model
# ============================================================
def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def _set_adapter_trainability(peft_model, active_adapter: str):
    """Activate the trainable policy LoRA while keeping the base model frozen."""
    peft_model.set_adapter(active_adapter)
    for name, param in peft_model.named_parameters():
        param.requires_grad = (
            "lora_" in name and f".{active_adapter}." in name
        )


def activate_policy_adapter(model):
    raw = unwrap_model(model)
    _set_adapter_trainability(raw, POLICY_ADAPTER_NAME)
    return raw


def activate_frozen_reference(reference_model):
    """Return the dedicated frozen SFT reference resident on its own GPU."""
    raw = unwrap_model(reference_model)
    raw.set_adapter(POLICY_ADAPTER_NAME)
    for parameter in raw.parameters():
        parameter.requires_grad = False
    raw.eval()
    return raw


def _set_policy_gradient_checkpointing(model, enabled: bool):
    raw = unwrap_model(model)
    if enabled:
        try:
            raw.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            raw.gradient_checkpointing_enable()
        raw.config.use_cache = False
    else:
        try:
            raw.gradient_checkpointing_disable()
        except Exception:
            pass
        raw.config.use_cache = False
    return bool(enabled)


def _configure_policy_processor_from_model(processor, model):
    try:
        vcfg = model.config.vision_config
        processor.image_processor.patch_size = vcfg.patch_size
        processor.image_processor.temporal_patch_size = vcfg.temporal_patch_size
        processor.image_processor.merge_size = vcfg.spatial_merge_size
    except Exception:
        pass

    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id


def load_policy_and_processor(
    policy_device,
    reference_device,
    gradient_checkpointing_mode: str = "auto",
    policy_adapter_dir: str = SFT_LORA_DIR,
):
    """Load the trainable policy and a separate frozen SFT KL reference.

    The policy is placed on the third GPU of each four-GPU pipeline.  A complete
    frozen Qwen2-VL-7B + SFT LoRA reference is placed on the fourth GPU.  This
    removes reference-forward activations from the policy GPU and allows the two
    independent forwards to overlap across devices before the KL tensors are
    copied back to the policy device.
    """
    print(f"[Rank {rank()}] Loading Processor for Qwen2-VL...")

    processor = Qwen2VLProcessor.from_pretrained(MODEL_ID, use_fast=False)
    processor.tokenizer.padding_side = "left"
    processor.image_processor.min_pixels = MIN_PIXELS
    processor.image_processor.max_pixels = MAX_PIXELS

    print(
        f"[Rank {rank()}] Loading trainable policy on {policy_device} from "
        f"{policy_adapter_dir}...",
        flush=True,
    )
    policy_base = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": int(policy_device.index)},
    )
    _configure_policy_processor_from_model(processor, policy_base)

    mode = str(gradient_checkpointing_mode).lower().strip()
    if mode not in {"auto", "on", "off"}:
        raise ValueError("policy_gradient_checkpointing must be auto, on, or off")
    total_gib = int(
        torch.cuda.get_device_properties(policy_device).total_memory // (1024 ** 3)
    )
    checkpointing_enabled = mode == "on" or (mode == "auto" and total_gib < 40)
    _set_policy_gradient_checkpointing(policy_base, checkpointing_enabled)
    print(
        f"[Rank {rank()}] Policy gradient checkpointing="
        f"{'on' if checkpointing_enabled else 'off'} | GPU={total_gib}GiB",
        flush=True,
    )

    policy_model = load_lora_adapter_compat(
        base_model=policy_base,
        adapter_dir=policy_adapter_dir,
        adapter_name=POLICY_ADAPTER_NAME,
        is_trainable=True,
    )
    activate_policy_adapter(policy_model)
    policy_model.train()
    try:
        policy_model.enable_input_require_grads()
    except Exception:
        pass

    print(
        f"[Rank {rank()}] Loading dedicated frozen SFT KL reference on "
        f"{reference_device}...",
        flush=True,
    )
    reference_base = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": int(reference_device.index)},
    )
    _configure_policy_processor_from_model(processor, reference_base)
    # The reference never generates and never backpropagates. Disable cache to
    # minimize its forward-memory footprint for long in-context prompts.
    _set_policy_gradient_checkpointing(reference_base, False)
    reference_model = load_lora_adapter_compat(
        base_model=reference_base,
        adapter_dir=SFT_LORA_DIR,
        adapter_name=POLICY_ADAPTER_NAME,
        is_trainable=False,
    )
    activate_frozen_reference(reference_model)

    policy_params = sum(
        parameter.numel()
        for parameter in policy_model.parameters()
        if parameter.requires_grad
    )
    reference_trainable = sum(
        parameter.numel()
        for parameter in reference_model.parameters()
        if parameter.requires_grad
    )
    if reference_trainable != 0:
        raise RuntimeError("Dedicated SFT reference model must be fully frozen.")

    torch.cuda.set_device(policy_device)
    print(
        f"[Rank {rank()}] SFT policy/reference ready | "
        f"policy_gpu={policy_device} | reference_gpu={reference_device} | "
        f"policy_trainable={policy_params:,} | "
        f"reference_trainable={reference_trainable}",
        flush=True,
    )
    return policy_model, reference_model, processor, checkpointing_enabled

# ============================================================
# Generation and batched completion log-probabilities
# ============================================================
@torch.no_grad()
def generate_prompt_batch_completions(
    model,
    processor,
    samples: List[Dict[str, Any]],
    num_generations: int,
    device,
    generation_microbatch: int = 0,
):
    """Generate G candidates for every prompt in one flattened multimodal batch.

    Repeating each prompt/image G times is more predictable for Qwen2-VL than relying
    on GenerationMixin to expand multimodal tensors for num_return_sequences.
    Returns (grouped_completions, selected_generation_microbatch).
    """
    raw_model = unwrap_model(model)
    records = []
    for prompt_index, sample in enumerate(samples):
        image = load_image(sample["image_path"])
        prompt_text = make_prompt_text(
            processor, sample["image_path"], sample["question"]
        )
        for generation_index in range(num_generations):
            records.append({
                "prompt_index": prompt_index,
                "generation_index": generation_index,
                "image": image,
                "prompt_text": prompt_text,
            })

    total_records = len(records)
    selected = total_records if generation_microbatch <= 0 else min(
        int(generation_microbatch), total_records
    )
    selected = max(1, selected)
    was_training = raw_model.training
    raw_model.eval()
    old_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"

    while True:
        try:
            flat_outputs: List[str] = []
            for offset in range(0, total_records, selected):
                chunk = records[offset: offset + selected]
                inputs = processor(
                    text=[record["prompt_text"] for record in chunk],
                    images=[record["image"] for record in chunk],
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                input_width = inputs["input_ids"].shape[1]
                generated = raw_model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=1.05,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.05,
                    num_return_sequences=1,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                )
                flat_outputs.extend([
                    processor.tokenizer.decode(
                        row[input_width:], skip_special_tokens=True
                    ).strip()
                    for row in generated
                ])
                del inputs, generated
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if selected <= 1:
                processor.tokenizer.padding_side = old_padding_side
                if was_training:
                    raw_model.train()
                raise
            next_selected = max(1, selected // 2)
            print(
                f"[Rank {rank()}] Generation batch OOM; retrying "
                f"microbatch={next_selected} (was {selected}).",
                flush=True,
            )
            selected = next_selected

    processor.tokenizer.padding_side = old_padding_side
    if was_training:
        raw_model.train()

    grouped = [[] for _ in samples]
    for record, completion in zip(records, flat_outputs):
        grouped[record["prompt_index"]].append(completion)
    # if is_rank0():
    #     for prompt_index, completions in enumerate(grouped):
    #         print(
    #             f"[Generation Diversity p{prompt_index}] "
    #             f"unique={len(set(c for c in completions if c))}/{len(completions)} "
    #             f"| first={completions[0][:80]!r}",
    #             flush=True,
    #         )
    return grouped, selected


@torch.no_grad()
def generate_batched_completions(
    model, processor, image_path: str, question: str, num_generations: int, device
):
    grouped, _ = generate_prompt_batch_completions(
        model,
        processor,
        [{"image_path": image_path, "question": question}],
        num_generations,
        device,
        generation_microbatch=num_generations,
    )
    return grouped[0]


def compute_multi_prompt_completion_token_logprobs(
    model,
    processor,
    records: List[Dict[str, str]],
    device,
):
    """Return log-probs/masks for completions belonging to different image prompts."""
    images = [load_image(record["image_path"]) for record in records]
    prompt_texts = [
        make_prompt_text(processor, record["image_path"], record["question"])
        for record in records
    ]
    eos = processor.tokenizer.eos_token or ""

    # Qwen2VLProcessor.__call__ mutates the supplied text list while expanding
    # <|image_pad|> placeholders to the visual-token length.  Therefore the same
    # list must never be reused for a second processor call.  Construct the full
    # texts before either call and pass fresh list objects to both invocations.
    full_texts = [
        prompt_text + record["completion"] + eos
        for prompt_text, record in zip(prompt_texts, records)
    ]

    old_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    try:
        prompt_inputs = processor(
            text=list(prompt_texts),
            images=list(images),
            padding=True,
            return_tensors="pt",
        ).to(device)
        full_inputs = processor(
            text=list(full_texts),
            images=list(images),
            padding=True,
            return_tensors="pt",
        ).to(device)
    finally:
        processor.tokenizer.padding_side = old_padding_side

    prompt_lens = prompt_inputs["attention_mask"].sum(dim=1)
    full_lens = full_inputs["attention_mask"].sum(dim=1)
    outputs = model(**full_inputs)
    logits = outputs.logits[:, :-1, :]
    labels = full_inputs["input_ids"][:, 1:]
    token_log_probs = torch.log_softmax(logits, dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)

    # Build the completion mask entirely on-device. Avoiding per-row .item()
    # calls prevents an implicit CUDA synchronization, which lets the frozen
    # reference forward on GPU 4 and the policy forward on GPU 3 overlap.
    padded_width = full_inputs["input_ids"].shape[1]
    first_token = padded_width - full_lens
    first_completion_label = (first_token + prompt_lens - 1).clamp_min(0)
    label_positions = torch.arange(
        labels.shape[1], device=labels.device
    ).unsqueeze(0)
    mask = label_positions >= first_completion_label.unsqueeze(1)
    mask &= full_inputs["attention_mask"][:, 1:].bool()
    return token_log_probs, mask


def compute_batched_completion_token_logprobs(
    model,
    processor,
    image_path: str,
    question: str,
    completions: List[str],
    device,
):
    records = [{
        "image_path": image_path,
        "question": question,
        "completion": completion,
    } for completion in completions]
    return compute_multi_prompt_completion_token_logprobs(
        model, processor, records, device
    )


def masked_sequence_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denominator = mask.sum(dim=1).clamp_min(1)
    return (values * mask).sum(dim=1) / denominator


def _make_probe_completion(processor, target_tokens: int = MAX_NEW_TOKENS) -> str:
    """Build a synthetic max-length completion for the log-prob memory probe.

    This probe is intentionally independent of expert/GT ideas; it is used only to
    estimate memory for policy/reference log-prob computation.
    """
    seed_text = json.dumps({
        "title": "Synthetic image-grounded business concept",
        "description": "A concrete B2B offering for a defined customer need and measurable benefit.",
        "implementation": "Deploy current software, AI, data integration, and an operational workflow.",
        "differentiation": "Use a specialized workflow and proprietary operational data rather than a generic platform.",
    }, ensure_ascii=False)
    token_ids = processor.tokenizer(
        seed_text, add_special_tokens=False
    )["input_ids"]
    if not token_ids:
        token_ids = [processor.tokenizer.eos_token_id]
    repeated = []
    while len(repeated) < target_tokens:
        repeated.extend(token_ids)
    return processor.tokenizer.decode(
        repeated[:target_tokens], skip_special_tokens=True
    )


def _empty_cuda_caches(*devices):
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def autotune_logprob_microbatch(
    model,
    reference_model,
    processor,
    sample,
    device,
    reference_device,
    requested: int,
    num_generations: int,
) -> int:
    """Select the largest policy/reference log-prob microbatch that fits.

    The frozen reference forward runs entirely on the fourth pipeline GPU.  Its
    kernels are enqueued before the policy forward, so independent GPU work can
    overlap.  Only the compact completion-token log-probabilities and mask are
    copied to the policy GPU for KL computation.
    """
    total_gib = int(
        torch.cuda.get_device_properties(device).total_memory // (1024 ** 3)
    )
    if requested > 0:
        first = min(int(requested), int(num_generations))
        candidates = list(range(first, 0, -1))
    elif total_gib >= 40:
        candidates = [min(4, num_generations), min(2, num_generations), 1]
    elif total_gib >= 22:
        candidates = [min(2, num_generations), 1]
    else:
        candidates = [1]
    candidates = list(dict.fromkeys(max(1, int(x)) for x in candidates))

    probe_completion = _make_probe_completion(processor)
    image_path = sample["image_path"]
    question = sample["question"]
    raw_model = unwrap_model(model)
    frozen_reference = activate_frozen_reference(reference_model)

    for candidate in candidates:
        try:
            _empty_cuda_caches(device, reference_device)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.reset_peak_memory_stats(reference_device)
            model.zero_grad(set_to_none=True)
            completions = [probe_completion] * candidate

            # Queue the independent frozen-reference forward first.
            with torch.no_grad(), torch.cuda.device(reference_device):
                ref_logps, ref_mask = compute_batched_completion_token_logprobs(
                    frozen_reference,
                    processor,
                    image_path,
                    question,
                    completions,
                    reference_device,
                )

            activate_policy_adapter(model)
            raw_model.train()
            sync_context = (
                model.no_sync()
                if isinstance(model, DDP)
                else contextlib.nullcontext()
            )
            with sync_context, torch.cuda.device(device):
                policy_logps, policy_mask = compute_batched_completion_token_logprobs(
                    model,
                    processor,
                    image_path,
                    question,
                    completions,
                    device,
                )
                # Force completion of the reference work and validate the
                # cross-device transfer used by the real KL path.
                ref_logps_policy = ref_logps.to(device, non_blocking=True)
                ref_mask_policy = ref_mask.to(device, non_blocking=True)
                common_mask = policy_mask & ref_mask_policy
                probe_loss = (
                    policy_logps.masked_select(common_mask).sum() * 0.0
                    + ref_logps_policy.masked_select(common_mask).sum() * 0.0
                )
                probe_loss.backward()

            torch.cuda.synchronize(device)
            torch.cuda.synchronize(reference_device)
            policy_peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            reference_peak_gib = (
                torch.cuda.max_memory_allocated(reference_device) / (1024 ** 3)
            )
            model.zero_grad(set_to_none=True)
            del (
                ref_logps,
                ref_mask,
                ref_logps_policy,
                ref_mask_policy,
                policy_logps,
                policy_mask,
                common_mask,
                probe_loss,
            )
            _empty_cuda_caches(device, reference_device)
            activate_policy_adapter(model)
            raw_model.train()
            print(
                f"[Rank {rank()}] logprob microbatch probe: selected={candidate} | "
                f"policy_peak={policy_peak_gib:.2f} GiB/{total_gib} GiB | "
                f"reference_peak={reference_peak_gib:.2f} GiB | "
                f"reference_gpu={reference_device}",
                flush=True,
            )
            return candidate
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            _empty_cuda_caches(device, reference_device)
            activate_policy_adapter(model)
            raw_model.train()
            print(
                f"[Rank {rank()}] logprob microbatch={candidate} OOM during "
                "policy/reference probe; trying a smaller value.",
                flush=True,
            )

    raise RuntimeError("Even logprob_microbatch=1 failed the startup memory probe.")

# ============================================================
# Training
# ============================================================
def train_custom_grpo(args: argparse.Namespace):
    policy_gpu, workers = setup_distributed()
    layout = get_runtime_layout()
    device = torch.device(f"cuda:{policy_gpu}")
    reference_device = torch.device(f"cuda:{layout['reference_gpu']}")
    if device == reference_device:
        raise RuntimeError(
            f"Policy and KL reference must use different GPUs: {device}"
        )

    random.seed(args.seed + rank())
    np.random.seed(args.seed + rank())
    torch.manual_seed(args.seed + rank())
    torch.cuda.manual_seed_all(args.seed + rank())
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print(
        f"[Rank {rank()} GPU layout] pipeline={layout['pipeline_gpus']} | "
        f"Judge={layout['judge_gpus']} | Policy=cuda:{policy_gpu} | "
        f"SFT-KL-Reference={reference_device}",
        flush=True,
    )
    resume_checkpoint = resolve_resume_checkpoint(args)
    policy_adapter_dir = resume_checkpoint or SFT_LORA_DIR
    model, reference_model, processor, checkpointing_enabled = load_policy_and_processor(
        device,
        reference_device,
        args.policy_gradient_checkpointing,
        policy_adapter_dir=policy_adapter_dir,
    )
    if workers > 1:
        model = DDP(
            model,
            device_ids=[policy_gpu],
            output_device=policy_gpu,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    local_prompts_per_update = max(
        1, math.ceil(args.target_global_prompts_per_step / workers)
    )
    rollout_superbatch_size = max(1, int(args.rollout_superbatch_size))
    if args.prompt_batch_size is not None:
        rollout_superbatch_size = max(1, int(args.prompt_batch_size))
    train_prompt_minibatch_size = (
        local_prompts_per_update
        if int(args.train_prompt_minibatch_size) <= 0
        else int(args.train_prompt_minibatch_size)
    )
    if train_prompt_minibatch_size > local_prompts_per_update:
        raise ValueError(
            f"train_prompt_minibatch_size={train_prompt_minibatch_size} exceeds "
            f"local prompts/update={local_prompts_per_update}."
        )
    if local_prompts_per_update % train_prompt_minibatch_size != 0:
        raise ValueError(
            f"local prompts/update={local_prompts_per_update} must be divisible by "
            f"train_prompt_minibatch_size={train_prompt_minibatch_size}."
        )
    if rollout_superbatch_size < train_prompt_minibatch_size:
        raise ValueError(
            "rollout_superbatch_size must be >= train_prompt_minibatch_size"
        )
    if rollout_superbatch_size % train_prompt_minibatch_size != 0:
        raise ValueError(
            "rollout_superbatch_size must be divisible by train_prompt_minibatch_size"
        )

    reward_manager = RewardModelManager(
        judge_axis_batch_size=args.judge_axis_batch_size,
        factscore_verifier_batch_size=args.factscore_verifier_batch_size,
        rollout_superbatch_size=rollout_superbatch_size,
        factscore_precision=args.factscore_precision,
        judge_precision=args.judge_precision,
        use_auxiliary=not args.no_auxiliary,
        branch_type=args.branch_type,
    )
    if reward_manager.use_auxiliary and not args.disable_async_overlap:
        args.disable_async_overlap = True
        if is_rank0():
            print(
                "[Auxiliary Safety] --disable_async_overlap enabled automatically: "
                "FActScore and the frozen KL reference share the fourth pipeline GPU.",
                flush=True,
            )
    torch.cuda.set_device(policy_gpu)

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    try:
        optimizer = torch.optim.AdamW(
            trainable_params, lr=args.learning_rate, fused=True
        )
    except TypeError:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    all_samples = extract_all_samples(DATA_DIR, IMAGE_DIR)
    validate_samples(all_samples)
    train_samples, test_samples, local_samples = split_dataset(
        all_samples, workers, rank(), seed=args.seed
    )

    if is_rank0():
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        train_image_count = len({
            canonical_image_key(sample["image_path"]) for sample in train_samples
        })
        test_image_count = len({
            canonical_image_key(sample["image_path"]) for sample in test_samples
        })
        print(
            f"Samples: total={len(all_samples)} | Train={len(train_samples)} | "
            f"Test={len(test_samples)} | workers={workers} | "
            f"samples/worker={len(local_samples)}\n"
            f"Images: Train={train_image_count} | Test={test_image_count} | "
            f"split={DOMAIN_STRATIFIED_SPLIT_REVISION} | manifest={TEST_FILE}"
        )
        for domain in EXPECTED_DOMAINS:
            train_domain_images = len({
                canonical_image_key(sample["image_path"])
                for sample in train_samples if sample["domain"] == domain
            })
            test_domain_images = len({
                canonical_image_key(sample["image_path"])
                for sample in test_samples if sample["domain"] == domain
            })
            total_domain_images = train_domain_images + test_domain_images
            print(
                f"  [Domain Split] {domain}: images "
                f"{train_domain_images} train / {test_domain_images} test "
                f"({test_domain_images / max(total_domain_images, 1):.4f} test) | "
                f"samples {sum(1 for x in train_samples if x['domain'] == domain)} train / "
                f"{sum(1 for x in test_samples if x['domain'] == domain)} test"
            )
    barrier()

    if not local_samples:
        raise RuntimeError(f"Rank {rank()} received no training samples.")

    resume_state = load_resume_training_state(
        checkpoint_dir=resume_checkpoint,
        optimizer=optimizer,
        device=device,
        local_sample_count=len(local_samples),
        local_prompts_per_update=local_prompts_per_update,
        workers=workers,
        args=args,
    )

    try:
        logprob_microbatch = autotune_logprob_microbatch(
            model=model,
            reference_model=reference_model,
            processor=processor,
            sample=local_samples[0],
            device=device,
            reference_device=reference_device,
            requested=args.logprob_microbatch,
            num_generations=args.num_generations,
        )
    except RuntimeError as exc:
        if checkpointing_enabled or "Even logprob_microbatch=1 failed" not in str(exc):
            raise
        print(
            f"[Rank {rank()}] Policy probe failed without checkpointing; "
            "enabling gradient checkpointing and retrying.",
            flush=True,
        )
        checkpointing_enabled = _set_policy_gradient_checkpointing(model, True)
        logprob_microbatch = autotune_logprob_microbatch(
            model=model,
            reference_model=reference_model,
            processor=processor,
            sample=local_samples[0],
            device=device,
            reference_device=reference_device,
            requested=args.logprob_microbatch,
            num_generations=args.num_generations,
        )

    updates_per_superbatch = math.ceil(
        rollout_superbatch_size / max(local_prompts_per_update, 1)
    )
    if is_rank0():
        print(
            f"target_global_prompts/update={args.target_global_prompts_per_step} | "
            f"local_prompts/update={local_prompts_per_update} | "
            f"rollout_superbatch_size={rollout_superbatch_size} | "
            f"train_prompt_minibatch_size={train_prompt_minibatch_size} | "
            f"updates/superbatch={updates_per_superbatch} | "
            f"logprob_microbatch={logprob_microbatch} | "
            f"policy_gradient_checkpointing={'on' if checkpointing_enabled else 'off'} | "
            f"reference_gpu={reference_device} | reference_precision=bf16 | "
            f"judge_precision={reward_manager.judge_precision} | "
            f"judge_request_batch={reward_manager.judge_axis_batch_size} | "
            f"branch_type={reward_manager.branch_type} | "
            f"judge_dimension_prompts/group={len(PBIG_DIMENSION_ORDER) if reward_manager.use_pbig else 0} | "
            f"creativity_prompts/group=1 | creativity_GT_refs/group=5 | "
            f"feasibility={'enabled' if reward_manager.use_auxiliary else 'disabled'} | "
            f"async_overlap={not args.disable_async_overlap} | "
            f"max_policy_lag<={updates_per_superbatch} optimizer steps"
        )

    global_step = int(resume_state["global_step"])
    start_epoch = int(resume_state["start_epoch"])
    start_prompt_offset = int(resume_state["start_prompt_offset"])
    accumulated_prompts = 0
    current_window_target = local_prompts_per_update
    optimizer.zero_grad(set_to_none=True)
    model.train()
    rollout_reward_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix=f"rollout-reward-r{rank()}"
    )
    selected_generation_microbatch = resume_state.get(
        "selected_generation_microbatch"
    )
    if selected_generation_microbatch is None:
        selected_generation_microbatch = int(args.generation_microbatch)
    else:
        selected_generation_microbatch = int(selected_generation_microbatch)

    def make_superbatches(samples):
        return [
            samples[offset: offset + rollout_superbatch_size]
            for offset in range(0, len(samples), rollout_superbatch_size)
        ]

    def reset_pipeline_peak_memory():
        for gpu_id in layout["pipeline_gpus"]:
            try:
                torch.cuda.reset_peak_memory_stats(gpu_id)
            except Exception:
                pass

    def pipeline_peak_memory_gib():
        return {
            int(gpu_id): torch.cuda.max_memory_allocated(gpu_id) / (1024 ** 3)
            for gpu_id in layout["pipeline_gpus"]
        }

    def build_rollout(sample_batch):
        nonlocal selected_generation_microbatch
        started = time.perf_counter()
        activate_policy_adapter(model)
        grouped, selected = generate_prompt_batch_completions(
            model=model,
            processor=processor,
            samples=sample_batch,
            num_generations=args.num_generations,
            device=device,
            generation_microbatch=selected_generation_microbatch,
        )
        selected_generation_microbatch = selected
        return {
            "samples": sample_batch,
            "completions": grouped,
            "generation_seconds": time.perf_counter() - started,
        }

    def reward_task(rollout):
        started = time.perf_counter()
        items = []
        for sample, completions in zip(
            rollout["samples"], rollout["completions"]
        ):
            items.append({
                "image_path": sample["image_path"],
                "topic": sample["question"],
                "generations": completions,
                "creativity_reference_ideas": sample["creativity_reference_ideas"],
            })
        results = reward_manager.total_reward_multi_batched(items)
        return {
            "results": results,
            "reward_seconds": time.perf_counter() - started,
        }

    def prepare_prompt_payloads(rollout, reward_results):
        payloads = []
        for sample, completions, reward_result in zip(
            rollout["samples"], rollout["completions"], reward_results
        ):
            rewards = list(reward_result["rewards"])
            rewards_tensor = torch.tensor(
                rewards, dtype=torch.float32, device=device
            )
            if rewards_tensor.numel() > 1:
                advantages = rewards_tensor - rewards_tensor.mean()
                reward_std = rewards_tensor.std(unbiased=False)
                if reward_std.item() >= args.min_reward_std:
                    advantages = advantages / (reward_std + 1e-6)
                else:
                    advantages = torch.zeros_like(rewards_tensor)
                    if is_rank0():
                        tqdm.write(
                            f"[Low GRPO Signal] std={reward_std.item():.6f}; "
                            "advantages set to zero."
                        )
            else:
                advantages = torch.zeros_like(rewards_tensor)
            records = []
            for completion in completions:
                records.append({
                    "image_path": sample["image_path"],
                    "question": sample["question"],
                    "completion": completion,
                })
            payloads.append({
                "records": records,
                "advantages": [float(x) for x in advantages.detach().cpu().tolist()],
                "rewards": rewards,
                "breakdown": dict(reward_result["breakdown"]),
            })
        return payloads

    ema_superbatch = None
    ema_per_prompt = None
    ema_generation = None
    ema_reward = None
    ema_train = None
    smoothing = 0.10

    def update_ema(old, value):
        return float(value) if old is None else (
            (1.0 - smoothing) * old + smoothing * float(value)
        )

    restore_rng_state_from_resume(resume_state.get("rng_state") or {})
    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_start_offset = (
                start_prompt_offset if epoch == start_epoch else 0
            )
            if epoch_start_offset >= len(local_samples):
                continue
            epoch_samples = local_samples[epoch_start_offset:]
            superbatches = make_superbatches(epoch_samples)
            processed_prompts = epoch_start_offset

            iterator = tqdm(
                total=len(local_samples),
                initial=epoch_start_offset,
                desc=f"Rank0 Epoch {epoch + 1}/{args.epochs}",
                dynamic_ncols=True,
                mininterval=1.0,
                smoothing=0.1,
                disable=not is_rank0(),
            )

            current_rollout = build_rollout(superbatches[0])
            current_reward_future = rollout_reward_executor.submit(
                reward_task, current_rollout
            )

            for superbatch_index, sample_superbatch in enumerate(superbatches):
                superbatch_started = time.perf_counter()
                reset_pipeline_peak_memory()
                superbatch_prompt_count = len(sample_superbatch)

                next_rollout = None
                next_reward_future = None
                if (
                    superbatch_index + 1 < len(superbatches)
                    and not args.disable_async_overlap
                ):
                    next_rollout = build_rollout(superbatches[superbatch_index + 1])

                reward_payload = current_reward_future.result()
                reward_results = reward_payload["results"]
                if len(reward_results) != superbatch_prompt_count:
                    raise RuntimeError("Reward superbatch size mismatch")

                if next_rollout is not None:
                    next_reward_future = rollout_reward_executor.submit(
                        reward_task, next_rollout
                    )

                prompt_payloads = prepare_prompt_payloads(
                    current_rollout, reward_results
                )
                train_started = time.perf_counter()
                superbatch_losses = []
                superbatch_rewards = []
                superbatch_kls = []
                superbatch_breakdowns = []

                for minibatch_offset in range(
                    0, superbatch_prompt_count, train_prompt_minibatch_size
                ):
                    minibatch_payloads = prompt_payloads[
                        minibatch_offset:
                        minibatch_offset + train_prompt_minibatch_size
                    ]
                    minibatch_prompt_count = len(minibatch_payloads)
                    if accumulated_prompts == 0:
                        remaining_prompts = len(local_samples) - (
                            processed_prompts + minibatch_offset
                        )
                        current_window_target = min(
                            local_prompts_per_update, remaining_prompts
                        )
                    will_step = (
                        accumulated_prompts + minibatch_prompt_count
                    ) >= current_window_target

                    flat_records = []
                    flat_advantages = []
                    minibatch_rewards = []
                    minibatch_breakdowns = []
                    for payload in minibatch_payloads:
                        flat_records.extend(payload["records"])
                        flat_advantages.extend(payload["advantages"])
                        minibatch_rewards.extend(payload["rewards"])
                        minibatch_breakdowns.append(payload["breakdown"])

                    advantages_tensor = torch.tensor(
                        flat_advantages, dtype=torch.float32, device=device
                    )
                    chunks = [
                        list(range(start, min(start + logprob_microbatch, len(flat_records))))
                        for start in range(0, len(flat_records), logprob_microbatch)
                    ]
                    prompt_scale = minibatch_prompt_count / max(current_window_target, 1)
                    minibatch_loss_value = 0.0
                    minibatch_kl_values = []
                    for chunk_number, indices in enumerate(chunks):
                        chunk_records = [flat_records[index] for index in indices]
                        # Queue the frozen-reference forward on the fourth GPU.
                        # The subsequent policy forward is enqueued on a different
                        # GPU, allowing the independent work to overlap.
                        with torch.no_grad(), torch.cuda.device(reference_device):
                            ref_logps_reference, ref_mask_reference = (
                                compute_multi_prompt_completion_token_logprobs(
                                    reference_model,
                                    processor,
                                    chunk_records,
                                    reference_device,
                                )
                            )

                        activate_policy_adapter(model)
                        sync_this_backward = (
                            will_step and chunk_number == len(chunks) - 1
                        )
                        sync_context = (
                            contextlib.nullcontext()
                            if not isinstance(model, DDP) or sync_this_backward
                            else model.no_sync()
                        )
                        with sync_context:
                            policy_logps, policy_mask = (
                                compute_multi_prompt_completion_token_logprobs(
                                    model, processor, chunk_records, device
                                )
                            )
                            ref_logps = ref_logps_reference.to(
                                device, non_blocking=True
                            )
                            ref_mask = ref_mask_reference.to(
                                device, non_blocking=True
                            )
                            common_mask = policy_mask & ref_mask
                            policy_seq_logp = masked_sequence_mean(
                                policy_logps, common_mask
                            )
                            log_ratio = ref_logps.detach() - policy_logps
                            per_token_kl = torch.exp(log_ratio) - log_ratio - 1.0
                            kl_per_sequence = masked_sequence_mean(
                                per_token_kl, common_mask
                            )
                            chunk_advantages = advantages_tensor[indices].detach()
                            policy_loss = -(
                                chunk_advantages * policy_seq_logp
                            ).mean()
                            kl_loss = kl_per_sequence.mean()
                            chunk_weight = len(indices) / max(len(flat_records), 1)
                            loss = prompt_scale * chunk_weight * (
                                policy_loss + args.beta_kl * kl_loss
                            )
                            loss.backward()
                        minibatch_loss_value += float(loss.detach().item())
                        minibatch_kl_values.extend(
                            float(value)
                            for value in kl_per_sequence.detach().cpu().tolist()
                        )
                        del (
                            ref_logps_reference,
                            ref_mask_reference,
                            ref_logps,
                            ref_mask,
                            policy_logps,
                            policy_mask,
                            common_mask,
                            policy_seq_logp,
                            log_ratio,
                            per_token_kl,
                            kl_per_sequence,
                            policy_loss,
                            kl_loss,
                            loss,
                        )

                    accumulated_prompts += minibatch_prompt_count
                    if will_step:
                        torch.nn.utils.clip_grad_norm_(
                            trainable_params, args.max_grad_norm
                        )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        accumulated_prompts = 0

                        if is_rank0() and global_step % args.logging_steps == 0:
                            tqdm.write(
                                f"[Step {global_step}] loss={minibatch_loss_value:.6f} "
                                f"reward={sum(minibatch_rewards)/max(len(minibatch_rewards),1):.4f} "
                                f"kl_sft={sum(minibatch_kl_values)/max(len(minibatch_kl_values),1):.6f}"
                            )
                        if args.save_steps > 0 and global_step % args.save_steps == 0:
                            completed_in_epoch = (
                                processed_prompts
                                + minibatch_offset
                                + minibatch_prompt_count
                            )
                            if completed_in_epoch >= len(local_samples):
                                next_epoch = epoch + 1
                                next_prompt_offset = 0
                            else:
                                next_epoch = epoch
                                next_prompt_offset = completed_in_epoch

                            barrier()
                            checkpoint_dir = os.path.join(
                                OUTPUT_DIR, f"checkpoint-{global_step}"
                            )
                            if is_rank0():
                                os.makedirs(checkpoint_dir, exist_ok=True)
                                activate_policy_adapter(model).save_pretrained(
                                    checkpoint_dir,
                                    selected_adapters=[POLICY_ADAPTER_NAME],
                                )
                                processor.save_pretrained(checkpoint_dir)
                            barrier()
                            save_resume_training_state(
                                checkpoint_dir=checkpoint_dir,
                                optimizer=optimizer,
                                global_step=global_step,
                                next_epoch=next_epoch,
                                next_prompt_offset=next_prompt_offset,
                                local_sample_count=len(local_samples),
                                local_prompts_per_update=local_prompts_per_update,
                                selected_generation_microbatch=(
                                    selected_generation_microbatch
                                ),
                                logprob_microbatch=logprob_microbatch,
                                args=args,
                            )
                            if is_rank0():
                                tqdm.write(
                                    f"Saved resumable checkpoint -> {checkpoint_dir} | "
                                    f"next_epoch={next_epoch} | "
                                    f"next_prompt_offset={next_prompt_offset}"
                                )

                    superbatch_losses.append(minibatch_loss_value)
                    superbatch_rewards.extend(minibatch_rewards)
                    superbatch_kls.extend(minibatch_kl_values)
                    superbatch_breakdowns.extend(minibatch_breakdowns)

                train_seconds = time.perf_counter() - train_started
                mean_reward = all_reduce_mean_scalar(
                    sum(superbatch_rewards) / max(len(superbatch_rewards), 1), device
                )
                mean_kl = all_reduce_mean_scalar(
                    sum(superbatch_kls) / max(len(superbatch_kls), 1), device
                )
                mean_breakdown = {
                    key: sum(float(value.get(key, 0.0)) for value in superbatch_breakdowns)
                    / max(len(superbatch_breakdowns), 1)
                    for key in ("vlm", "market", "six_metric", "creativity", "feasibility")
                }
                total_loss_value = sum(superbatch_losses)

                superbatch_seconds = time.perf_counter() - superbatch_started
                per_prompt_seconds = superbatch_seconds / max(superbatch_prompt_count, 1)
                ema_superbatch = update_ema(ema_superbatch, superbatch_seconds)
                ema_per_prompt = update_ema(ema_per_prompt, per_prompt_seconds)
                ema_generation = update_ema(
                    ema_generation, current_rollout["generation_seconds"]
                )
                ema_reward = update_ema(
                    ema_reward, reward_payload["reward_seconds"]
                )
                ema_train = update_ema(ema_train, train_seconds)
                processed_prompts += superbatch_prompt_count

                if is_rank0():
                    iterator.set_postfix({
                        "Loss": f"{total_loss_value:.4f}",
                        "Rwd": f"{mean_reward:.4f}",
                        "VLM6": f"{mean_breakdown['six_metric']:.2f}" if reward_manager.use_pbig else "skip",
                        "Cr": f"{mean_breakdown['creativity']:.2f}",
                        "Feas": f"{mean_breakdown['feasibility']:.2f}" if reward_manager.use_auxiliary else "off",
                        "KL": f"{mean_kl:.4f}",
                        "SBatch": superbatch_prompt_count,
                        "EMA/p": f"{ema_per_prompt:.1f}s",
                    })
                    iterator.update(superbatch_prompt_count)

                if superbatch_index + 1 < len(superbatches):
                    if args.disable_async_overlap:
                        next_rollout = build_rollout(
                            superbatches[superbatch_index + 1]
                        )
                        next_reward_future = rollout_reward_executor.submit(
                            reward_task, next_rollout
                        )
                    current_rollout = next_rollout
                    current_reward_future = next_reward_future

            iterator.close()
    finally:
        rollout_reward_executor.shutdown(wait=True, cancel_futures=False)
        reward_manager.close()

    barrier()
    if is_rank0():
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        activate_policy_adapter(model).save_pretrained(
            OUTPUT_DIR, selected_adapters=[POLICY_ADAPTER_NAME]
        )
        processor.save_pretrained(OUTPUT_DIR)
        print(f"Saved final GRPO adapter -> {OUTPUT_DIR}")
    barrier()

# ============================================================
# Main / internal launcher
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--num_generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--beta_kl", type=float, default=BETA_KL)
    parser.add_argument(
        "--target_global_prompts_per_step",
        type=int,
        default=TARGET_GLOBAL_PROMPTS_PER_STEP,
    )
    parser.add_argument(
        "--rollout_superbatch_size", type=int, default=4,
        help="Prompts scored together per worker; A6000 aggressive default=4",
    )
    parser.add_argument(
        "--train_prompt_minibatch_size", type=int, default=0,
        help="0=local prompts/update (2 on 8 GPUs, 4 on 4 GPUs)",
    )
    parser.add_argument(
        "--prompt_batch_size", type=int, default=None,
        help="Deprecated alias for --rollout_superbatch_size",
    )
    parser.add_argument(
        "--generation_microbatch",
        type=int,
        default=0,
        help="0=all rollout_superbatch_size*num_generations rows; halves on OOM",
    )
    parser.add_argument("--max_grad_norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument("--save_steps", type=int, default=SAVE_STEPS)
    parser.add_argument("--logging_steps", type=int, default=LOGGING_STEPS)
    parser.add_argument(
        "--min_reward_std",
        type=float,
        default=MIN_REWARD_STD_FOR_NORMALIZATION,
    )
    parser.add_argument(
        "--judge_axis_batch_size",
        type=int,
        default=6,
        help=("Independent Judge request batch. Open runs six PBIG prompts plus one "
              "creativity prompt per group; closed runs creativity only."),
    )
    parser.add_argument(
        "--factscore_verifier_batch_size",
        type=int,
        default=0,
        help="0=auto; verifier batch size for the auxiliary FActScore component",
    )
    parser.add_argument(
        "--factscore_precision",
        choices=["auto", "4bit", "bf16"],
        default="auto",
        help=("FActScore precision on the frozen-reference GPU; auto selects 4bit "
              "because both models remain resident on an A6000"),
    )
    parser.add_argument(
        "--branch_type",
        choices=["open", "closed"],
        default="open",
        help=("open: PBIG-6 + creativity + feasibility; closed: creativity + "
              "feasibility only. Eval6M remains PBIG-6 for both branches."),
    )
    parser.add_argument(
        "--no_auxiliary",
        action="store_true",
        help=("Open branch only: disable feasibility while retaining PBIG-6 and "
              "the separate creativity reward."),
    )
    parser.add_argument(
        "--judge_precision",
        choices=["auto", "4bit", "8bit"],
        default="4bit",
        help="Judge precision; production A6000 default=4bit",
    )
    parser.add_argument(
        "--policy_gradient_checkpointing",
        choices=["auto", "on", "off"],
        default="off",
        help="Policy gradient checkpointing; production A6000 default=off",
    )
    parser.add_argument(
        "--logprob_microbatch",
        type=int,
        default=2,
        help="Policy/reference log-prob microbatch; production A6000 default=2",
    )
    parser.add_argument(
        "--disable_async_overlap",
        action="store_true",
        help="Disable rollout prefetch/overlap for debugging",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default="auto",
        help=(
            "auto/latest=resume from the highest valid checkpoint-N in OUTPUT_DIR; "
            "or provide an explicit checkpoint directory"
        ),
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Ignore existing checkpoint-N directories and start from SFT",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.rollout_superbatch_size < 1:
        parser.error("--rollout_superbatch_size must be >= 1")
    if args.prompt_batch_size is not None and args.prompt_batch_size < 1:
        parser.error("--prompt_batch_size must be >= 1")
    if int(args.judge_axis_batch_size) < 1:
        parser.error("--judge_axis_batch_size must be >= 1")
    if args.branch_type == "closed" and args.no_auxiliary:
        parser.error("--no_auxiliary is supported only with --branch_type open")
    return args


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _distributed_worker(worker_rank: int, worker_count: int, args_dict: Dict[str, Any], port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(worker_rank * GPUS_PER_PIPELINE + 2)
    dist.init_process_group("nccl", rank=worker_rank, world_size=worker_count)
    try:
        train_custom_grpo(argparse.Namespace(**args_dict))
    except Exception:
        print(f"[Rank {worker_rank}] Fatal error:")
        traceback.print_exc()
        raise
    finally:
        cleanup_distributed()


def main():
    args = parse_args()
    if args.num_generations < 2:
        raise ValueError("--num_generations must be at least 2 for comparative GRPO.")
    if args.rollout_superbatch_size < 1:
        raise ValueError("--rollout_superbatch_size must be at least 1.")
    if args.prompt_batch_size is not None and args.prompt_batch_size < 1:
        raise ValueError("--prompt_batch_size must be at least 1.")
    visible_gpus = torch.cuda.device_count()
    if visible_gpus not in (4, 8):
        raise RuntimeError(f"Expose exactly 4 or 8 GPUs; found {visible_gpus}.")

    # External torchrun is supported, but plain Python is the recommended interface.
    external_world = int(os.environ.get("WORLD_SIZE", "1"))
    if external_world > 1:
        if not dist.is_initialized():
            external_rank = int(os.environ.get("RANK", "0"))
            torch.cuda.set_device(external_rank * GPUS_PER_PIPELINE + 2)
            dist.init_process_group("nccl")
        try:
            train_custom_grpo(args)
        finally:
            cleanup_distributed()
        return

    worker_count = visible_gpus // GPUS_PER_PIPELINE
    if worker_count == 1:
        train_custom_grpo(args)
    else:
        import torch.multiprocessing as mp
        mp.spawn(
            _distributed_worker,
            args=(worker_count, vars(args), _find_free_port()),
            nprocs=worker_count,
            join=True,
        )


if __name__ == "__main__":
    main()
