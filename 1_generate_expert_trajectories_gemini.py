#!/usr/bin/env python3
"""
Resume-safe Gemini generator for MBA-Bench reference ideas.

Key guarantees
--------------
1. Never creates deterministic/template fallback ideas.
2. Preserves existing valid OpenAI- or Gemini-generated idea slots.
3. Removes known legacy fallback slots from contaminated files.
4. Stores incomplete work only under OUTPUT_DIR/_partial/.
5. Promotes a sample into OUTPUT_DIR/*.json only after all 15 idea slots are valid.
6. Failed slots remain missing and are retried on the next --skip_existing run.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple

from PIL import Image
from tqdm import tqdm
from google import genai
from google.genai import types


BUSINESS_LENSES: Dict[str, Dict[str, str]] = {
    "cost_resource_efficiency": {
        "display_name": "Cost / Resource Efficiency",
        "short_name": "cost",
        "description": (
            "operating cost reduction, resource productivity, energy/material "
            "efficiency, waste reduction, sustainability-linked savings"
        ),
    },
    "intelligent_automation": {
        "display_name": "Intelligent Automation",
        "short_name": "auto",
        "description": (
            "AI automation, sensing, computer vision, predictive analytics, "
            "technical intelligence, scalable decision support"
        ),
    },
    "customer_operations_growth": {
        "display_name": "Customer / Operations Growth",
        "short_name": "ops",
        "description": (
            "customer experience, service workflow, spatial/productivity flow, "
            "conversion, retention, operational throughput"
        ),
    },
}

DOMAIN_QUESTIONS = {
    "general_scene": (
        "Identify market-grounded startup opportunities from visible objects, "
        "activities, spatial context, and everyday physical-environment needs."
    ),
    "composition_layout": (
        "Identify startup opportunities from UI layout, information hierarchy, "
        "interaction friction, accessibility, conversion, and digital service workflow."
    ),
    "spatial_human_flow": (
        "Identify startup opportunities from human movement, spatial interaction, "
        "crowd/queue behavior, navigation, service flow, and space utilization."
    ),
    "visual_condition_anomaly": (
        "Identify startup opportunities from product condition, visible defects, "
        "anomaly signals, quality inspection, maintenance, and returns prevention."
    ),
    "aesthetic_texture_material": (
        "Identify startup opportunities from texture, material quality, surface "
        "appearance, tactile/visual differentiation, design, and product experience."
    ),
    "technical_visual_artifact": (
        "Identify B2B manufacturing opportunities from PCB trace/pad patterns, "
        "possible defect risks, AOI automation, yield optimization, and electronics "
        "quality control."
    ),
}

BAD_TERMS = [
    "page not found",
    "404",
    "access denied",
    "access to this page has been denied",
    "forbidden",
    "captcha",
    "verify you are human",
    "queries per second",
    "definition & meaning",
    "dictionary definition",
    "query, queries",
]

IDEA_JSON_FIELDS = ("title", "description", "implementation", "differentiation")
LEGACY_FALLBACK_TITLES = {
    "ResourceFlow Optimizer",
    "VisionOps Automation",
    "ExperienceFlow Assistant",
    "MarketGrounded Assistant",
}

IDEA_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query_index": {"type": "integer"},
        "source_query": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "topic": {"type": "string"},
        "idea": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "implementation": {"type": "string"},
                "differentiation": {"type": "string"},
            },
            "required": list(IDEA_JSON_FIELDS),
        },
        "visual_grounding": {"type": "string"},
        "market_grounding": {"type": "string"},
    },
    "required": [
        "query_index",
        "source_query",
        "evidence_ids",
        "topic",
        "idea",
        "visual_grounding",
        "market_grounding",
    ],
}

# Gemini 3.5 Flash standard paid pricing, USD per 1M tokens.
GEMINI_INPUT_USD_PER_M = 1.50
GEMINI_OUTPUT_USD_PER_M = 9.00


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(value).lower()).strip()


def shorten_field(value: Any, max_chars: int) -> str:
    text = clean_text(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def extract_json(text: str) -> Any:
    text = clean_text(text)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except Exception:
            pass
    for left, right in [("{", "}"), ("[", "]")]:
        start = text.find(left)
        end = text.rfind(right)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
    return None


def canonical_idea_json(obj: Any) -> Dict[str, str]:
    """Return a strict four-field idea object, or {} when invalid."""
    if isinstance(obj, str):
        obj = extract_json(obj)
    if not isinstance(obj, dict):
        return {}

    nested = obj.get("idea") or obj.get("business_idea") or obj.get("product")
    if isinstance(nested, str):
        nested = extract_json(nested)
    candidate = nested if isinstance(nested, dict) else obj

    aliases = {
        "title": ["title", "product_title", "product title", "name", "product_name"],
        "description": ["description", "product_description", "product description", "summary"],
        "implementation": ["implementation", "implementation_plan", "how_to_build", "technology_application"],
        "differentiation": ["differentiation", "competitive_advantage", "unique_value", "defensibility"],
    }

    lowered = {str(k).strip().lower(): v for k, v in candidate.items()}
    out: Dict[str, str] = {}
    for field, keys in aliases.items():
        value = None
        for key in keys:
            if key in candidate:
                value = candidate[key]
                break
            if key.lower() in lowered:
                value = lowered[key.lower()]
                break
        value = clean_text(value)
        if not value:
            return {}
        out[field] = value

    out["title"] = shorten_field(out["title"], 96)
    out["description"] = shorten_field(out["description"], 700)
    out["implementation"] = shorten_field(out["implementation"], 700)
    out["differentiation"] = shorten_field(out["differentiation"], 600)
    return out


def idea_json_to_string(obj: Mapping[str, str], indent: int = 2) -> str:
    return json.dumps(
        {field: obj[field] for field in IDEA_JSON_FIELDS},
        ensure_ascii=False,
        indent=indent,
    )


def idea_json_to_flat_text(obj: Mapping[str, str]) -> str:
    return (
        f"Product title: {obj['title']}\n"
        f"Product description: {obj['description']}\n"
        f"Implementation: {obj['implementation']}\n"
        f"Differentiation: {obj['differentiation']}"
    )


def read_jsonl(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def safe_json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json_if_valid(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def resolve_image_path(row: Mapping[str, Any], fallback_base_dir: str) -> Path:
    for key in ["image_path", "selected_image_path", "raw_image_path"]:
        value = row.get(key)
        if not value:
            continue
        direct = Path(str(value))
        if direct.exists():
            return direct
        relative = Path(fallback_base_dir) / str(value)
        if relative.exists():
            return relative
    raise FileNotFoundError(
        f"Could not resolve image path for sample_id={row.get('sample_id')}"
    )


def resize_image(path: Path, max_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max_side <= 0:
        return image
    width, height = image.size
    scale = max(width, height) / float(max_side)
    if scale <= 1:
        return image
    return image.resize(
        (max(1, int(width / scale)), max(1, int(height / scale))),
        Image.LANCZOS,
    )


def summarize_visual_annotation(row: Mapping[str, Any]) -> str:
    value = row.get("visual_annotation")
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)[:1200]
    return clean_text(value)[:1200]


def card_by_index(cards: Iterable[Mapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for card in cards:
        try:
            index = int(card.get("source_query_index"))
        except Exception:
            continue
        out[index] = dict(card)
    return out


def validate_precomputed_evidence(path: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    evidence_path = Path(path)
    if not evidence_path.exists():
        raise FileNotFoundError(f"precomputed evidence not found: {path}")

    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for line_no, line in enumerate(
        evidence_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = clean_text(row.get("sample_id"))
        lens = clean_text(row.get("business_lens"))
        queries = row.get("core_retrieval_queries", []) or []
        cards = row.get("web_evidence_cards", []) or []

        if not sample_id or lens not in BUSINESS_LENSES:
            raise RuntimeError(f"bad evidence key at line {line_no}")
        if len(queries) != 5:
            raise RuntimeError(
                f"line {line_no}: expected 5 core queries, got {len(queries)}"
            )
        if len(cards) != 5:
            raise RuntimeError(
                f"line {line_no}: expected 5 evidence cards, got {len(cards)}"
            )

        indexed = card_by_index(cards)
        for index in range(1, 6):
            if index not in indexed:
                raise RuntimeError(
                    f"line {line_no}: missing W{index}/source_query_index={index}"
                )
            card = indexed[index]
            blob = " ".join(
                [
                    str(card.get("title", "")),
                    str(card.get("url", "")),
                    str(card.get("text", ""))[:1200],
                ]
            ).lower()
            if not card.get("url") or not card.get("text"):
                raise RuntimeError(f"line {line_no}: empty evidence card W{index}")
            if any(term in blob for term in BAD_TERMS):
                raise RuntimeError(
                    f"line {line_no}: bad evidence term W{index}: {card.get('title')}"
                )

        key = (sample_id, lens)
        if key in out:
            raise RuntimeError(f"duplicate evidence for {key}")
        out[key] = row
    return out


def safe_output_name(row: Mapping[str, Any]) -> str:
    domain = normalize_key(row.get("domain", "domain")).replace(" ", "_")
    sample_seed = row.get("sample_id") or Path(
        clean_text(row.get("image_path"))
    ).stem
    sample_id = normalize_key(sample_seed).replace(" ", "_")
    return f"{domain}__{sample_id}.json"


def extract_item_title(item: Mapping[str, Any]) -> str:
    direct = clean_text(item.get("title"))
    if direct:
        return direct
    idea_json = item.get("idea_json")
    if isinstance(idea_json, dict):
        return clean_text(idea_json.get("title"))
    idea = item.get("idea")
    parsed = extract_json(idea) if isinstance(idea, str) else idea
    if isinstance(parsed, dict):
        return clean_text(parsed.get("title"))
    return ""


def is_valid_provider_idea(item: Any, expected_q: int) -> bool:
    """Accept legacy valid OpenAI items and new Gemini items; reject known fallbacks."""
    if not isinstance(item, dict):
        return False
    try:
        query_index = int(item.get("query_index"))
    except Exception:
        return False
    if query_index != expected_q:
        return False
    if item.get("evidence_ids") != [f"W{expected_q}"]:
        return False
    if extract_item_title(item) in LEGACY_FALLBACK_TITLES:
        return False
    if not canonical_idea_json(item):
        return False

    source = clean_text(item.get("generation_source"))
    if source and source not in {"openai", "gemini"}:
        return False
    return True


def collect_valid_items(
    records: Iterable[Mapping[str, Any]],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Merge valid slots. Later records override earlier records."""
    merged: Dict[str, Dict[int, Dict[str, Any]]] = {
        lens: {} for lens in BUSINESS_LENSES
    }
    for record in records:
        if not isinstance(record, Mapping):
            continue
        for result in record.get("results", []) or []:
            if not isinstance(result, Mapping):
                continue
            lens = clean_text(result.get("business_lens"))
            if lens not in BUSINESS_LENSES:
                continue
            for item in result.get("data", []) or []:
                if not isinstance(item, dict):
                    continue
                try:
                    query_index = int(item.get("query_index"))
                except Exception:
                    continue
                if 1 <= query_index <= 5 and is_valid_provider_idea(
                    item, query_index
                ):
                    merged[lens][query_index] = item
    return merged


def compact_evidence_card(
    card: Mapping[str, Any], preview_chars: int = 360, include_full_text: bool = False
) -> Dict[str, Any]:
    out = {
        "evidence_id": card.get("evidence_id"),
        "source_query_index": card.get("source_query_index"),
        "source_query": card.get("source_query"),
        "title": card.get("title"),
        "domain": card.get("domain"),
        "url": card.get("url"),
        "text_preview": clean_text(card.get("text"))[:preview_chars],
    }
    if include_full_text:
        out["text"] = card.get("text")
    return out


def attach_matched_evidence(
    idea: Mapping[str, Any], card: Mapping[str, Any]
) -> Dict[str, Any]:
    query_index = int(idea["query_index"])
    item = dict(idea)
    item["evidence_ids"] = [f"W{query_index}"]
    item["matched_web_evidence"] = [
        {
            "evidence_id": card.get("evidence_id"),
            "source_query_index": card.get("source_query_index"),
            "source_query": card.get("source_query"),
            "title": card.get("title"),
            "url": card.get("url"),
            "domain": card.get("domain"),
        }
    ]
    return item


def build_record(
    row: Mapping[str, Any],
    evidence_by_lens: Mapping[str, Mapping[str, Any]],
    valid_items: Mapping[str, Mapping[int, Mapping[str, Any]]],
    preview_chars: int,
    include_full_text: bool,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for lens in BUSINESS_LENSES:
        pack = evidence_by_lens[lens]
        cards = pack.get("web_evidence_cards", []) or []
        data = [
            dict(valid_items[lens][index])
            for index in sorted(valid_items.get(lens, {}))
            if index in valid_items[lens]
        ]
        results.append(
            {
                "business_lens": lens,
                "core_retrieval_queries": pack.get("core_retrieval_queries", []),
                "web_evidence_cards": [
                    compact_evidence_card(card, preview_chars, include_full_text)
                    for card in cards
                ],
                "data": data,
            }
        )

    complete_slots = sum(len(result["data"]) for result in results)
    return {
        "schema_version": "mba_marketdb",
        "sample_id": clean_text(row.get("sample_id")),
        "domain": row.get("domain"),
        "domain_display_name": row.get("domain_display_name"),
        "source_dataset": row.get("source_dataset"),
        "image_path": row.get("image_path"),
        "domain_question": DOMAIN_QUESTIONS.get(row.get("domain"), ""),
        "results": results,
        "generation_stage": "mba_marketdb_5query_5evidence_5structured_json_idea_pairs",
        "generation_status": "complete" if complete_slots == 15 else "partial",
        "completed_idea_slots": complete_slots,
        "expected_idea_slots": 15,
    }


def is_complete_generation_record(record: Mapping[str, Any]) -> bool:
    results = record.get("results", []) or []
    if len(results) != 3:
        return False
    seen_lenses = set()
    total_ideas = 0
    for result in results:
        if not isinstance(result, Mapping):
            return False
        lens = result.get("business_lens")
        if lens not in BUSINESS_LENSES or lens in seen_lenses:
            return False
        seen_lenses.add(lens)
        if len(result.get("core_retrieval_queries", []) or []) != 5:
            return False
        if len(result.get("web_evidence_cards", []) or []) != 5:
            return False
        data = result.get("data", []) or []
        if len(data) != 5:
            return False
        for expected_q, item in enumerate(data, 1):
            if not is_valid_provider_idea(item, expected_q):
                return False
        total_ideas += len(data)
    return seen_lenses == set(BUSINESS_LENSES) and total_ideas == 15


def is_complete_generation_file(path: Path) -> bool:
    return is_complete_generation_record(read_json_if_valid(path))


def rebuild_flat_jsonl(
    out_dir: Path, rows: Iterable[Mapping[str, Any]], flat_path: Path
) -> int:
    tmp = flat_path.with_suffix(flat_path.suffix + ".tmp")
    written = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            path = out_dir / safe_output_name(row)
            if not is_complete_generation_file(path):
                continue
            handle.write(path.read_text(encoding="utf-8").strip() + "\n")
            written += 1
    tmp.replace(flat_path)
    return written


class UsageTracker:
    def __init__(self, log_path: Path, max_estimated_cost_usd: float):
        self.log_path = log_path
        self.max_estimated_cost_usd = max(0.0, max_estimated_cost_usd)
        self.lock = threading.Lock()
        self.estimated_cost_usd = 0.0
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def ensure_budget_available(self) -> None:
        with self.lock:
            if (
                self.max_estimated_cost_usd > 0
                and self.estimated_cost_usd >= self.max_estimated_cost_usd
            ):
                raise RuntimeError(
                    "Gemini run budget guard reached: "
                    f"estimated_cost_usd={self.estimated_cost_usd:.6f} "
                    f"limit={self.max_estimated_cost_usd:.6f}"
                )

    def record(
        self,
        response: Any,
        sample_id: str,
        lens: str,
        query_index: int,
        model: str,
    ) -> None:
        usage = getattr(response, "usage_metadata", None)
        prompt = int(getattr(usage, "prompt_token_count", 0) or 0)
        candidates = int(getattr(usage, "candidates_token_count", 0) or 0)
        thoughts = int(getattr(usage, "thoughts_token_count", 0) or 0)
        billable_output = candidates + thoughts
        estimated = (
            prompt * GEMINI_INPUT_USD_PER_M
            + billable_output * GEMINI_OUTPUT_USD_PER_M
        ) / 1_000_000.0

        record = {
            "ts": time.time(),
            "sample_id": sample_id,
            "business_lens": lens,
            "query_index": query_index,
            "model": model,
            "prompt_tokens": prompt,
            "candidate_tokens": candidates,
            "thought_tokens": thoughts,
            "estimated_cost_usd": estimated,
        }
        with self.lock:
            self.prompt_tokens += prompt
            self.output_tokens += billable_output
            self.estimated_cost_usd += estimated
            self.calls += 1
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def summary(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "output_plus_thought_tokens": self.output_tokens,
                "estimated_cost_usd": self.estimated_cost_usd,
                "run_budget_limit_usd": self.max_estimated_cost_usd,
            }


class GeminiIdeaGenerator:
    def __init__(
        self,
        model: str,
        api_key: str,
        max_retries: int,
        max_output_tokens: int,
        thinking_level: str,
        usage_tracker: UsageTracker,
    ):
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        timeout_ms = int(
            os.environ.get("GEMINI_REQUEST_TIMEOUT_MS", "180000")
        )
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
        self.model = model
        self.max_retries = max(0, max_retries)
        self.max_output_tokens = max(64, max_output_tokens)
        self.thinking_level = thinking_level
        self.usage_tracker = usage_tracker

    @staticmethod
    def build_prompt(
        row: Mapping[str, Any],
        lens_key: str,
        lens_info: Mapping[str, str],
        query_index: int,
        source_query: str,
        evidence_card: Mapping[str, Any],
        evidence_context_chars: int,
    ) -> str:
        domain = clean_text(row.get("domain"))
        domain_display = clean_text(row.get("domain_display_name")) or domain
        facts = evidence_card.get("extracted_market_facts") or []
        facts_text = (
            "; ".join(str(value) for value in facts[:8])
            if isinstance(facts, list)
            else ""
        )
        evidence_text = clean_text(evidence_card.get("text"))[
            :evidence_context_chars
        ]
        source_query_json = json.dumps(source_query, ensure_ascii=False)

        return f"""
You are an expert venture analyst creating MBA-Bench reference ideas.
Generate exactly ONE startup idea from exactly ONE image/query/evidence pair.

Image/domain context:
- domain: {domain_display}
- source dataset: {clean_text(row.get('source_dataset'))}
- domain-specific question: {DOMAIN_QUESTIONS.get(domain, '')}
- visual metadata: {summarize_visual_annotation(row)}

Business opportunity lens:
- lens_key: {lens_key}
- lens_name: {lens_info['display_name']}
- lens_focus: {lens_info['description']}

Strict paired market evidence:
- Q{query_index}: {source_query}
- W{query_index} title: {clean_text(evidence_card.get('title'))}
- W{query_index} domain: {clean_text(evidence_card.get('domain'))}
- W{query_index} URL: {clean_text(evidence_card.get('url'))}
- W{query_index} market facts: {facts_text}
- W{query_index} evidence text: {evidence_text}

Rules:
1. Use only Q{query_index} and W{query_index}; do not use other evidence.
2. Ground the idea in both visible image cues and W{query_index}.
3. Do not invent evidence.
4. Do not copy a company name or source title as the product title.
5. Produce one concrete, technically feasible, differentiated business idea.
6. Return JSON only and follow the requested schema exactly.
7. Set query_index to {query_index}.
8. Set source_query exactly to {source_query_json}.
9. Set evidence_ids exactly to ["W{query_index}"].
10. Set topic equal to idea.title.
""".strip()

    def _config(self) -> Dict[str, Any]:
        # google-genai generate_content structured output configuration.
        # response_format belongs to the Interactions API and is rejected by
        # GenerateContentConfig. Use response_mime_type plus JSON Schema here.
        return {
            "max_output_tokens": self.max_output_tokens,
            "response_mime_type": "application/json",
            "response_json_schema": IDEA_RESPONSE_SCHEMA,
            "thinking_config": {"thinking_level": self.thinking_level},
        }

    def generate_one(
        self,
        image: Image.Image,
        row: Mapping[str, Any],
        lens_key: str,
        query_index: int,
        source_query: str,
        evidence_card: Mapping[str, Any],
        evidence_context_chars: int,
    ) -> Dict[str, Any]:
        sample_id = clean_text(row.get("sample_id"))
        prompt = self.build_prompt(
            row=row,
            lens_key=lens_key,
            lens_info=BUSINESS_LENSES[lens_key],
            query_index=query_index,
            source_query=source_query,
            evidence_card=evidence_card,
            evidence_context_chars=evidence_context_chars,
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.usage_tracker.ensure_budget_available()
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[image, prompt],
                    config=self._config(),
                )
                self.usage_tracker.record(
                    response=response,
                    sample_id=sample_id,
                    lens=lens_key,
                    query_index=query_index,
                    model=self.model,
                )
                parsed = extract_json(getattr(response, "text", "") or "")
                item = self.parse_response(
                    parsed,
                    expected_q=query_index,
                    source_query=source_query,
                )
                return attach_matched_evidence(item, evidence_card)
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                sleep_seconds = min(60.0, 2.0 * (2**attempt) + random.random())
                time.sleep(sleep_seconds)

        raise RuntimeError(
            "Gemini idea generation failed after retries: "
            f"sample={sample_id} lens={lens_key} q={query_index}/5 "
            f"error={type(last_error).__name__}: {last_error}"
        ) from last_error

    def parse_response(
        self, obj: Any, expected_q: int, source_query: str
    ) -> Dict[str, Any]:
        if isinstance(obj, list) and obj:
            obj = obj[0]
        if not isinstance(obj, dict):
            raise RuntimeError("Gemini did not return a JSON object")

        idea_obj = canonical_idea_json(obj)
        if not idea_obj:
            idea_obj = canonical_idea_json(obj.get("idea"))
        if not idea_obj:
            raise RuntimeError(f"invalid structured idea JSON: {obj}")
        if idea_obj["title"] in LEGACY_FALLBACK_TITLES:
            raise RuntimeError("Gemini returned a reserved legacy fallback title")

        return {
            "query_index": int(expected_q),
            "source_query": source_query,
            "evidence_ids": [f"W{expected_q}"],
            "generation_source": "gemini",
            "generation_model": self.model,
            "topic": idea_obj["title"],
            "idea": idea_json_to_string(idea_obj, indent=2),
            "idea_json": idea_obj,
            "idea_text": idea_json_to_flat_text(idea_obj),
            "title": idea_obj["title"],
            "description": idea_obj["description"],
            "implementation": idea_obj["implementation"],
            "differentiation": idea_obj["differentiation"],
            "visual_grounding": clean_text(obj.get("visual_grounding")),
            "market_grounding": clean_text(obj.get("market_grounding")),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_manifest_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--precomputed_web_evidence_jsonl", required=True)
    parser.add_argument("--require_precomputed_web_evidence", action="store_true")
    parser.add_argument("--strict_web_evidence", action="store_true")
    parser.add_argument("--min_web_evidence_cards", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--image_workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fallback_base_dir", default="/workspace")
    parser.add_argument(
        "--gemini_model",
        default=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
    )
    parser.add_argument(
        "--gemini_api_key", default=os.environ.get("GEMINI_API_KEY", "")
    )
    parser.add_argument("--max_image_side", type=int, default=768)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--idea_max_output_tokens", type=int, default=1200)
    parser.add_argument(
        "--thinking_level",
        choices=["minimal", "low", "medium", "high"],
        default=os.environ.get("GEMINI_THINKING_LEVEL", "minimal"),
    )
    parser.add_argument(
        "--evidence_context_chars",
        type=int,
        default=int(os.environ.get("EVIDENCE_CONTEXT_CHARS", "360")),
    )
    parser.add_argument("--include_full_evidence_text_in_outputs", action="store_true")
    parser.add_argument("--evidence_text_preview_chars", type=int, default=360)
    parser.add_argument(
        "--max_estimated_cost_usd",
        type=float,
        default=float(os.environ.get("GEMINI_MAX_ESTIMATED_COST_USD", "0")),
        help="Optional estimated cost cap for this run; 0 disables the guard.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.source_manifest_jsonl, limit=args.limit)
    evidence = validate_precomputed_evidence(
        args.precomputed_web_evidence_jsonl
    )
    expected_evidence = len(rows) * len(BUSINESS_LENSES)
    if (
        args.require_precomputed_web_evidence
        and len(evidence) < expected_evidence
    ):
        raise RuntimeError(
            f"precomputed evidence rows too small: {len(evidence)}/{expected_evidence}"
        )
    if args.min_web_evidence_cards != 5:
        raise RuntimeError("strict generator requires --min_web_evidence_cards 5")
    if not args.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    out_dir = Path(args.output_dir)
    partial_dir = out_dir / "_partial"
    state_dir = out_dir / "_state"
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    flat_path = out_dir / "all_expert_trajectories.jsonl"
    failure_log_path = state_dir / "generation_failures.jsonl"
    usage_log_path = state_dir / "gemini_usage.jsonl"
    failure_lock = threading.Lock()
    usage_tracker = UsageTracker(
        log_path=usage_log_path,
        max_estimated_cost_usd=args.max_estimated_cost_usd,
    )

    if args.overwrite:
        for path in out_dir.glob("*.json"):
            path.unlink()
        for path in partial_dir.glob("*.json"):
            path.unlink()
        flat_path.unlink(missing_ok=True)
        failure_log_path.unlink(missing_ok=True)
        usage_log_path.unlink(missing_ok=True)

    print(
        f"[GEN START] images={len(rows)} workers={args.image_workers} "
        f"model={args.gemini_model} thinking={args.thinking_level} output={out_dir}"
    )

    rows_to_process: List[Tuple[int, Dict[str, Any]]] = []
    skipped = 0
    for index, row in enumerate(rows, 1):
        final_path = out_dir / safe_output_name(row)
        partial_path = partial_dir / safe_output_name(row)
        if args.skip_existing and is_complete_generation_file(final_path):
            skipped += 1
            partial_path.unlink(missing_ok=True)
            continue
        rows_to_process.append((index, row))

    print(
        f"[GEN RESUME] remaining_samples={len(rows_to_process)} "
        f"skipped_complete={skipped}/{len(rows)}"
    )

    # Preflight sanitization happens before any paid Gemini request. Every legacy
    # root JSON that is not fully valid is reduced to valid provider-generated
    # slots only, moved under _partial/, and removed from the final output root.
    sanitized_files = 0
    preserved_slots_preflight = 0
    for _, row in rows_to_process:
        final_path = out_dir / safe_output_name(row)
        partial_path = partial_dir / safe_output_name(row)
        if not final_path.exists():
            continue

        sample_id = clean_text(row.get("sample_id"))
        evidence_by_lens: Dict[str, Dict[str, Any]] = {}
        for lens in BUSINESS_LENSES:
            key = (sample_id, lens)
            if key not in evidence:
                raise RuntimeError(
                    f"missing precomputed evidence during preflight: "
                    f"sample={sample_id} lens={lens}"
                )
            evidence_by_lens[lens] = evidence[key]

        valid_items = collect_valid_items(
            [read_json_if_valid(final_path), read_json_if_valid(partial_path)]
        )
        preserved = sum(len(values) for values in valid_items.values())
        if preserved:
            partial_record = build_record(
                row=row,
                evidence_by_lens=evidence_by_lens,
                valid_items=valid_items,
                preview_chars=args.evidence_text_preview_chars,
                include_full_text=args.include_full_evidence_text_in_outputs,
            )
            safe_json_dump(partial_path, partial_record)
            preserved_slots_preflight += preserved
        final_path.unlink()
        sanitized_files += 1

    if rows_to_process:
        # Prevent an old flat JSONL containing fallback data from being consumed.
        flat_path.unlink(missing_ok=True)

    print(
        f"[PREFLIGHT SANITIZE] moved_incomplete_files={sanitized_files} "
        f"preserved_valid_slots={preserved_slots_preflight}"
    )

    def log_failure(
        idx_row: Tuple[int, Dict[str, Any]], exc: Exception
    ) -> None:
        index, row = idx_row
        record = {
            "ts": time.time(),
            "index": index,
            "sample_id": clean_text(row.get("sample_id")),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        with failure_lock:
            with failure_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def process_one(
        idx_row: Tuple[int, Dict[str, Any]]
    ) -> Tuple[int, str, float, int, int]:
        index, row = idx_row
        started = time.time()
        sample_id = clean_text(row.get("sample_id"))
        final_path = out_dir / safe_output_name(row)
        partial_path = partial_dir / safe_output_name(row)

        evidence_by_lens: Dict[str, Dict[str, Any]] = {}
        for lens in BUSINESS_LENSES:
            key = (sample_id, lens)
            if key not in evidence:
                raise RuntimeError(
                    f"missing precomputed evidence for sample={sample_id} lens={lens}"
                )
            evidence_by_lens[lens] = evidence[key]

        # Merge valid slots from the old final file and any previous partial state.
        # The partial record is loaded last so the latest successful slot wins.
        final_record = read_json_if_valid(final_path)
        partial_record = read_json_if_valid(partial_path)
        valid_items = collect_valid_items([final_record, partial_record])
        preserved_before = sum(len(values) for values in valid_items.values())

        # Immediately sanitize legacy contaminated files: write only valid slots to
        # _partial/, then remove the incomplete/fallback-contaminated root JSON.
        sanitized = build_record(
            row=row,
            evidence_by_lens=evidence_by_lens,
            valid_items=valid_items,
            preview_chars=args.evidence_text_preview_chars,
            include_full_text=args.include_full_evidence_text_in_outputs,
        )
        if is_complete_generation_record(sanitized):
            safe_json_dump(final_path, sanitized)
            partial_path.unlink(missing_ok=True)
            return index, sample_id, time.time() - started, preserved_before, 0

        safe_json_dump(partial_path, sanitized)
        if final_path.exists():
            final_path.unlink()

        image_path = resolve_image_path(row, args.fallback_base_dir)
        image = resize_image(image_path, args.max_image_side)
        generator = GeminiIdeaGenerator(
            model=args.gemini_model,
            api_key=args.gemini_api_key,
            max_retries=args.max_retries,
            max_output_tokens=args.idea_max_output_tokens,
            thinking_level=args.thinking_level,
            usage_tracker=usage_tracker,
        )

        generated_now = 0
        for lens in BUSINESS_LENSES:
            pack = evidence_by_lens[lens]
            queries = pack.get("core_retrieval_queries", []) or []
            cards = card_by_index(pack.get("web_evidence_cards", []) or [])
            for query_index in range(1, 6):
                if query_index in valid_items[lens]:
                    continue
                if len(queries) < query_index or query_index not in cards:
                    raise RuntimeError(
                        f"missing Q/W pair: sample={sample_id} lens={lens} q={query_index}"
                    )

                item = generator.generate_one(
                    image=image,
                    row=row,
                    lens_key=lens,
                    query_index=query_index,
                    source_query=queries[query_index - 1],
                    evidence_card=cards[query_index],
                    evidence_context_chars=args.evidence_context_chars,
                )
                valid_items[lens][query_index] = item
                generated_now += 1

                # Save each successful slot immediately. If the next slot fails,
                # this work is preserved and the failed slot remains absent.
                partial_record_now = build_record(
                    row=row,
                    evidence_by_lens=evidence_by_lens,
                    valid_items=valid_items,
                    preview_chars=args.evidence_text_preview_chars,
                    include_full_text=args.include_full_evidence_text_in_outputs,
                )
                safe_json_dump(partial_path, partial_record_now)

        complete_record = build_record(
            row=row,
            evidence_by_lens=evidence_by_lens,
            valid_items=valid_items,
            preview_chars=args.evidence_text_preview_chars,
            include_full_text=args.include_full_evidence_text_in_outputs,
        )
        if not is_complete_generation_record(complete_record):
            raise RuntimeError(
                f"sample incomplete after generation: sample={sample_id}"
            )

        safe_json_dump(final_path, complete_record)
        partial_path.unlink(missing_ok=True)
        return (
            index,
            sample_id,
            time.time() - started,
            preserved_before,
            generated_now,
        )

    started_all = time.time()
    written = 0
    failed = 0

    if not rows_to_process:
        print("[GEN] nothing to process; all final files are complete")
    elif args.image_workers <= 1:
        progress = tqdm(
            rows_to_process, desc="GEN", unit="img", dynamic_ncols=True
        )
        for idx_row in progress:
            try:
                _, sample_id, elapsed, preserved, generated = process_one(
                    idx_row
                )
            except Exception as exc:
                failed += 1
                log_failure(idx_row, exc)
                sample_id = clean_text(idx_row[1].get("sample_id"))
                progress.set_postfix(
                    complete=f"{skipped + written}/{len(rows)}", failed=failed
                )
                tqdm.write(
                    f"[GEN IMAGE FAILED] sample={sample_id} "
                    f"error={type(exc).__name__}: {exc}"
                )
                continue
            written += 1
            progress.set_postfix(
                complete=f"{skipped + written}/{len(rows)}", failed=failed
            )
            tqdm.write(
                f"[GEN IMAGE] complete={skipped + written}/{len(rows)} "
                f"sample={sample_id} preserved={preserved} "
                f"gemini_generated={generated} sec={elapsed:.1f}"
            )
    else:
        with futures.ThreadPoolExecutor(
            max_workers=max(1, args.image_workers)
        ) as executor:
            future_map = {
                executor.submit(process_one, idx_row): idx_row
                for idx_row in rows_to_process
            }
            progress = tqdm(
                futures.as_completed(future_map),
                total=len(future_map),
                desc="GEN",
                unit="img",
                dynamic_ncols=True,
            )
            for future in progress:
                idx_row = future_map[future]
                try:
                    _, sample_id, elapsed, preserved, generated = future.result()
                except Exception as exc:
                    failed += 1
                    log_failure(idx_row, exc)
                    sample_id = clean_text(idx_row[1].get("sample_id"))
                    progress.set_postfix(
                        complete=f"{skipped + written}/{len(rows)}",
                        failed=failed,
                    )
                    tqdm.write(
                        f"[GEN IMAGE FAILED] sample={sample_id} "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    continue
                written += 1
                progress.set_postfix(
                    complete=f"{skipped + written}/{len(rows)}", failed=failed
                )
                tqdm.write(
                    f"[GEN IMAGE] complete={skipped + written}/{len(rows)} "
                    f"sample={sample_id} preserved={preserved} "
                    f"gemini_generated={generated} sec={elapsed:.1f}"
                )

    flat_written = rebuild_flat_jsonl(out_dir, rows, flat_path)
    usage_summary = usage_tracker.summary()
    print(
        f"[GEN DONE] completed_now={written} failed_now={failed} "
        f"skipped_complete={skipped} flat_jsonl_rows={flat_written} "
        f"total_seconds={time.time()-started_all:.1f}"
    )
    print(f"[PARTIAL DIR] {partial_dir}")
    print(f"[FAIL LOG] {failure_log_path}")
    print(f"[USAGE LOG] {usage_log_path}")
    print("[GEMINI USAGE THIS RUN] " + json.dumps(usage_summary))


if __name__ == "__main__":
    main()
