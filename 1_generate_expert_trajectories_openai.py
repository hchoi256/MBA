#!/usr/bin/env python3
"""
1_generate_expert_trajectories_openai.py

Strict compact-quality 5-query -> 5-evidence -> 5-structured-JSON-idea generator for MBA-Bench.

Required benchmark structure:
  1 image
    × 3 business opportunity lenses
    × 5 query/evidence/idea pairs
  = 15 structured JSON ideas per image.

Guarantee:
- This script does NOT call DDG.
- It requires precomputed evidence from 0_prepare_market_evidence.py.
- For each lens, Idea i must use Query i and Evidence Wi.
- Every generated idea is canonicalized to four JSON keys: title, description, implementation, differentiation.
"""

import argparse
import concurrent.futures as futures
import base64
import hashlib
import io
import json
import os
import random
import re
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image
from tqdm import tqdm
from openai import OpenAI

BUSINESS_LENSES = {
    "cost_resource_efficiency": {
        "display_name": "Cost / Resource Efficiency",
        "short_name": "cost",
        "description": "operating cost reduction, resource productivity, energy/material efficiency, waste reduction, sustainability-linked savings",
    },
    "intelligent_automation": {
        "display_name": "Intelligent Automation",
        "short_name": "auto",
        "description": "AI automation, sensing, computer vision, predictive analytics, technical intelligence, scalable decision support",
    },
    "customer_operations_growth": {
        "display_name": "Customer / Operations Growth",
        "short_name": "ops",
        "description": "customer experience, service workflow, spatial/productivity flow, conversion, retention, operational throughput",
    },
}

DOMAIN_QUESTIONS = {
    "general_scene": "Identify market-grounded startup opportunities from visible objects, activities, spatial context, and everyday physical-environment needs.",
    "composition_layout": "Identify startup opportunities from UI layout, information hierarchy, interaction friction, accessibility, conversion, and digital service workflow.",
    "spatial_human_flow": "Identify startup opportunities from human movement, spatial interaction, crowd/queue behavior, navigation, service flow, and space utilization.",
    "visual_condition_anomaly": "Identify startup opportunities from product condition, visible defects, anomaly signals, quality inspection, maintenance, and returns prevention.",
    "aesthetic_texture_material": "Identify startup opportunities from texture, material quality, surface appearance, tactile/visual differentiation, design, and product experience.",
    "technical_visual_artifact": "Identify B2B manufacturing opportunities from PCB trace/pad patterns, possible defect risks, AOI automation, yield optimization, and electronics quality control.",
}

BAD_TERMS = [
    "page not found", "404", "access denied", "access to this page has been denied",
    "forbidden", "captcha", "verify you are human", "queries per second",
    "definition & meaning", "dictionary definition", "query, queries",
]

IDEA_JSON_FIELDS = ("title", "description", "implementation", "differentiation")
FALLBACK_TITLES = {"ResourceFlow Optimizer", "VisionOps Automation", "ExperienceFlow Assistant"}
IDEA_JSON_SCHEMA_TEXT = """
Each idea must be a structured JSON business idea with exactly four non-empty string fields:
{
  "title": "concise product name",
  "description": "core function, target user, user/business need, and benefit",
  "implementation": "how the product would be built or deployed using current technology and the image/evidence context",
  "differentiation": "what makes it meaningfully different or defensible compared with existing solutions"
}
Do not mention patents. Do not add extra keys inside the idea object.
""".strip()


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def normalize_key(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(x).lower()).strip()


def shorten_field(x: Any, max_chars: int) -> str:
    text = clean_text(x)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def canonical_idea_json(obj: Any) -> Dict[str, str]:
    """Return a strict four-key idea object or {} when invalid."""
    if isinstance(obj, str):
        parsed = extract_json(obj)
        if isinstance(parsed, dict):
            obj = parsed
    if not isinstance(obj, dict):
        return {}

    # Accept either a nested idea object or top-level fields from the model.
    nested = obj.get("idea") or obj.get("business_idea") or obj.get("product")
    if isinstance(nested, str):
        nested_parsed = extract_json(nested)
        if isinstance(nested_parsed, dict):
            nested = nested_parsed
    if isinstance(nested, dict):
        candidate = nested
    else:
        candidate = obj

    aliases = {
        "title": ["title", "product_title", "product title", "name", "product_name"],
        "description": ["description", "product_description", "product description", "summary"],
        "implementation": ["implementation", "implementation_plan", "how_to_build", "technology_application"],
        "differentiation": ["differentiation", "competitive_advantage", "unique_value", "defensibility"],
    }
    out = {}
    lowered = {str(k).strip().lower(): v for k, v in candidate.items()}
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


def idea_json_to_string(obj: Dict[str, str], indent: int = 2) -> str:
    return json.dumps({field: obj[field] for field in IDEA_JSON_FIELDS}, ensure_ascii=False, indent=indent)


def idea_json_to_flat_text(obj: Dict[str, str]) -> str:
    return (
        f"Product title: {obj['title']}\n"
        f"Product description: {obj['description']}\n"
        f"Implementation: {obj['implementation']}\n"
        f"Differentiation: {obj['differentiation']}"
    )


def read_jsonl(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def safe_json_dump(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def resolve_image_path(row: Dict[str, Any], fallback_base_dir: str) -> Path:
    for key in ["image_path", "selected_image_path", "raw_image_path"]:
        v = row.get(key)
        if not v:
            continue
        p = Path(str(v))
        if p.exists():
            return p
        p2 = Path(fallback_base_dir) / str(v)
        if p2.exists():
            return p2
    raise FileNotFoundError(f"Could not resolve image path for sample_id={row.get('sample_id')}")


def resize_image(image: Image.Image, max_side: int) -> Image.Image:
    image = image.convert("RGB")
    if max_side <= 0:
        return image
    w, h = image.size
    scale = max(w, h) / float(max_side)
    if scale <= 1:
        return image
    return image.resize((max(1, int(w / scale)), max(1, int(h / scale))), Image.LANCZOS)


def image_to_data_url(path: Path, max_side: int = 768, jpeg_quality: int = 82) -> str:
    image = resize_image(Image.open(path).convert("RGB"), max_side)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_response_text(resp: Any) -> str:
    out = getattr(resp, "output_text", None)
    if isinstance(out, str) and out.strip():
        return out.strip()
    parts = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            t = getattr(c, "text", None)
            if isinstance(t, str):
                parts.append(t)
    return "\n".join(parts).strip()


def extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    for a, b in [("{", "}"), ("[", "]")]:
        s, e = text.find(a), text.rfind(b)
        if s >= 0 and e > s:
            try:
                return json.loads(text[s:e+1])
            except Exception:
                pass
    return None


def summarize_visual_annotation(row: Dict[str, Any]) -> str:
    va = row.get("visual_annotation")
    if isinstance(va, dict):
        return json.dumps(va, ensure_ascii=False)[:1200]
    return clean_text(va)[:1200]


def card_by_index(cards: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out = {}
    for c in cards:
        try:
            idx = int(c.get("source_query_index"))
        except Exception:
            continue
        out[idx] = c
    return out


def compact_pair_block(core_queries: List[str], cards: List[Dict[str, Any]], max_chars: int = 1200) -> str:
    by_idx = card_by_index(cards)
    lines = []
    for i in range(1, 6):
        q = core_queries[i - 1] if len(core_queries) >= i else ""
        c = by_idx.get(i, {})
        facts = c.get("extracted_market_facts") or []
        facts_text = "; ".join(str(x) for x in facts[:8]) if isinstance(facts, list) else ""
        evidence_text = clean_text(c.get("text"))[:max_chars]
        lines.append(
            f"Pair {i}:\n"
            f"Q{i}: {q}\n"
            f"W{i}: title={clean_text(c.get('title'))}; domain={clean_text(c.get('domain'))}; "
            f"url={clean_text(c.get('url'))}; "
            f"market_facts={facts_text}; "
            f"evidence_text={evidence_text}"
        )
    return "\n".join(lines)


def validate_precomputed_evidence(path: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"precomputed evidence not found: {path}")
    out = {}
    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        sample_id = clean_text(r.get("sample_id"))
        lens = clean_text(r.get("business_lens"))
        queries = r.get("core_retrieval_queries", []) or []
        cards = r.get("web_evidence_cards", []) or []
        if not sample_id or lens not in BUSINESS_LENSES:
            raise RuntimeError(f"bad evidence key at line {line_no}")
        if len(queries) != 5:
            raise RuntimeError(f"line {line_no}: expected 5 core queries, got {len(queries)}")
        if len(cards) != 5:
            raise RuntimeError(f"line {line_no}: expected 5 evidence cards, got {len(cards)}")
        by_idx = card_by_index(cards)
        for i in range(1, 6):
            if i not in by_idx:
                raise RuntimeError(f"line {line_no}: missing W{i}/source_query_index={i}")
            c = by_idx[i]
            blob = " ".join([str(c.get("title","")), str(c.get("url","")), str(c.get("text",""))[:1200]]).lower()
            if not c.get("url") or not c.get("text"):
                raise RuntimeError(f"line {line_no}: empty evidence card W{i}")
            if any(t in blob for t in BAD_TERMS):
                raise RuntimeError(f"line {line_no}: bad evidence term W{i}: {c.get('title')}")
        key = (sample_id, lens)
        if key in out:
            raise RuntimeError(f"duplicate evidence for {key}")
        out[key] = r
    return out


def safe_output_name(row: Dict[str, Any]) -> str:
    domain = normalize_key(row.get("domain", "domain")).replace(" ", "_")
    sample_id = normalize_key(row.get("sample_id") or Path(clean_text(row.get("image_path"))).stem).replace(" ", "_")
    return f"{domain}__{sample_id}.json"


def extract_item_title(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    title = clean_text(item.get("title"))
    if title:
        return title
    idea_json = item.get("idea_json")
    if isinstance(idea_json, dict):
        return clean_text(idea_json.get("title"))
    idea = item.get("idea")
    if isinstance(idea, dict):
        return clean_text(idea.get("title"))
    if isinstance(idea, str):
        parsed = extract_json(idea)
        if isinstance(parsed, dict):
            return clean_text(parsed.get("title"))
    return ""


def is_fallback_idea_item(item: Dict[str, Any]) -> bool:
    return extract_item_title(item) in FALLBACK_TITLES


def is_complete_generation_record(r: Dict[str, Any]) -> bool:
    results = r.get("results", []) or []
    if len(results) != 3:
        return False
    total_ideas = 0
    total_cards = 0
    for res in results:
        if res.get("business_lens") not in BUSINESS_LENSES:
            return False
        if len(res.get("core_retrieval_queries", []) or []) != 5:
            return False
        if len(res.get("web_evidence_cards", []) or []) != 5:
            return False
        data = res.get("data", []) or []
        if len(data) != 5:
            return False
        total_ideas += len(data)
        total_cards += len(res.get("web_evidence_cards", []) or [])
        for i, item in enumerate(data, 1):
            if item.get("query_index") != i:
                return False
            if item.get("evidence_ids") != [f"W{i}"]:
                return False
            if not canonical_idea_json(item):
                return False
            generation_source = clean_text(item.get("generation_source"))
            if generation_source:
                if generation_source != "openai":
                    return False
            else:
                # Backward compatibility: accept legacy files only when they are not known fallback outputs.
                if is_fallback_idea_item(item):
                    return False
    return total_ideas == 15 and total_cards == 15


class OpenAIStrictFiveIdeaGenerator:
    """
    Per-idea OpenAI generator.

    Previous behavior:
      1 image -> 1 OpenAI call -> 15 ideas

    Current behavior:
      1 image -> 15 OpenAI calls
      each call uses exactly:
        image + domain/question/visual_annotation + one business lens + one query/evidence pair
      and returns exactly 1 structured idea.
    """

    def __init__(self, model: str, api_key: str, image_detail: str, timeout: float, max_retries: int, max_image_side: int):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.image_detail = image_detail
        self.max_retries = max_retries
        self.max_image_side = max_image_side

    def generate(self, image_path: Path, row: Dict[str, Any], evidence_by_lens: Dict[str, Dict[str, Any]], idea_temperature: float, max_tokens: int) -> Dict[str, List[Dict[str, Any]]]:
        domain = clean_text(row.get("domain"))
        domain_display = clean_text(row.get("domain_display_name")) or domain
        source_dataset = clean_text(row.get("source_dataset"))
        visual_annotation = summarize_visual_annotation(row)
        domain_question = DOMAIN_QUESTIONS.get(domain, "")
        sample_id = clean_text(row.get("sample_id"))

        evidence_context_chars = int(os.environ.get("EVIDENCE_CONTEXT_CHARS", "360"))
        data_url = image_to_data_url(image_path, max_side=self.max_image_side)

        out = {k: [] for k in BUSINESS_LENSES}

        for lens_key, lens_info in BUSINESS_LENSES.items():
            pack = evidence_by_lens[lens_key]
            queries = pack.get("core_retrieval_queries", []) or []
            cards = pack.get("web_evidence_cards", []) or []
            by_idx = card_by_index(cards)

            for qi in range(1, 6):
                if len(queries) < qi:
                    raise RuntimeError(f"sample={sample_id} lens={lens_key}: missing Q{qi}")
                if qi not in by_idx:
                    raise RuntimeError(f"sample={sample_id} lens={lens_key}: missing W{qi}")

                q = queries[qi - 1]
                c = by_idx[qi]
                idea = self.generate_one_idea(
                    data_url=data_url,
                    sample_id=sample_id,
                    domain=domain,
                    domain_display=domain_display,
                    source_dataset=source_dataset,
                    visual_annotation=visual_annotation,
                    domain_question=domain_question,
                    lens_key=lens_key,
                    lens_info=lens_info,
                    query_index=qi,
                    source_query=q,
                    evidence_card=c,
                    evidence_context_chars=evidence_context_chars,
                    idea_temperature=idea_temperature,
                    max_tokens=max_tokens,
                )
                out[lens_key].append(idea)

        if not all(len(out.get(k, [])) == 5 for k in BUSINESS_LENSES):
            raise RuntimeError(f"per-idea generation incomplete: { {k: len(out.get(k, [])) for k in BUSINESS_LENSES} }")
        return out

    def build_one_idea_prompt(
        self,
        domain_display: str,
        source_dataset: str,
        visual_annotation: str,
        domain_question: str,
        lens_key: str,
        lens_info: Dict[str, str],
        query_index: int,
        source_query: str,
        evidence_card: Dict[str, Any],
        evidence_context_chars: int,
    ) -> str:
        facts = evidence_card.get("extracted_market_facts") or []
        facts_text = "; ".join(str(x) for x in facts[:8]) if isinstance(facts, list) else ""
        evidence_text = clean_text(evidence_card.get("text"))[:evidence_context_chars]
        evidence_block = (
            f"Q{query_index}: {source_query}\n"
            f"W{query_index}: title={clean_text(evidence_card.get('title'))}; "
            f"domain={clean_text(evidence_card.get('domain'))}; "
            f"url={clean_text(evidence_card.get('url'))}; "
            f"market_facts={facts_text}; "
            f"evidence_text={evidence_text}"
        )

        prompt = f"""
You are an expert venture analyst building MBA-Bench market-grounded startup ideation trajectories.

You will generate exactly ONE startup idea for exactly ONE query/evidence pair.

Image/domain context:
- domain: {domain_display}
- source dataset: {source_dataset}
- domain-specific ideation question: {domain_question}
- visual metadata: {visual_annotation}

Business opportunity lens:
- lens_key: {lens_key}
- lens_name: {lens_info['display_name']}
- lens_focus: {lens_info['description']}

Single prevalidated DDG query/evidence pair:
{evidence_block}

Task:
Generate exactly 1 startup idea.
The mapping is strict:
- This idea must use Q{query_index} and W{query_index} only.
- Do not use other queries or other evidence.
- Ground the idea in both visible image cues and W{query_index}.
- Do not invent evidence.
- Do not copy company names or source titles as the product/topic/title.

Return valid JSON only. Do not include markdown fences, explanations, comments, or prose outside JSON.
The JSON must have exactly this top-level structure:
{{
  "query_index": {query_index},
  "source_query": "{source_query}",
  "evidence_ids": ["W{query_index}"],
  "topic": "same as idea.title",
  "idea": {{
    "title": "concise product name",
    "description": "core function, target user, user/business need, and benefit",
    "implementation": "how the product would be built or deployed using current technology and the image/evidence context",
    "differentiation": "what makes it meaningfully different or defensible compared with existing solutions"
  }},
  "visual_grounding": "how the idea uses visible image cues",
  "market_grounding": "how the idea uses the paired evidence W{query_index}"
}}

Structured idea schema:
{IDEA_JSON_SCHEMA_TEXT}
""".strip()
        return prompt

    def generate_one_idea(
        self,
        data_url: str,
        sample_id: str,
        domain: str,
        domain_display: str,
        source_dataset: str,
        visual_annotation: str,
        domain_question: str,
        lens_key: str,
        lens_info: Dict[str, str],
        query_index: int,
        source_query: str,
        evidence_card: Dict[str, Any],
        evidence_context_chars: int,
        idea_temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        prompt = self.build_one_idea_prompt(
            domain_display=domain_display,
            source_dataset=source_dataset,
            visual_annotation=visual_annotation,
            domain_question=domain_question,
            lens_key=lens_key,
            lens_info=lens_info,
            query_index=query_index,
            source_query=source_query,
            evidence_card=evidence_card,
            evidence_context_chars=evidence_context_chars,
        )

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                # print(
                #     f"[GEN IDEA CALL] sample={sample_id} domain={domain} lens={lens_key} "
                #     f"q={query_index}/5 attempt={attempt+1}/{self.max_retries+1} "
                #     f"prompt_chars={len(prompt)} evidence_context_chars={evidence_context_chars} max_tokens={max_tokens}",
                #     flush=True,
                # )
                resp = self.client.responses.create(
                    model=self.model,
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": data_url, "detail": self.image_detail},
                        ],
                    }],
                    temperature=float(idea_temperature),
                    top_p=0.9,
                    max_output_tokens=int(max_tokens),
                )
                text = extract_response_text(resp)
                obj = extract_json(text)
                parsed = self.parse_one_idea(obj, query_index, source_query)
                # print(
                #     f"[GEN IDEA DONE] sample={sample_id} lens={lens_key} q={query_index}/5 title={parsed.get('title','')}",
                #     flush=True,
                # )
                return parsed
            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                wait = 2.0 * (2 ** attempt) + random.random()
                # print(
                #     f"[GEN IDEA RETRY] sample={sample_id} lens={lens_key} q={query_index}/5 "
                #     f"{attempt+1}/{self.max_retries} sleep={wait:.1f}s error={type(e).__name__}",
                #     flush=True,
                # )
                time.sleep(wait)
        # Final repair path: try one strict text-only repair call. If that also fails,
        # mark this idea as failed and let the caller skip the entire image for later resume.
        repair_prompt = prompt + "\n\nCRITICAL REPAIR: The previous response was invalid. Return exactly one JSON object only. No markdown. No prose."
        repair_err = None
        try:
            repair_resp = self.client.responses.create(
                model=self.model,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": repair_prompt},
                    ],
                }],
                temperature=0.0,
                top_p=0.9,
                max_output_tokens=int(max(max_tokens, 1200)),
            )
            repair_text = extract_response_text(repair_resp)
            repair_obj = extract_json(repair_text)
            parsed = self.parse_one_idea(repair_obj, query_index, source_query)
            return parsed
        except Exception as e:
            repair_err = e

        msg = (
            f"OpenAI idea generation failed after retries and repair: "
            f"sample={sample_id} lens={lens_key} q={query_index}/5 "
            f"primary_error={type(last_err).__name__}: {last_err}; "
            f"repair_error={type(repair_err).__name__}: {repair_err}"
        )
        print(f"[GEN IDEA FAILED] {msg}", flush=True)
        raise RuntimeError(msg) from (repair_err or last_err)


    def fallback_one_idea(
        self,
        sample_id: str,
        domain_display: str,
        visual_annotation: str,
        lens_key: str,
        lens_info: Dict[str, str],
        query_index: int,
        source_query: str,
        evidence_card: Dict[str, Any],
        last_err: Any,
    ) -> Dict[str, Any]:
        """Deterministic no-abort fallback used only after OpenAI repeatedly fails JSON parsing."""
        title_seed = {
            "cost_resource_efficiency": "ResourceFlow Optimizer",
            "intelligent_automation": "VisionOps Automation",
            "customer_operations_growth": "ExperienceFlow Assistant",
        }.get(lens_key, "MarketGrounded Assistant")

        ev_title = clean_text(evidence_card.get("title"))
        ev_domain = clean_text(evidence_card.get("domain"))
        facts = evidence_card.get("extracted_market_facts") or []
        facts_text = "; ".join(str(x) for x in facts[:3]) if isinstance(facts, list) else ""
        visual = clean_text(visual_annotation)[:360] or f"the {domain_display} image context"
        market = clean_text(facts_text or ev_title or source_query)[:420]

        idea_obj = {
            "title": title_seed,
            "description": shorten_field(
                f"A market-grounded product for users operating in {domain_display} contexts. "
                f"It uses the visible scene context and the paired market signal from W{query_index} "
                f"to identify practical cost, workflow, or service improvements tied to: {source_query}.",
                700,
            ),
            "implementation": shorten_field(
                f"Deploy a lightweight computer-vision and workflow analytics layer that converts visible cues "
                f"from the scene into operational recommendations. The system links detected objects, space usage, "
                f"or activity patterns with the W{query_index} evidence signal from {ev_domain or 'the validated source'} "
                f"and presents prioritized actions through a dashboard or API.",
                700,
            ),
            "differentiation": shorten_field(
                f"Unlike generic analytics tools, it binds one specific visual context to one validated market evidence "
                f"card and produces a targeted action path for the {lens_info.get('display_name', lens_key)} lens.",
                600,
            ),
        }

        return {
            "query_index": int(query_index),
            "source_query": source_query,
            "evidence_ids": [f"W{query_index}"],
            "topic": idea_obj["title"],
            "idea": idea_json_to_string(idea_obj, indent=2),
            "idea_json": idea_obj,
            "idea_text": idea_json_to_flat_text(idea_obj),
            "title": idea_obj["title"],
            "description": idea_obj["description"],
            "implementation": idea_obj["implementation"],
            "differentiation": idea_obj["differentiation"],
            "visual_grounding": shorten_field(f"Uses visible scene cues from the image metadata: {visual}", 700),
            "market_grounding": shorten_field(
                f"Uses W{query_index}, the paired DDG evidence for query '{source_query}'. "
                f"Evidence title/domain: {ev_title} / {ev_domain}. Market signal: {market}",
                700,
            ),
        }


    def parse_one_idea(self, obj: Any, query_index: int, source_query: str) -> Dict[str, Any]:
        if isinstance(obj, list) and obj:
            obj = obj[0]
        if isinstance(obj, dict) and isinstance(obj.get("result"), dict):
            obj = obj["result"]
        if isinstance(obj, dict) and isinstance(obj.get("results"), dict):
            # Accept accidental wrapper formats.
            obj = obj["results"]

        if not isinstance(obj, dict):
            raise RuntimeError("model did not return a JSON object")

        idea_obj = canonical_idea_json(obj)
        if not idea_obj:
            nested = obj.get("idea") or obj.get("business_idea") or obj.get("product")
            idea_obj = canonical_idea_json(nested)
        if not idea_obj:
            raise RuntimeError(f"bad structured idea JSON: {obj}")

        topic = idea_obj["title"]
        return {
            "query_index": int(query_index),
            "source_query": source_query,
            "evidence_ids": [f"W{query_index}"],
            "generation_source": "openai",
            "topic": topic,
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

    # Kept for backward compatibility with older debug helpers. The per-idea path
    # does not use this batch parser.
    def parse_ideas(self, obj: Any, evidence_by_lens: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        if isinstance(obj, dict) and isinstance(obj.get("results"), dict):
            obj = obj["results"]
        out = {k: [] for k in BUSINESS_LENSES}
        if not isinstance(obj, dict):
            return out

        for lens_key in BUSINESS_LENSES:
            pack = evidence_by_lens[lens_key]
            queries = pack.get("core_retrieval_queries", []) or []
            vals = obj.get(lens_key, [])
            if isinstance(vals, dict):
                vals = vals.get("ideas", [])
            if not isinstance(vals, list):
                vals = []

            by_idx = {}
            for x in vals:
                if not isinstance(x, dict):
                    continue
                try:
                    qi = int(x.get("query_index"))
                except Exception:
                    continue
                if 1 <= qi <= 5 and qi not in by_idx:
                    by_idx[qi] = x

            for qi in range(1, 6):
                x = by_idx.get(qi, {})
                idea_obj = canonical_idea_json(x)
                if not idea_obj:
                    continue
                topic = idea_obj["title"]
                out[lens_key].append({
                    "query_index": qi,
                    "source_query": queries[qi - 1],
                    "evidence_ids": [f"W{qi}"],
                    "topic": topic,
                    "idea": idea_json_to_string(idea_obj, indent=2),
                    "idea_json": idea_obj,
                    "idea_text": idea_json_to_flat_text(idea_obj),
                    "title": idea_obj["title"],
                    "description": idea_obj["description"],
                    "implementation": idea_obj["implementation"],
                    "differentiation": idea_obj["differentiation"],
                    "visual_grounding": clean_text(x.get("visual_grounding")),
                    "market_grounding": clean_text(x.get("market_grounding")),
                })
        return out

def compact_evidence_card(c: Dict[str, Any], preview_chars: int = 360, include_full_text: bool = False) -> Dict[str, Any]:
    out = {
        "evidence_id": c.get("evidence_id"),
        "source_query_index": c.get("source_query_index"),
        "source_query": c.get("source_query"),
        "title": c.get("title"),
        "domain": c.get("domain"),
        "url": c.get("url"),
        "text_preview": clean_text(c.get("text"))[:preview_chars],
    }
    if include_full_text:
        out["text"] = c.get("text")
    return out


def enrich_ideas(ideas: List[Dict[str, Any]], cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_idx = card_by_index(cards)
    out = []
    for idea in ideas:
        qi = int(idea["query_index"])
        c = by_idx[qi]
        item = dict(idea)
        item["evidence_ids"] = [f"W{qi}"]
        item["matched_web_evidence"] = [{
            "evidence_id": c.get("evidence_id"),
            "source_query_index": c.get("source_query_index"),
            "source_query": c.get("source_query"),
            "title": c.get("title"),
            "url": c.get("url"),
            "domain": c.get("domain"),
        }]
        out.append(item)
    return out



def is_complete_generation_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        r = json.loads(path.read_text(encoding="utf-8"))
        return is_complete_generation_record(r)
    except Exception:
        return False


def rebuild_flat_jsonl(out_dir: Path, rows: List[Dict[str, Any]], flat_path: Path):
    with flat_path.open("w", encoding="utf-8") as flat:
        written = 0
        for row in rows:
            p = out_dir / safe_output_name(row)
            if not is_complete_generation_file(p):
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            flat.write(json.dumps(r, ensure_ascii=False) + "\n")
            written += 1
    return written


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source_manifest_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--precomputed_web_evidence_jsonl", required=True)
    p.add_argument("--require_precomputed_web_evidence", action="store_true")
    p.add_argument("--strict_web_evidence", action="store_true")
    p.add_argument("--min_web_evidence_cards", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--image_workers", type=int, default=1, help="Parallel image-level workers for OpenAI idea generation.")
    p.add_argument("--overwrite", action="store_true", help="Delete existing output files and rebuild from scratch.")
    p.add_argument("--fallback_base_dir", default="/workspace")
    p.add_argument("--openai_model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"))
    p.add_argument("--openai_api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--openai_image_detail", default=os.environ.get("OPENAI_IMAGE_DETAIL", "low"))
    p.add_argument("--max_image_side", type=int, default=768)
    p.add_argument("--request_timeout", type=float, default=120.0)
    p.add_argument("--max_retries", type=int, default=2)
    p.add_argument("--idea_temperature", type=float, default=0.7)
    p.add_argument("--idea_max_output_tokens", type=int, default=900)
    p.add_argument("--include_full_evidence_text_in_outputs", action="store_true")
    p.add_argument("--evidence_text_preview_chars", type=int, default=360)
    # Compatibility args accepted but unused.
    p.add_argument("--evidence_index_dir", default="")
    p.add_argument("--web_cache_dir", default="")
    p.add_argument("--use_market_db", action="store_true")
    p.add_argument("--retriever_device", default="cpu")
    p.add_argument("--top_k_per_query", type=int, default=8)
    p.add_argument("--fetch_k_per_query", type=int, default=80)
    p.add_argument("--query_temperature", type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()
    rows = read_jsonl(args.source_manifest_jsonl, limit=args.limit)
    evidence = validate_precomputed_evidence(args.precomputed_web_evidence_jsonl)
    expected = len(rows) * len(BUSINESS_LENSES)
    if args.require_precomputed_web_evidence and len(evidence) < expected:
        raise RuntimeError(f"precomputed evidence rows too small: {len(evidence)}/{expected}")
    if args.min_web_evidence_cards != 5:
        raise RuntimeError("strict 5-query generator requires --min_web_evidence_cards 5")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flat_path = out_dir / "all_expert_trajectories.jsonl"
    failure_log_path = out_dir / "generation_failures.jsonl"
    failure_log_lock = threading.Lock()

    if args.overwrite:
        for p in out_dir.glob("*.json"):
            p.unlink()
        if flat_path.exists():
            flat_path.unlink()
        if failure_log_path.exists():
            failure_log_path.unlink()

    print(f"[GEN START] images={len(rows)} workers={args.image_workers} output={out_dir}")

    rows_to_process = []
    skipped = 0
    for idx, row in enumerate(rows, 1):
        out_file = out_dir / safe_output_name(row)
        if args.skip_existing and is_complete_generation_file(out_file):
            skipped += 1
            continue
        rows_to_process.append((idx, row))

    print(f"[GEN RESUME] remaining={len(rows_to_process)} skipped_existing={skipped}/{len(rows)}")

    def log_failure(idx_row: Tuple[int, Dict[str, Any]], exc: Exception):
        idx, row = idx_row
        sample_id = clean_text(row.get("sample_id"))
        rec = {
            "idx": idx,
            "sample_id": sample_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "ts": time.time(),
        }
        with failure_log_lock:
            with failure_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def process_one(idx_row: Tuple[int, Dict[str, Any]]) -> Tuple[int, str, float]:
        idx, row = idx_row
        image_start = time.time()
        out_file = out_dir / safe_output_name(row)
        sample_id = clean_text(row.get("sample_id"))
        image_path = resolve_image_path(row, args.fallback_base_dir)

        evidence_by_lens = {}
        for lens_key in BUSINESS_LENSES:
            key = (sample_id, lens_key)
            if key not in evidence:
                raise RuntimeError(f"missing precomputed evidence for sample={sample_id} lens={lens_key}")
            evidence_by_lens[lens_key] = evidence[key]

        gen_local = OpenAIStrictFiveIdeaGenerator(
            model=args.openai_model,
            api_key=args.openai_api_key,
            image_detail=args.openai_image_detail,
            timeout=args.request_timeout,
            max_retries=args.max_retries,
            max_image_side=args.max_image_side,
        )

        ideas_by_lens = gen_local.generate(
            image_path=image_path,
            row=row,
            evidence_by_lens=evidence_by_lens,
            idea_temperature=args.idea_temperature,
            max_tokens=args.idea_max_output_tokens,
        )

        results = []
        for lens_key in BUSINESS_LENSES:
            pack = evidence_by_lens[lens_key]
            cards = pack.get("web_evidence_cards", []) or []
            results.append({
                "business_lens": lens_key,
                "core_retrieval_queries": pack.get("core_retrieval_queries", []),
                "web_evidence_cards": [compact_evidence_card(c, args.evidence_text_preview_chars, args.include_full_evidence_text_in_outputs) for c in cards],
                "data": enrich_ideas(ideas_by_lens.get(lens_key, []), cards),
            })

        total_ideas = sum(len(r["data"]) for r in results)
        if total_ideas != 15:
            raise RuntimeError(f"Expected 15 ideas for sample={sample_id}, got {total_ideas}")

        rec = {
            "schema_version": "mba_marketdb",
            "sample_id": sample_id,
            "domain": row.get("domain"),
            "domain_display_name": row.get("domain_display_name"),
            "source_dataset": row.get("source_dataset"),
            "image_path": row.get("image_path"),
            "domain_question": DOMAIN_QUESTIONS.get(row.get("domain"), ""),
            "results": results,
            "generation_stage": "mba_marketdb_5query_5evidence_5structured_json_idea_pairs",
        }
        safe_json_dump(out_file, rec)
        return idx, sample_id, time.time() - image_start

    t0 = time.time()
    written = 0
    failed = 0

    if not rows_to_process:
        print("[Generate] nothing to process; all per-image files already complete")
    elif args.image_workers <= 1:
        pbar = tqdm(rows_to_process, desc="GEN", unit="img", dynamic_ncols=True)
        for idx_row in pbar:
            try:
                idx, sample_id, elapsed = process_one(idx_row)
            except Exception as exc:
                failed += 1
                log_failure(idx_row, exc)
                _, row = idx_row
                sample_id = clean_text(row.get("sample_id"))
                pbar.set_postfix(done=f"{skipped + written}/{len(rows)}", failed=failed)
                tqdm.write(f"[GEN IMAGE FAILED] sample={sample_id} error={type(exc).__name__}: {exc}")
                continue
            written += 1
            done = skipped + written
            pbar.set_postfix(done=f"{done}/{len(rows)}", failed=failed)
            tqdm.write(f"[GEN IMAGE] {done}/{len(rows)} sample={sample_id} ideas=15 sec={elapsed:.1f}")
    else:
        with futures.ThreadPoolExecutor(max_workers=max(1, int(args.image_workers))) as ex:
            future_map = {ex.submit(process_one, idx_row): idx_row for idx_row in rows_to_process}
            pbar = tqdm(futures.as_completed(future_map), total=len(future_map), desc="GEN", unit="img", dynamic_ncols=True)
            for fut in pbar:
                idx_row = future_map[fut]
                try:
                    idx, sample_id, elapsed = fut.result()
                except Exception as exc:
                    failed += 1
                    log_failure(idx_row, exc)
                    _, row = idx_row
                    sample_id = clean_text(row.get("sample_id"))
                    pbar.set_postfix(done=f"{skipped + written}/{len(rows)}", failed=failed)
                    tqdm.write(f"[GEN IMAGE FAILED] sample={sample_id} error={type(exc).__name__}: {exc}")
                    continue
                written += 1
                done = skipped + written
                pbar.set_postfix(done=f"{done}/{len(rows)}", failed=failed)
                tqdm.write(f"[GEN IMAGE] {done}/{len(rows)} sample={sample_id} ideas=15 sec={elapsed:.1f}")

    flat_written = rebuild_flat_jsonl(out_dir, rows, flat_path)
    print(
        f"[GEN DONE] written_now={written} failed_now={failed} skipped_existing={skipped} "
        f"flat_jsonl_rows={flat_written} total_seconds={time.time()-t0:.1f}"
    )
    print(f"[GEN FAIL LOG] {failure_log_path}")


if __name__ == "__main__":
    main()
