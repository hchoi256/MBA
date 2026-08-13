#!/usr/bin/env python3
"""
0_prepare_market_evidence.py

Strict 5-query compact-quality DDG market evidence prefetch for AIStarBench.

Required benchmark structure:
  1 image
    × 3 business lenses
    × 5 focused market queries
    × 1 valid DDG market evidence page per query

Output:
  One compact JSONL row per image × lens.
  Each row contains exactly 5 core_retrieval_queries and exactly 5 web_evidence_cards.
  Redundant raw fields such as query_generation_text, retrieval_queries, and ddg_search_results are not saved in the main evidence file.
  W1 corresponds to query 1, W2 to query 2, ..., W5 to query 5.

Guarantees:
- Runs BEFORE 1_generate_expert_trajectories_openai.py.
- Uses one OpenAI vision call per image to generate all 15 queries.
- For each core query, searches DDG and validates candidate URLs.
- 404 / 403 / 429 / access denied / CAPTCHA / dictionary / empty pages are rejected.
- If any query cannot get valid evidence, this script fails and 1_generate never runs.
"""

import argparse
import base64
import concurrent.futures as futures
import hashlib
import io
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, unquote, urlparse

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

DOMAIN_HINTS = {
    "general_scene": "everyday physical spaces, home/retail/facility services, visible user needs",
    "composition_layout": "mobile app UI/UX, accessibility, conversion, interface workflow",
    "spatial_human_flow": "crowd flow, queue management, public facility navigation, space utilization",
    "visual_condition_anomaly": "visual inspection, defect detection, quality control, anomaly detection, predictive maintenance",
    "aesthetic_texture_material": "texture, material pattern, visual design, fashion/interior/product material workflow",
    "technical_visual_artifact": "PCB inspection, circuit board defect detection, hardware QA, electronics manufacturing",
}

DOMAIN_QUESTIONS = {
    "general_scene": "Identify market-grounded startup opportunities from visible objects, activities, spatial context, and everyday physical-environment needs.",
    "composition_layout": "Identify startup opportunities from UI layout, information hierarchy, interaction friction, accessibility, conversion, and digital service workflow.",
    "spatial_human_flow": "Identify startup opportunities from human movement, spatial interaction, crowd/queue behavior, navigation, service flow, and space utilization.",
    "visual_condition_anomaly": "Identify startup opportunities from product condition, visible defects, anomaly signals, quality inspection, maintenance, and returns prevention.",
    "aesthetic_texture_material": "Identify startup opportunities from texture, material quality, surface appearance, tactile/visual differentiation, design, and product experience.",
    "technical_visual_artifact": "Identify B2B manufacturing opportunities from PCB trace/pad patterns, possible defect risks, AOI automation, yield optimization, and electronics quality control.",
}

SKIP_DOMAIN_PATTERNS = [
    "duckduckgo.com", "google.com", "bing.com", "yahoo.com",
    "facebook.com", "instagram.com", "tiktok.com", "pinterest.com", "x.com", "twitter.com",
    "youtube.com", "youtu.be", "reddit.com", "quora.com",
    "dictionary.com", "wordwebonline.com", "merriam-webster.com", "wiktionary.org",
    "wikipedia.org", "wikimedia.org",
    "amazon.com", "amazon.co.", "ebay.com", "etsy.com",
    "apple.com/app-store", "play.google.com",
    "storage.googleapis.com", "blogspot.", "tumblr.com",
]

BAD_TITLE_TERMS = [
    "queries per second", "definition & meaning", "dictionary definition",
    "query, queries", "page not found", "404", "access denied", "forbidden", "not found",
]

BAD_TEXT_TERMS = [
    "page not found", "404", "access to this page has been denied", "access denied",
    "forbidden", "captcha", "verify you are human", "enable javascript",
    "oops! page not found", "not found the page", "this page could not be found",
    "you are being rate limited", "bot detection", "request blocked",
    "please enable cookies", "automation tools to browse the website",
    "are you a robot", "cloudflare ray id",
]

MARKET_SIGNAL_TERMS = [
    "market", "industry", "business", "customer", "customers", "company", "companies",
    "startup", "solution", "platform", "software", "tool", "service", "product",
    "technology", "automation", "ai", "computer vision", "analytics", "workflow",
    "efficiency", "sustainable", "sustainability", "design", "materials", "inspection",
    "quality control", "manufacturing", "supply chain", "case study", "report",
    "retail", "operations", "management", "monitoring", "optimization",
]


QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "into",
    "is", "it", "of", "on", "or", "the", "to", "with", "without", "using", "use", "uses",
    "home", "homes", "living", "room", "rooms", "space", "spaces", "solutions", "solution",
    "systems", "system", "tools", "tool", "market", "report", "case", "study", "industry",
}

HIGH_QUALITY_DOMAIN_HINTS = [
    ".gov", ".edu", ".org", "energy.gov", "epa.gov", "nist.gov", "energystar.gov",
    "ellenmacarthurfoundation.org", "mckinsey.com", "deloitte.com", "pwc.com",
    "gartner.com", "forrester.com", "iea.org", "oecd.org", "who.int", "worldbank.org",
    "springer", "sciencedirect", "ieee", "acm.org", "nature.com", "mdpi.com",
]

LOW_QUALITY_DOMAIN_HINTS = [
    "pinterest", "facebook", "instagram", "tiktok", "youtube", "quora", "reddit",
    "amazon.", "etsy.", "ebay.", "shopify", "medium.com",
    "storage.googleapis.com", "blogspot.", "tumblr.com",
]

GENERIC_PAGE_TERMS = [
    "shop", "cart", "account", "sign in", "track order", "registry", "favorites",
    "homepage", "home page", "loading", "menu", "skip to content",
]


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def normalize_key(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(x).lower()).strip()


def stable_hash(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8", errors="ignore")).hexdigest()


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


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def domain_of_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def should_skip_url(url: str) -> bool:
    u = clean_text(url).lower()
    if not u.startswith(("http://", "https://")):
        return True
    return any(p in u for p in SKIP_DOMAIN_PATTERNS)


def query_terms(query: str) -> List[str]:
    terms = []
    for t in re.findall(r"[a-zA-Z][a-zA-Z0-9]{2,}", clean_text(query).lower()):
        if t not in QUERY_STOPWORDS:
            terms.append(t)
    seen, out = set(), []
    for t in terms:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out[:12]


def is_generic_homepage(url: str, title: str, text: str) -> bool:
    try:
        parsed = urlparse(url)
        path = (parsed.path or "/").strip()
    except Exception:
        path = "/"
    blob = " ".join([clean_text(title), clean_text(text)[:1200]]).lower()
    if path in {"", "/"}:
        generic_hits = sum(1 for t in GENERIC_PAGE_TERMS if t in blob)
        return generic_hits >= 2 or len(clean_text(text)) < 1200
    return False


def evidence_relevance_score(core_query: str, hit: Dict[str, Any], fetched: Dict[str, Any]) -> Tuple[float, List[str]]:
    title = clean_text(hit.get("title"))
    snippet = clean_text(hit.get("snippet"))
    url = clean_text(fetched.get("final_url") or hit.get("url"))
    domain = domain_of_url(url) or clean_text(hit.get("domain"))
    text = clean_text(fetched.get("text"))
    blob = " ".join([title, snippet, text[:2200]]).lower()
    terms = query_terms(core_query)
    matched = [t for t in terms if t in blob]
    score = 0.0
    reasons = []
    if terms:
        score += 45.0 * (len(matched) / max(1, len(terms)))
        reasons.append(f"query_overlap={len(matched)}/{len(terms)}")
    else:
        reasons.append("query_overlap=0/0")
    market_hits = [t for t in MARKET_SIGNAL_TERMS if t in blob]
    score += min(20.0, 2.5 * len(market_hits))
    reasons.append(f"market_signal_terms={len(market_hits)}")
    if any(h in domain or h in url for h in HIGH_QUALITY_DOMAIN_HINTS):
        score += 18.0
        reasons.append("trusted_or_report_like_domain")
    if any(h in domain or h in url for h in LOW_QUALITY_DOMAIN_HINTS):
        score -= 15.0
        reasons.append("low_quality_domain_penalty")
    path = urlparse(url).path if url else ""
    if path and path not in {"/", ""} and len(path.strip("/")) >= 8:
        score += 8.0
        reasons.append("specific_deep_page")
    if is_generic_homepage(url, title, text):
        score -= 35.0
        reasons.append("generic_homepage_penalty")
    if len(text) >= 900:
        score += 6.0
        reasons.append("substantial_text")
    if len(text) >= 1800:
        score += 4.0
        reasons.append("rich_text")
    return round(score, 2), reasons


SALIENT_MARKET_TERMS = set(MARKET_SIGNAL_TERMS) | {
    "market size", "market share", "cagr", "forecast", "projected", "projection",
    "growth", "revenue", "segment", "adoption", "demand", "driver", "trend",
    "cost", "saving", "efficiency", "waste", "customer", "automation", "ai",
    "software", "platform", "inspection", "quality", "sustainability", "energy",
}

BOILERPLATE_SENTENCE_TERMS = {
    "cookie", "cookies", "privacy policy", "terms of use", "subscribe", "newsletter",
    "sign in", "log in", "contact us", "click here", "all rights reserved", "skip to content",
    "home page", "shopping cart", "related articles", "advertisement",
}


def split_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    # Keep compact page-extracted text readable while avoiding dependency-heavy NLP.
    rough = re.split(r"(?<=[.!?])\s+|\s+[•●▪]\s+|\s+-\s+(?=[A-Z0-9])", text)
    out = []
    for s in rough:
        s = clean_text(s)
        if len(s) < 45 or len(s) > 850:
            continue
        low = s.lower()
        if any(t in low for t in BOILERPLATE_SENTENCE_TERMS):
            continue
        # Remove sentences that are mostly navigation fragments.
        alpha = sum(ch.isalpha() for ch in s)
        if alpha < max(20, int(len(s) * 0.35)):
            continue
        out.append(s)
    return out


def extract_market_facts(text: str, max_facts: int = 12) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    patterns = [
        r"(?:USD|US\$|\$)\s?\d+(?:\.\d+)?\s?(?:billion|million|trillion|bn|mn)?",
        r"\d+(?:\.\d+)?\s?%",
        r"CAGR\s*(?:of)?\s*\d+(?:\.\d+)?\s?%",
        r"\d{4}\s*(?:-|to|–)\s*\d{4}",
        r"\d+(?:\.\d+)?\s?(?:billion|million|trillion)\b",
    ]
    seen, facts = set(), []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I):
            fact = clean_text(m.group(0))
            k = fact.lower()
            if k and k not in seen:
                facts.append(fact)
                seen.add(k)
            if len(facts) >= max_facts:
                return facts
    return facts


def salient_sentence_score(sentence: str, query: str) -> float:
    low = sentence.lower()
    terms = query_terms(query)
    score = 0.0
    if terms:
        score += 7.0 * sum(1 for t in terms if t in low)
    score += 3.0 * sum(1 for t in SALIENT_MARKET_TERMS if t in low)
    if re.search(r"(?:USD|US\$|\$)\s?\d|\d+(?:\.\d+)?\s?%|CAGR|\b\d{4}\b", sentence, re.I):
        score += 12.0
    if re.search(r"market size|market share|forecast|projected|projected to reach|growth|driver|trend|adoption|demand|cost savings?|efficien", low, re.I):
        score += 8.0
    # Prefer compact factual sentences over long page dumps.
    if len(sentence) <= 240:
        score += 2.0
    elif len(sentence) > 500:
        score -= 3.0
    return score


def build_salient_market_text(core_query: str, hit: Dict[str, Any], fetched: Dict[str, Any], max_chars: int) -> Tuple[str, Dict[str, Any]]:
    """Return a query-aware, sentence-bound evidence block instead of an arbitrary prefix cut."""
    title = clean_text(hit.get("title"))
    snippet = clean_text(hit.get("snippet"))
    raw_text = clean_text(fetched.get("text"))
    max_chars = max(800, int(max_chars or 2500))

    sentences = split_sentences(raw_text)
    scored = []
    for i, s in enumerate(sentences):
        sc = salient_sentence_score(s, core_query)
        if sc > 0:
            scored.append((sc, i, s))
    # Keep top evidence sentences, but restore source order for readability.
    top = sorted(scored, key=lambda x: (-x[0], x[1]))[:18]
    top = sorted(top, key=lambda x: x[1])

    parts = []
    if title:
        parts.append(f"Title: {title}.")
    if snippet and snippet.lower() not in (title or "").lower():
        parts.append(f"Search snippet: {snippet}")

    facts = extract_market_facts(raw_text)
    if facts:
        parts.append("Extracted market facts: " + "; ".join(facts) + ".")

    for _, _, s in top:
        if sum(len(p) + 1 for p in parts) + len(s) + 1 > max_chars:
            continue
        if s not in parts:
            parts.append(s)
        if sum(len(p) + 1 for p in parts) >= max_chars * 0.92:
            break

    if len(parts) <= 2:
        # Fallback: still sentence-bound, not a raw mid-word slice.
        for s in sentences[:10]:
            if sum(len(p) + 1 for p in parts) + len(s) + 1 <= max_chars:
                parts.append(s)

    out = clean_text(" ".join(parts))
    if len(out) > max_chars:
        cut = out[:max_chars]
        # avoid ending mid-token when possible
        end = max(cut.rfind("."), cut.rfind(";"), cut.rfind(")"))
        if end > int(max_chars * 0.65):
            out = cut[: end + 1]
        else:
            out = cut.rsplit(" ", 1)[0].rstrip() + "..."

    meta = {
        "raw_text_char_len": len(raw_text),
        "salient_text_char_len": len(out),
        "selection_method": "query_aware_salient_sentences_v1",
        "selected_sentence_count": len(top),
        "extracted_market_facts": facts,
        "source_text_was_cache_truncated": bool(fetched.get("text_cache_truncated")),
        "source_text_cache_char_len": int(fetched.get("text_cache_char_len") or len(raw_text)),
    }
    return out, meta


# Backward-compatible argument parser continues below.
def clean_query_candidate(q: Any) -> str:
    q = clean_text(q)
    if not q:
        return ""
    q = q.strip().strip("` ")
    q = re.sub(r"^[-•*\s]+", "", q)
    q = re.sub(r"^\d+[\.)]\s*", "", q)
    q = q.strip().strip(",").strip("[]{}").strip("\"'")
    q = re.sub(r"^queries\s*[:=]\s*", "", q, flags=re.I).strip()
    q = q.strip().strip(",").strip("[]{}").strip("\"'")
    q = clean_text(q)
    q_norm = normalize_key(q)
    if not q or q_norm in {"queries", "query", "q", "json", "query list"}:
        return ""
    if re.fullmatch(r"[\"']?queries[\"']?\s*:?\s*\[?", q, flags=re.I):
        return ""
    if len(q.split()) < 2:
        return ""
    if not re.search(r"[A-Za-z0-9]", q):
        return ""
    return q[:180]


def extract_json_from_text(text: str):
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


def parse_all_lens_queries(text: str) -> Dict[str, List[str]]:
    obj = extract_json_from_text(text)
    out = {k: [] for k in BUSINESS_LENSES}
    if isinstance(obj, dict):
        raw = obj.get("queries_by_lens", obj)
        if isinstance(raw, dict):
            for lens in BUSINESS_LENSES:
                vals = raw.get(lens, [])
                if isinstance(vals, dict):
                    vals = vals.get("queries", [])
                if isinstance(vals, str):
                    vals = [vals]
                if isinstance(vals, list):
                    for q in vals:
                        if isinstance(q, dict):
                            q = q.get("query") or q.get("q")
                        q = clean_query_candidate(q)
                        if q:
                            out[lens].append(q)
    for lens in out:
        seen, clean = set(), []
        for q in out[lens]:
            k = normalize_key(q)
            if k not in seen:
                clean.append(q)
                seen.add(k)
            if len(clean) >= 5:
                break
        out[lens] = clean
    return out


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


def summarize_visual_annotation(row: Dict[str, Any]) -> str:
    va = row.get("visual_annotation")
    if isinstance(va, dict):
        return json.dumps(va, ensure_ascii=False)[:1200]
    return clean_text(va)[:1200]


def fallback_queries(row: Dict[str, Any], lens_key: str) -> List[str]:
    domain = clean_text(row.get("domain"))
    hint = DOMAIN_HINTS.get(domain, domain)
    if lens_key == "cost_resource_efficiency":
        return [
            f"{hint} cost reduction market solution",
            f"{hint} resource efficiency product case study",
            f"{hint} energy material savings industry report",
            f"{hint} waste reduction business solution",
            f"{hint} sustainability ROI operations case study",
        ]
    if lens_key == "intelligent_automation":
        return [
            f"{hint} AI automation software platform",
            f"{hint} computer vision analytics market solution",
            f"{hint} sensor monitoring technology case study",
            f"{hint} predictive intelligence startup product",
            f"{hint} automation decision support industry report",
        ]
    return [
        f"{hint} customer experience improvement case study",
        f"{hint} operational efficiency market solution",
        f"{hint} service workflow optimization platform",
        f"{hint} conversion retention customer journey report",
        f"{hint} productivity throughput management software",
    ]


class OpenAIAllLensQueryGenerator:
    def __init__(self, model: str, api_key: str, image_detail: str, timeout: float, max_retries: int, max_image_side: int):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.image_detail = image_detail
        self.max_retries = max_retries
        self.max_image_side = max_image_side

    def generate(self, image_path: Path, row: Dict[str, Any]) -> Tuple[str, Dict[str, List[str]]]:
        domain = clean_text(row.get("domain"))
        domain_display = clean_text(row.get("domain_display_name")) or domain
        source_dataset = clean_text(row.get("source_dataset"))
        visual_annotation = summarize_visual_annotation(row)
        domain_question = DOMAIN_QUESTIONS.get(domain, "")
        lens_block = "\n".join([f"- {k}: {v['display_name']} ({v['description']})" for k, v in BUSINESS_LENSES.items()])
        prompt = f"""
You are preparing strict query-level real-world market evidence for AIStarBench startup ideation.

Benchmark structure:
- For each image, there are 3 business lenses.
- For each lens, generate exactly 5 focused market-search queries.
- Later, each query will retrieve exactly one validated DuckDuckGo evidence page.
- Later, each query/evidence pair will produce exactly one startup idea.

Image/domain context:
- domain: {domain_display}
- source dataset: {source_dataset}
- domain hint: {DOMAIN_HINTS.get(domain, domain)}
- domain-specific ideation question: {domain_question}
- visual metadata: {visual_annotation}

Business lenses:
{lens_block}

Task:
Generate exactly 5 focused DuckDuckGo search queries for EACH business lens.
Each query must represent a DIFFERENT market angle and must be suitable for retrieving real market/product/solution/report/case-study evidence.

Rules:
- Concrete product/service/industry/report/case-study queries only.
- Avoid generic "startup idea" wording.
- Do not include "queries:" as a query.
- Output valid JSON only.

Format:
{{
  "queries_by_lens": {{
    "cost_resource_efficiency": ["Q1", "Q2", "Q3", "Q4", "Q5"],
    "intelligent_automation": ["Q1", "Q2", "Q3", "Q4", "Q5"],
    "customer_operations_growth": ["Q1", "Q2", "Q3", "Q4", "Q5"]
  }}
}}
""".strip()

        data_url = image_to_data_url(image_path, max_side=self.max_image_side)
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": data_url, "detail": self.image_detail},
                        ],
                    }],
                    temperature=0.2,
                    top_p=0.9,
                    max_output_tokens=900,
                )
                text = extract_response_text(resp)
                parsed = parse_all_lens_queries(text)
                for lens in BUSINESS_LENSES:
                    if len(parsed.get(lens, [])) < 5:
                        merged, seen = [], set()
                        for q in parsed.get(lens, []) + fallback_queries(row, lens):
                            q = clean_query_candidate(q)
                            k = normalize_key(q)
                            if q and k not in seen:
                                merged.append(q)
                                seen.add(k)
                            if len(merged) >= 5:
                                break
                        parsed[lens] = merged
                return text, parsed
            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                wait = 2.0 * (2 ** attempt) + random.random()
                print(f"[OpenAI query retry] {attempt+1}/{self.max_retries} error={repr(e)} sleep={wait:.1f}s")
                time.sleep(wait)
        raise RuntimeError(f"OpenAI query generation failed: {repr(last_err)}")


class DDGStrictFiveQueryEvidenceFetcher:
    def __init__(
        self,
        cache_dir: str,
        max_results_per_query: int,
        max_variants_per_query: int,
        max_fetch_candidates_per_query: int,
        fetch_workers: int,
        timeout: float,
        ddg_sleep: float,
        fetch_sleep: float,
        max_chars_per_page: int,
        user_agent: str,
        enable_ddgs_fallback: bool,
    ):
        self.cache_dir = Path(cache_dir)
        self.search_cache = self.cache_dir / "ddg_search_marketdb"
        self.fetch_cache = self.cache_dir / "url_fetch_marketdb"
        self.search_cache.mkdir(parents=True, exist_ok=True)
        self.fetch_cache.mkdir(parents=True, exist_ok=True)
        self.max_results_per_query = int(max_results_per_query)
        self.max_variants_per_query = int(max_variants_per_query)
        self.max_fetch_candidates_per_query = int(max_fetch_candidates_per_query)
        self.fetch_workers = max(1, int(fetch_workers))
        self.timeout = float(timeout)
        self.ddg_sleep = float(ddg_sleep)
        self.fetch_sleep = float(fetch_sleep)
        self.max_chars_per_page = int(max_chars_per_page)
        self.user_agent = user_agent
        self.enable_ddgs_fallback = bool(enable_ddgs_fallback)

    def query_variants(self, query: str, row: Dict[str, Any], lens_key: str) -> List[str]:
        q = clean_query_candidate(query)
        domain = clean_text(row.get("domain"))
        hint = DOMAIN_HINTS.get(domain, domain)
        suffixes = ["", "market solution", "case study", "industry report", "software platform"]
        variants, seen = [], set()
        for s in suffixes:
            v = clean_query_candidate(f"{q} {s}".strip())
            k = normalize_key(v)
            if v and k not in seen:
                variants.append(v)
                seen.add(k)
            if len(variants) >= self.max_variants_per_query:
                return variants
        v = clean_query_candidate(f"{hint} {q} market evidence")
        if v and normalize_key(v) not in seen and len(variants) < self.max_variants_per_query:
            variants.append(v)
        return variants[: self.max_variants_per_query]

    def unwrap_ddg_href(self, href: str) -> str:
        href = clean_text(href)
        if not href:
            return ""
        if href.startswith("//"):
            href = "https:" + href
        if href.startswith("/"):
            href = "https://duckduckgo.com" + href
        try:
            parsed = urlparse(href)
            qs = parse_qs(parsed.query)
            if qs.get("uddg"):
                return unquote(qs["uddg"][0])
        except Exception:
            pass
        return href

    def search_direct_html(self, query: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        import httpx
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://duckduckgo.com/",
        }
        endpoints = [
            ("post", "https://html.duckduckgo.com/html/", {"q": query, "kl": "us-en"}),
            ("get", "https://lite.duckduckgo.com/lite/", {"q": query}),
        ]
        results, seen, errors = [], set(), []
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True, headers=headers) as client:
                for method, endpoint, params in endpoints:
                    try:
                        resp = client.post(endpoint, data=params) if method == "post" else client.get(endpoint, params=params)
                        if not (200 <= resp.status_code < 400) or not resp.text:
                            errors.append(f"{endpoint} status={resp.status_code}")
                            continue
                        soup = BeautifulSoup(resp.text, "lxml")
                        anchors = []
                        for sel in ["a.result__a", "a.result-link", ".result__title a", "a[href*='uddg=']"]:
                            anchors.extend(soup.select(sel))
                        if not anchors:
                            anchors.extend([a for a in soup.find_all("a", href=True)])
                        for a in anchors:
                            title = clean_text(a.get_text(" "))
                            url = self.unwrap_ddg_href(a.get("href", ""))
                            if not title or not url or should_skip_url(url):
                                continue
                            if any(t in title.lower() for t in BAD_TITLE_TERMS):
                                continue
                            if domain_of_url(url) in {"duckduckgo.com", "html.duckduckgo.com", "lite.duckduckgo.com"}:
                                continue
                            k = normalize_key(url)
                            if not k or k in seen:
                                continue
                            seen.add(k)
                            results.append({
                                "query": query,
                                "title": title,
                                "url": url,
                                "domain": domain_of_url(url),
                                "snippet": "",
                                "search_backend": f"direct:{endpoint}",
                            })
                            if len(results) >= self.max_results_per_query:
                                return results, errors
                    except Exception as e:
                        errors.append(f"{endpoint} error={repr(e)}")
        except Exception as e:
            errors.append(f"httpx error={repr(e)}")
        return results, errors

    def search_ddgs_html_lite(self, query: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        if not self.enable_ddgs_fallback:
            return [], ["ddgs fallback disabled"]
        errors, results, seen = [], [], set()
        classes = []
        try:
            from ddgs import DDGS as DDGS1
            classes.append(("ddgs", DDGS1))
        except Exception as e:
            errors.append(f"import ddgs failed: {repr(e)}")
            # Legacy fallback only when the new package is unavailable.
            # Importing duckduckgo_search when ddgs exists emits repeated RuntimeWarning noise.
            try:
                from duckduckgo_search import DDGS as DDGS2
                classes.append(("duckduckgo_search", DDGS2))
            except Exception as e2:
                errors.append(f"import duckduckgo_search failed: {repr(e2)}")
        for pkg_name, DDGS in classes:
            for backend in ["lite", "html"]:
                try:
                    try:
                        ctx = DDGS(timeout=int(self.timeout))
                    except TypeError:
                        ctx = DDGS()
                    with ctx as ddgs:
                        try:
                            raw = ddgs.text(query, max_results=self.max_results_per_query, backend=backend)
                        except TypeError:
                            raw = ddgs.text(query, max_results=self.max_results_per_query)
                        for item in raw or []:
                            title = clean_text(item.get("title"))
                            url = clean_text(item.get("href") or item.get("url"))
                            snippet = clean_text(item.get("body") or item.get("snippet"))
                            if not title or not url or should_skip_url(url):
                                continue
                            if any(t in title.lower() for t in BAD_TITLE_TERMS):
                                continue
                            k = normalize_key(url)
                            if not k or k in seen:
                                continue
                            seen.add(k)
                            results.append({
                                "query": query,
                                "title": title,
                                "url": url,
                                "domain": domain_of_url(url),
                                "snippet": snippet,
                                "search_backend": f"{pkg_name}:{backend}",
                            })
                            if len(results) >= self.max_results_per_query:
                                return results, errors
                except Exception as e:
                    errors.append(f"{pkg_name}:{backend} error={repr(e)}")
        return results, errors

    def search(self, query: str) -> List[Dict[str, Any]]:
        query = clean_query_candidate(query)
        if not query:
            return []
        cache_path = self.search_cache / f"{stable_hash('search::' + query)}.json"
        cached = load_json(cache_path)
        if isinstance(cached, list):
            return cached

        results, errors = self.search_direct_html(query)
        if not results:
            more, more_errors = self.search_ddgs_html_lite(query)
            results, errors = more, errors + more_errors

        safe_json_dump(cache_path, results)
        if self.ddg_sleep:
            time.sleep(self.ddg_sleep)
        return results

    def fetch_url(self, url: str) -> Dict[str, Any]:
        cache_path = self.fetch_cache / f"{stable_hash(url)}.json"
        cached = load_json(cache_path)
        if isinstance(cached, dict):
            return cached
        try:
            import httpx
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://duckduckgo.com/",
            }
            with httpx.Client(timeout=self.timeout, follow_redirects=True, headers=headers) as client:
                resp = client.get(url)
            content_type = resp.headers.get("content-type", "")
            html = resp.text or ""
            status = int(resp.status_code)
            text = ""
            if html and ("text/html" in content_type.lower() or "<html" in html[:500].lower()):
                try:
                    import trafilatura
                    text = trafilatura.extract(html, url=str(resp.url), include_comments=False, include_tables=True, favor_recall=True) or ""
                except Exception:
                    text = ""
                if not text:
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(html, "lxml")
                        for tag in soup(["script", "style", "noscript", "svg"]):
                            tag.decompose()
                        text = soup.get_text(" ")
                    except Exception:
                        text = ""
            raw_text = clean_text(text)
            # Keep enough page text for query-aware salient extraction. The final MarketDB card
            # will store a compact evidence block, not this raw prefix.
            cache_limit = max(12000, self.max_chars_per_page * 8)
            out = {
                "url": url,
                "final_url": str(resp.url),
                "fetch_ok": 200 <= status < 400,
                "status_code": status,
                "content_type": content_type,
                "text": raw_text[:cache_limit],
                "raw_text_char_len": len(raw_text),
                "text_cache_char_len": min(len(raw_text), cache_limit),
                "text_cache_truncated": len(raw_text) > cache_limit,
                "error": "",
            }
        except Exception as e:
            out = {"url": url, "final_url": url, "fetch_ok": False, "status_code": None, "content_type": "", "text": "", "error": repr(e)}
        safe_json_dump(cache_path, out)
        if self.fetch_sleep:
            time.sleep(self.fetch_sleep)
        return out

    def validate_page(self, hit: Dict[str, Any], fetched: Dict[str, Any], core_query: str = "") -> Tuple[bool, str, float, List[str]]:
        url = clean_text(fetched.get("final_url") or hit.get("url"))
        title = clean_text(hit.get("title"))
        text = clean_text(fetched.get("text"))
        status = fetched.get("status_code")
        ok = bool(fetched.get("fetch_ok"))
        if should_skip_url(url):
            return False, "skip_domain", 0.0, []
        if not ok or not isinstance(status, int) or not (200 <= status < 400):
            return False, f"bad_status_{status}", 0.0, []
        blob = " ".join([title, url, clean_text(hit.get("snippet")), text[:1800]]).lower()
        if any(t in title.lower() for t in BAD_TITLE_TERMS):
            return False, "bad_title", 0.0, []
        if any(t in blob for t in BAD_TEXT_TERMS):
            return False, "bad_text", 0.0, []
        if len(text) < 450:
            return False, "too_short", 0.0, []
        if is_generic_homepage(url, title, text):
            return False, "generic_homepage", 0.0, []
        if not any(t in blob for t in MARKET_SIGNAL_TERMS):
            return False, "no_market_signal", 0.0, []
        score, reasons = evidence_relevance_score(core_query, hit, fetched)
        if score < 22.0:
            return False, f"low_relevance_{score}", score, reasons
        return True, "valid", score, reasons

    def one_card_for_query(self, core_query: str, query_index: int, row: Dict[str, Any], lens_key: str, used_urls: set) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        all_results, filtered = [], []

        for retrieval_query in self.query_variants(core_query, row, lens_key):
            hits = self.search(retrieval_query)
            candidates = []

            for rank, h in enumerate(hits, 1):
                h = dict(h)
                h["rank_within_query"] = rank
                h["source_query"] = core_query
                h["source_query_index"] = query_index
                all_results.append(h)

                url = h.get("url", "")
                key = normalize_key(url)
                if not url or key in used_urls:
                    continue

                candidates.append(h)
                if len(candidates) >= self.max_fetch_candidates_per_query:
                    break

            if not candidates:
                continue

            # Fast policy: take the first valid accessible/relevant URL in DDG rank order.
            # This avoids fetching every candidate just to pick the highest score.
            # If early candidates are invalid, continue until one valid W_i is found.
            for h in candidates:
                fetched = self.fetch_url(h["url"])
                ok, reason, score, reasons = self.validate_page(h, fetched, core_query)
                if not ok:
                    filtered.append({
                        "query": h.get("query"),
                        "title": h.get("title"),
                        "url": h.get("url"),
                        "domain": h.get("domain"),
                        "status_code": fetched.get("status_code"),
                        "reason": reason,
                        "evidence_relevance_score": score,
                    })
                    continue

                used_urls.add(normalize_key(h["url"]))
                salient_text, salient_meta = build_salient_market_text(
                    core_query=core_query,
                    hit=h,
                    fetched=fetched,
                    max_chars=self.max_chars_per_page,
                )
                card = {
                    "evidence_id": f"W{query_index}",
                    "source_query_index": query_index,
                    "source_query": core_query,
                    "title": h.get("title"),
                    "domain": h.get("domain") or domain_of_url(h.get("url", "")),
                    "url": fetched.get("final_url") or h.get("url"),
                    # text is now a compact, query-aware evidence block, not an arbitrary page prefix.
                    "text": salient_text,
                    "evidence_text_mode": salient_meta.get("selection_method"),
                    "raw_text_char_len": salient_meta.get("raw_text_char_len"),
                    "source_text_cache_char_len": salient_meta.get("source_text_cache_char_len"),
                    "source_text_was_cache_truncated": salient_meta.get("source_text_was_cache_truncated"),
                    "extracted_market_facts": salient_meta.get("extracted_market_facts", []),
                    "selected_sentence_count": salient_meta.get("selected_sentence_count"),
                }
                return all_results, card

        preview = "; ".join([f"{x.get('reason')}:{x.get('domain')}" for x in filtered[:8]])
        raise RuntimeError(
            f"Could not find valid DDG evidence for source_query_index={query_index} query={core_query!r}. "
            f"filtered={len(filtered)} preview={preview}"
        )

    def retrieve_five_query_cards(self, core_queries: List[str], row: Dict[str, Any], lens_key: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        merged, seen = [], set()
        for q in core_queries + fallback_queries(row, lens_key):
            q = clean_query_candidate(q)
            k = normalize_key(q)
            if q and k not in seen:
                merged.append(q)
                seen.add(k)
            if len(merged) >= 5:
                break
        if len(merged) != 5:
            raise RuntimeError(f"Expected exactly 5 core queries for lens={lens_key}, got {len(merged)}")

        raw_results, cards, used_urls = [], [], set()
        for i, q in enumerate(merged, 1):
            rr, card = self.one_card_for_query(q, i, row, lens_key, used_urls)
            raw_results.extend(rr)
            cards.append(card)
        return raw_results, cards, merged


def validate_output_file(path: Path, expected_rows: int):
    rows = read_jsonl(str(path))
    bad = []
    if expected_rows and len(rows) != expected_rows:
        bad.append(f"row_count {len(rows)} != {expected_rows}")
    seen = set()
    card_count = 0
    for i, r in enumerate(rows, 1):
        key = (r.get("sample_id"), r.get("business_lens"))
        if key in seen:
            bad.append(f"duplicate key {key}")
        seen.add(key)

        queries = r.get("core_retrieval_queries", []) or []
        cards = r.get("web_evidence_cards", []) or []
        if len(queries) != 5:
            bad.append(f"line {i} core_queries {len(queries)} != 5")
        if len(cards) != 5:
            bad.append(f"line {i} evidence_cards {len(cards)} != 5")

        qidx = {}
        for c in cards:
            card_count += 1
            idx = c.get("source_query_index")
            qidx[idx] = qidx.get(idx, 0) + 1
            blob = " ".join([str(c.get("title","")), str(c.get("url","")), str(c.get("text",""))[:1500]]).lower()
            if not c.get("url") or not c.get("text"):
                bad.append(f"line {i} empty url/text")
            if any(t in blob for t in BAD_TEXT_TERMS + BAD_TITLE_TERMS):
                bad.append(f"line {i} bad term in evidence {c.get('title')}")
            try:
                qi = int(c.get("source_query_index"))
            except Exception:
                bad.append(f"line {i} bad source_query_index {c.get('source_query_index')}")
                continue
            if qi < 1 or qi > 5:
                bad.append(f"line {i} source_query_index out of range {qi}")
            if c.get("evidence_id") != f"W{qi}":
                bad.append(f"line {i} evidence_id {c.get('evidence_id')} != W{qi}")
            if c.get("source_query") != queries[qi - 1]:
                bad.append(f"line {i} source_query mismatch for W{qi}")

        for qi in range(1, 6):
            if qidx.get(qi, 0) != 1:
                bad.append(f"line {i} source_query_index={qi} count {qidx.get(qi, 0)} != 1")
    print(f"[VALIDATE] evidence rows={len(rows)} unique={len(seen)} cards={card_count} expected_cards_per_lens=5 issues={len(bad)}")
    if bad:
        for b in bad[:30]:
            print("BAD:", b)
        raise SystemExit(1)
    print("[VALIDATE] strict 5-query DDG market evidence prefetch PASS")


def is_complete_evidence_row(r: Dict[str, Any]) -> bool:
    try:
        sample_id = clean_text(r.get("sample_id"))
        lens = clean_text(r.get("business_lens"))
        queries = r.get("core_retrieval_queries", []) or []
        cards = r.get("web_evidence_cards", []) or []
        if not sample_id or lens not in BUSINESS_LENSES:
            return False
        if len(queries) != 5 or len(cards) != 5:
            return False
        by_idx = {}
        for c in cards:
            i = int(c.get("source_query_index"))
            by_idx[i] = c
        for i in range(1, 6):
            c = by_idx.get(i)
            if not c:
                return False
            if c.get("evidence_id") != f"W{i}":
                return False
            if c.get("source_query") != queries[i - 1]:
                return False
            if not c.get("url") or not c.get("text"):
                return False
        return True
    except Exception:
        return False


def load_completed_evidence_rows(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    completed: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not path.exists():
        return completed
    try:
        for r in read_jsonl(str(path)):
            if not is_complete_evidence_row(r):
                continue
            key = (clean_text(r.get("sample_id")), clean_text(r.get("business_lens")))
            completed[key] = r
    except Exception:
        return completed
    return completed


def rewrite_completed_evidence_rows(path: Path, completed: Dict[Tuple[str, str], Dict[str, Any]], manifest_rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = []
    for row in manifest_rows:
        sid = clean_text(row.get("sample_id"))
        for lens in BUSINESS_LENSES:
            key = (sid, lens)
            if key in completed:
                ordered.append(completed[key])
    tmp = path.with_suffix(path.suffix + ".resume_tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ddg_prefetch", "validate_ddg"], default="ddg_prefetch")
    p.add_argument("--source_manifest_jsonl")
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--expected_rows", type=int, default=0)
    p.add_argument("--fallback_base_dir", default="/workspace")
    p.add_argument("--web_cache_dir", default="/workspace/data/aistar/web_cache")
    p.add_argument("--max_ddg_results_per_query", type=int, default=6)
    p.add_argument("--max_variants_per_query", type=int, default=4)
    p.add_argument("--max_fetch_candidates_per_query", type=int, default=4)
    p.add_argument("--fetch_workers", type=int, default=1)
    p.add_argument("--web_timeout", type=float, default=10.0)
    p.add_argument("--ddg_sleep", type=float, default=0.1)
    p.add_argument("--fetch_sleep", type=float, default=0.0)
    p.add_argument("--max_chars_per_page", type=int, default=4000)
    p.add_argument("--openai_model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"))
    p.add_argument("--openai_api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--openai_image_detail", default=os.environ.get("OPENAI_IMAGE_DETAIL", "low"))
    p.add_argument("--max_image_side", type=int, default=768)
    p.add_argument("--max_retries", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--disable_ddgs_fallback", action="store_true")
    p.add_argument("--continue_on_image_error", action="store_true", help="Log failed images and continue accumulating completed JSONL rows.")
    p.add_argument("--retry_failed_until_complete", action="store_true", help="Keep retrying failed images inside the same process until the DDG evidence file is complete.")
    p.add_argument("--max_image_attempts", type=int, default=0, help="Maximum attempts per image when --retry_failed_until_complete is set. 0 means retry indefinitely.")
    p.add_argument("--retry_sleep", type=float, default=10.0, help="Initial sleep seconds before retrying failed images in the same run.")
    p.add_argument("--retry_sleep_backoff", type=float, default=1.5, help="Backoff multiplier for retry sleep between failed-image rounds.")
    p.add_argument("--retry_max_sleep", type=float, default=120.0, help="Maximum sleep seconds between retry rounds.")
    p.add_argument("--retry_workers", type=int, default=0, help="Worker count for retry rounds. 0 uses min(image_workers, remaining failed images).")
    p.add_argument("--failed_jsonl", default="", help="Path for failed image/lens jobs. Defaults to <output_jsonl>.failed.jsonl")
    p.add_argument("--image_workers", type=int, default=1, help="Parallel image-level workers for DDG/query preparation.")
    p.add_argument("--resume", action="store_true", help="Keep complete existing evidence rows and process only missing images.")
    p.add_argument("--overwrite", action="store_true", help="Delete existing output_jsonl and rebuild from scratch.")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    if args.mode == "validate_ddg":
        validate_output_file(Path(args.output_jsonl), args.expected_rows)
        return

    if not args.source_manifest_jsonl:
        raise ValueError("--source_manifest_jsonl is required for ddg_prefetch")
    if not args.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    rows = read_jsonl(args.source_manifest_jsonl, limit=args.limit)
    expected_rows = len(rows) * len(BUSINESS_LENSES)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and output_jsonl.exists():
        output_jsonl.unlink()

    completed = load_completed_evidence_rows(output_jsonl) if args.resume else {}
    if completed:
        rewrite_completed_evidence_rows(output_jsonl, completed, rows)

    rows_to_process = []
    for idx, row in enumerate(rows, 1):
        sid = clean_text(row.get("sample_id"))
        if args.resume and all((sid, lens) in completed for lens in BUSINESS_LENSES):
            continue
        rows_to_process.append((idx, row))

    if not output_jsonl.exists():
        output_jsonl.touch()

    print(f"[DDG START] images={len(rows)} remaining={len(rows_to_process)} completed_rows={len(completed)}/{expected_rows} workers={args.image_workers} output={output_jsonl}")

    def process_one(idx_row: Tuple[int, Dict[str, Any]]) -> Tuple[int, str, float, List[Dict[str, Any]]]:
        idx, row = idx_row
        image_start = time.time()
        sample_id = clean_text(row.get("sample_id"))
        image_path = resolve_image_path(row, args.fallback_base_dir)

        qgen = OpenAIAllLensQueryGenerator(
            model=args.openai_model,
            api_key=args.openai_api_key,
            image_detail=args.openai_image_detail,
            timeout=120.0,
            max_retries=args.max_retries,
            max_image_side=args.max_image_side,
        )
        fetcher = DDGStrictFiveQueryEvidenceFetcher(
            cache_dir=args.web_cache_dir,
            max_results_per_query=args.max_ddg_results_per_query,
            max_variants_per_query=args.max_variants_per_query,
            max_fetch_candidates_per_query=args.max_fetch_candidates_per_query,
            fetch_workers=args.fetch_workers,
            timeout=args.web_timeout,
            ddg_sleep=args.ddg_sleep,
            fetch_sleep=args.fetch_sleep,
            max_chars_per_page=args.max_chars_per_page,
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36 AIStarMarketDB/1.0",
            enable_ddgs_fallback=not args.disable_ddgs_fallback,
        )

        raw_query_text, queries_by_lens = qgen.generate(image_path, row)
        records = []

        for lens_key in BUSINESS_LENSES:
            ddg_results, cards, core_queries = fetcher.retrieve_five_query_cards(
                queries_by_lens.get(lens_key, []),
                row,
                lens_key,
            )
            rec = {
                "schema_version": "aistar_marketdb",
                "sample_id": sample_id,
                "domain": row.get("domain"),
                "source_dataset": row.get("source_dataset"),
                "image_path": row.get("image_path"),
                "domain_question": DOMAIN_QUESTIONS.get(row.get("domain"), ""),
                "business_lens": lens_key,
                "core_retrieval_queries": core_queries,
                "web_evidence_cards": cards,
            }
            records.append(rec)

        return idx, sample_id, time.time() - image_start, records

    written = len(completed)
    t0 = time.time()
    failed_count = 0
    failed_jsonl = Path(args.failed_jsonl) if args.failed_jsonl else output_jsonl.with_suffix(output_jsonl.suffix + ".failed.jsonl")
    attempt_counts: Dict[str, int] = {}

    def log_image_failure(idx_row: Tuple[int, Dict[str, Any]], exc: BaseException, attempt: int, will_retry: bool):
        idx, row = idx_row
        rec = {
            "idx": idx,
            "sample_id": clean_text(row.get("sample_id")),
            "domain": row.get("domain"),
            "source_dataset": row.get("source_dataset"),
            "image_path": row.get("image_path"),
            "attempt": attempt,
            "will_retry": bool(will_retry),
            "error_type": type(exc).__name__,
            "error": repr(exc),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        failed_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with failed_jsonl.open("a", encoding="utf-8") as ef:
            ef.write(json.dumps(rec, ensure_ascii=False) + "\n")
        suffix = "retrying" if will_retry else "giving_up"
        # tqdm.write(f"[DDG FAIL] idx={idx} sample={rec['sample_id']} attempt={attempt} {suffix} error={rec['error_type']}: {rec['error'][:240]}")

    def write_success_records(out, records: List[Dict[str, Any]], sample_id: str, elapsed: float):
        nonlocal written
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
        out.flush()
        done_images = written // len(BUSINESS_LENSES)
        # tqdm.write(f"[DDG IMAGE] {done_images}/{len(rows)} sample={sample_id} rows={len(records)} sec={elapsed:.1f}")

    def run_batch(batch_rows: List[Tuple[int, Dict[str, Any]]], workers: int, round_no: int) -> List[Tuple[int, Dict[str, Any]]]:
        nonlocal failed_count
        failed_rows: List[Tuple[int, Dict[str, Any]]] = []
        desc = "DDG" if round_no == 1 else f"DDG retry{round_no}"
        workers = max(1, min(int(workers), len(batch_rows))) if batch_rows else 1
        with output_jsonl.open("a", encoding="utf-8") as out:
            if workers <= 1:
                pbar = tqdm(batch_rows, desc=desc, unit="img", dynamic_ncols=True)
                for idx_row in pbar:
                    sid = clean_text(idx_row[1].get("sample_id"))
                    attempt_counts[sid] = attempt_counts.get(sid, 0) + 1
                    attempt = attempt_counts[sid]
                    try:
                        idx, sample_id, elapsed, records = process_one(idx_row)
                    except Exception as e:
                        maxed = bool(args.max_image_attempts and attempt >= args.max_image_attempts)
                        will_retry = bool(args.retry_failed_until_complete and not maxed)
                        failed_count += 1
                        log_image_failure(idx_row, e, attempt, will_retry)
                        if will_retry:
                            failed_rows.append(idx_row)
                            continue
                        if args.continue_on_image_error:
                            continue
                        raise
                    write_success_records(out, records, sample_id, elapsed)
                    done_images = written // len(BUSINESS_LENSES)
                    pbar.set_postfix(done=f"{done_images}/{len(rows)}", rows=f"{written}/{expected_rows}")
            else:
                with futures.ThreadPoolExecutor(max_workers=workers) as ex:
                    future_map = {}
                    for idx_row in batch_rows:
                        sid = clean_text(idx_row[1].get("sample_id"))
                        attempt_counts[sid] = attempt_counts.get(sid, 0) + 1
                        future_map[ex.submit(process_one, idx_row)] = idx_row
                    pbar = tqdm(futures.as_completed(future_map), total=len(future_map), desc=desc, unit="img", dynamic_ncols=True)
                    for fut in pbar:
                        idx_row = future_map[fut]
                        sid = clean_text(idx_row[1].get("sample_id"))
                        attempt = attempt_counts[sid]
                        try:
                            idx, sample_id, elapsed, records = fut.result()
                        except Exception as e:
                            maxed = bool(args.max_image_attempts and attempt >= args.max_image_attempts)
                            will_retry = bool(args.retry_failed_until_complete and not maxed)
                            failed_count += 1
                            log_image_failure(idx_row, e, attempt, will_retry)
                            if will_retry:
                                failed_rows.append(idx_row)
                                continue
                            if args.continue_on_image_error:
                                continue
                            raise
                        write_success_records(out, records, sample_id, elapsed)
                        done_images = written // len(BUSINESS_LENSES)
                        pbar.set_postfix(done=f"{done_images}/{len(rows)}", rows=f"{written}/{expected_rows}")
        return failed_rows

    if not rows_to_process:
        print("[PREPARE] nothing to process; all evidence rows already complete")
    else:
        retry_round = 1
        pending_rows = rows_to_process
        sleep_s = max(0.0, float(args.retry_sleep))
        while pending_rows:
            if retry_round == 1:
                workers = args.image_workers
            elif args.retry_workers > 0:
                workers = args.retry_workers
            else:
                workers = min(max(1, int(args.image_workers)), len(pending_rows))
            print(f"[DDG ROUND] round={retry_round} pending_images={len(pending_rows)} workers={workers} written_rows={written}/{expected_rows}")
            failed_rows = run_batch(pending_rows, workers, retry_round)
            if not failed_rows:
                break
            if not args.retry_failed_until_complete:
                pending_rows = failed_rows
                break
            pending_rows = failed_rows
            if args.max_image_attempts and all(attempt_counts.get(clean_text(r[1].get("sample_id")), 0) >= args.max_image_attempts for r in pending_rows):
                print(f"[DDG INCOMPLETE] {len(pending_rows)} images reached max_image_attempts={args.max_image_attempts}; failures logged to {failed_jsonl}")
                validate_output_file(output_jsonl, 0)
                raise SystemExit(2)
            print(f"[DDG RETRY] failed_images={len(pending_rows)} sleep={sleep_s:.1f}s before retrying inside the same run")
            if sleep_s > 0:
                time.sleep(sleep_s)
            sleep_s = min(float(args.retry_max_sleep), max(0.0, sleep_s * float(args.retry_sleep_backoff)))
            retry_round += 1

    if failed_count and not args.retry_failed_until_complete:
        print(f"[DDG INCOMPLETE] failed_images={failed_count}; completed rows were kept in {output_jsonl}; failures logged to {failed_jsonl}; rerun with --resume to retry failures.")
        validate_output_file(output_jsonl, 0)
        raise SystemExit(2)

    validate_output_file(output_jsonl, expected_rows)
    print(f"[DDG DONE] total_seconds={time.time()-t0:.1f}")


if __name__ == "__main__":
    main()
