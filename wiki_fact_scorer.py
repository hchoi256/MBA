"""
WikiFactScorer (final)
- atom 분해 + verifier 판정 + 점수 집계: FActScore의 채점 로직을 따름
- 검색만 open-domain (위키 전체 FTS5 BM25) — FActScore의 topic 고정 검색과 다른 부분
- scoring_mode 로 두 채점 정책 중 선택:
    "factscore"     : supported / (검증된 atom 수)   ← FActScore와 동일한 방식 (기본)
    "contradiction" : 1 - contradicted / checkable    ← 환각(모순)만 처벌 (창의성 비처벌)
"""

import re
import sqlite3
import numpy as np
import torch
from typing import List


# ── claim filter (제안/명령형 atom 제외) ────────────────────
_PROPOSAL_PATTERNS = re.compile(
    r"\b(should|could|would|can|may|might|must|let'?s|aim|goal|enable|allow"
    r"|provide|offer|help|support|create|build|develop|design|deploy"
    r"|implement|integrate|launch|introduce|explore|consider"
    r"|recommend|suggest|propose|plan)\b", re.IGNORECASE)
_FACTUAL_TRIGGERS = re.compile(
    r"\b(\d[\d,\.%]+|percent|billion|million|trillion"
    r"|founded|established|located|headquartered|born|died"
    r"|is|are|was|were|has|have|had|contains|consists|comprises)\b", re.IGNORECASE)

def is_checkable(atom: str) -> bool:
    if _PROPOSAL_PATTERNS.search(atom):
        return False
    if _FACTUAL_TRIGGERS.search(atom):
        return True
    return True


# ── FTS5 retriever (AND 우선 + 선택도 정렬 → 빠르고 정확) ───
_STOPWORDS = {
    "the","a","an","of","to","and","in","is","are","on","for","with",
    "that","this","it","be","as","at","by","we","or","from","which",
    "offer","offers","higher","lower","much","more","less","into","than",
    "has","have","had","can","will","would","should","may","use","uses",
    "using","provide","provides","make","makes","create","creates",
}

class FTS5Retriever:
    def __init__(self, fts_db_path: str):
        self.conn = sqlite3.connect(fts_db_path, check_same_thread=False)

    def _tokens(self, query: str):
        toks = [t for t in re.findall(r"\w+", query.lower())
                if t not in _STOPWORDS and len(t) > 2]
        seen, out = set(), []
        for t in toks:
            if t not in seen:
                seen.add(t); out.append(t)
        out.sort(key=lambda t: (any(c.isdigit() for c in t), len(t)), reverse=True)
        return out

    def _run(self, match_expr: str, k: int):
        try:
            return self.conn.execute(
                "SELECT title, para, bm25(passages_fts) AS s "
                "FROM passages_fts WHERE passages_fts MATCH ? "
                "ORDER BY s LIMIT ?", (match_expr, k)).fetchall()
        except sqlite3.OperationalError:
            return []

    def retrieve(self, query: str, k: int = 4):
        toks = self._tokens(query)
        if not toks:
            return []
        rows = []
        for n in (min(len(toks), 5), 4, 3, 2):
            if n < 1 or n > len(toks):
                continue
            rows = self._run(" AND ".join(toks[:n]), k)
            if len(rows) >= min(k, 3):
                break
        if not rows:
            rows = self._run(" OR ".join(toks[:3]), k)
        return [{"title": t, "text": p} for t, p, _ in rows]

    def close(self):
        self.conn.close()


# ── WikiFactScorer ──────────────────────────────────────────
class WikiFactScorer:
    def __init__(
        self,
        fts_db_path: str,
        af_generator,
        model,
        tokenizer,
        device: str = "cuda",
        retrieval_k: int = 4,
        verifier_batch_size: int = 16,
        max_checkable_atoms: int = 12,
        passage_char_cap: int = 350,
        scoring_mode: str = "factscore",   # "factscore" | "contradiction"
        debug_prob: float = 0.05,
    ):
        assert scoring_mode in ("factscore", "contradiction")
        self.retriever = FTS5Retriever(fts_db_path)
        self.af_gen    = af_generator
        self.model     = model
        self.tok       = tokenizer
        self.device    = device
        self.k         = retrieval_k
        self.bs        = verifier_batch_size
        self.max_atoms = max_checkable_atoms
        self.pchar     = passage_char_cap
        self.mode      = scoring_mode
        self.debug_prob = debug_prob
        try:
            self.tok.truncation_side = "left"
        except Exception:
            pass

    # ── atom 분해 ────────────────────────────────────────────
    def _get_atoms(self, generation: str) -> List[str]:
        if not generation.strip():
            return []
        try:
            result, _ = self.af_gen.run(generation)
            return [atom for _, facts in result for atom in facts]
        except Exception:
            return []

    # ── context + 프롬프트 (모드별) ─────────────────────────
    def _context(self, passages) -> str:
        ctx = ""
        for psg in reversed(passages):
            text = psg["text"].replace("<s>", "").replace("</s>", "").strip()
            ctx += "Title: {}\nText: {}\n\n".format(psg["title"], text[: self.pchar])
        return ctx.strip()

    def _build_prompt(self, atom: str, passages) -> str:
        ctx = self._context(passages)
        if self.mode == "factscore":
            return ("Answer the question based on the given context.\n\n" + ctx
                    + "\n\nInput: {} True or False?\nOutput:".format(atom.strip()))
        else:  # contradiction
            return ("Answer based on the given context.\n\n" + ctx
                    + "\n\nClaim: {}\nIs the claim Supported, Contradicted, or "
                      "Unknown given the context? Answer in one word.\nAnswer:".format(atom.strip()))

    # ── 배치 verifier → raw 디코드 텍스트 반환 ──────────────
    @torch.no_grad()
    def _verify_raw(self, prompts: List[str]) -> List[str]:
        outs = []
        for i in range(0, len(prompts), self.bs):
            chunk = prompts[i : i + self.bs]
            enc = self.tok(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024).to(self.device)
            gen = self.model.generate(
                **enc, max_new_tokens=6, do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
            new = gen[:, enc["input_ids"].shape[1]:]
            outs.extend(self.tok.batch_decode(new, skip_special_tokens=True))
        return outs

    @staticmethod
    def _parse_factscore(text: str) -> bool:
        t = text.lower()
        if "true" in t and "false" not in t:  return True
        if "false" in t and "true" not in t:  return False
        if "true" in t and "false" in t:      return t.index("true") > t.index("false")
        return True  # 판정 불가 → 중립(True)

    @staticmethod
    def _parse_contradiction(text: str) -> str:
        t = text.lower()
        if any(w in t for w in ("contradict","refut","false","incorrect","wrong")):
            return "contradicted"
        if any(w in t for w in ("support","true","correct","confirm","yes")):
            return "supported"
        return "unknown"

    # ── 단일 generation 채점 ────────────────────────────────
    def score_one(self, generation: str) -> float:
        atoms = [a for a in self._get_atoms(generation) if is_checkable(a)]
        atoms = atoms[: self.max_atoms]
        if not atoms:
            return float("nan")

        prompts, has_ev = [], []
        for atom in atoms:
            psgs = self.retriever.retrieve(atom, k=self.k)
            if psgs:
                prompts.append(self._build_prompt(atom, psgs)); has_ev.append(True)
            else:
                has_ev.append(False)

        raw = self._verify_raw(prompts) if prompts else []

        if self.mode == "factscore":
            verds = [self._parse_factscore(r) for r in raw]
            if not verds:
                return float("nan")
            score = float(np.mean(verds))
            # if np.random.rand() < self.debug_prob:
            #     ev_atoms = [a for a, e in zip(atoms, has_ev) if e]
            #     for a, v in zip(ev_atoms, verds):
            #         print(f"  [{'T' if v else 'F'}] {a[:75]}")
            return score
        else:  # contradiction
            it = iter(self._parse_contradiction(r) for r in raw)
            verds = [next(it) if e else "unknown" for e in has_ev]
            n = len(verds)
            n_con = sum(1 for v in verds if v == "contradicted")
            score = 1.0 - (n_con / n)
            # if np.random.rand() < self.debug_prob:
            #     tagmap = {"supported":"S","contradicted":"X","unknown":"?"}
            #     for a, v in zip(atoms, verds):
            #         print(f"  [{tagmap[v]}] {a[:75]}")
            return score

    # ── 배치 채점 (GRPO 진입점) ─────────────────────────────
    def score_batched(self, generations: List[str]) -> List[float]:
        raw = [self.score_one(g) for g in generations]
        valid = [s for s in raw if not (isinstance(s, float) and np.isnan(s))]
        fb = float(np.mean(valid)) if valid else (1.0 if self.mode == "contradiction" else 0.5)
        return [fb if (isinstance(s, float) and np.isnan(s)) else s for s in raw]