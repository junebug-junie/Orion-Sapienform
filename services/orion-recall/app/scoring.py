# app/scoring.py
from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional, Set

from .types import Fragment, RecallQuery, PhiContext


KIND_WEIGHTS: Dict[str, float] = {
    "collapse": 1.25,
    "enrichment": 1.15,
    "chat": 0.9,
    "association": 0.85,
    "biometrics": 0.7,
    "rdf": 1.0,
}


def _now_ts() -> float:
    return time.time()


def _norm_tags(tags: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for t in tags or []:
        s = str(t).strip().lower()
        if s:
            out.add(s)
    return out


def _tokenize(text: str) -> Set[str]:
    if not text:
        return set()
    return {
        token.strip(".,!?;:()[]{}\"'").lower()
        for token in text.split()
        if token.strip()
    }


def _base_salience(f: Fragment) -> float:
    s = float(f.salience or 0.0)
    v = abs(float(getattr(f, "valence", 0.0) or 0.0))
    a = float(getattr(f, "arousal", 0.0) or 0.0)

    s = max(0.0, min(s, 1.0))
    v = max(0.0, min(v, 1.0))
    a = max(0.0, min(a, 1.0))

    base = 0.2 + 0.4 * s + 0.2 * v + 0.2 * a
    return float(max(0.0, min(base, 1.2)))


def _recency_weight(f: Fragment, now_ts: float, half_life_days: float) -> float:
    if not f.ts:
        return 1.0

    age_sec = max(0.0, now_ts - float(f.ts))
    age_days = age_sec / 86400.0

    if half_life_days <= 0:
        return 1.0

    weight = 0.5 ** (age_days / half_life_days)
    return float(max(0.2, min(1.5, weight * 1.2)))


def _kind_weight(f: Fragment) -> float:
    return float(KIND_WEIGHTS.get(f.kind, 1.0))


def _query_alignment_weight(f: Fragment, q: RecallQuery) -> float:
    q_tokens = _tokenize(q.text)
    q_tags = _norm_tags(q.tags)

    f_tokens = _tokenize(f.text)
    f_tags = _norm_tags(f.tags)

    if not q_tokens and not q_tags:
        return 1.0

    tag_overlap = len(q_tags & f_tags)
    token_overlap = len(q_tokens & f_tokens)

    tag_boost = 1.0 + 0.18 * tag_overlap
    tok_boost = 1.0 + 0.08 * min(token_overlap, 10)

    weight = tag_boost * tok_boost
    return float(max(0.8, min(1.8, weight)))


def _phi_alignment_weight(f: Fragment, phi: Optional[PhiContext]) -> float:
    if phi is None:
        return 1.0

    try:
        m = f.meta or {}
        fv = float(m.get("spark_phi_valence", 0.0) or 0.0)
        fe = float(m.get("spark_phi_energy", 0.0) or 0.0)
        fn = float(m.get("spark_phi_novelty", 0.0) or 0.0)

        dv = abs(phi.valence - fv)
        de = abs(phi.energy - fe)
        dn = abs(phi.novelty - fn)

        avg_d = (dv + de + dn) / 3.0

        if avg_d < 0.05:
            return 1.35
        if avg_d < 0.15:
            return 1.15
        if avg_d < 0.30:
            return 1.0
        if avg_d < 0.50:
            return 0.9
        return 0.8
    except Exception:
        return 1.0


def score_fragments(
    fragments: List[Fragment],
    query: RecallQuery,
    *,
    half_life_days_short: float = 3.0,
    half_life_days_deep: float = 14.0,
) -> List[Fragment]:
    if not fragments:
        return []

    now_ts = _now_ts()

    if query.mode == "short_term":
        half_life = half_life_days_short
    elif query.mode == "deep":
        half_life = half_life_days_deep
    else:
        half_life = (half_life_days_short + half_life_days_deep) / 2.0

    scored: List[Fragment] = []

    for f in fragments:
        base = _base_salience(f)
        rec_w = _recency_weight(f, now_ts, half_life)
        kind_w = _kind_weight(f)
        q_w = _query_alignment_weight(f, query)
        phi_w = _phi_alignment_weight(f, query.phi)

        score = base * rec_w * kind_w * q_w * phi_w
        score = float(max(0.0, min(score, 5.0)))

        f.salience = score
        scored.append(f)

    scored.sort(key=lambda x: x.salience, reverse=True)
    return scored
