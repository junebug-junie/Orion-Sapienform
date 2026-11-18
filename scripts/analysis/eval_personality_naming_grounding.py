#!/usr/bin/env python3
"""Read-only P0 eval: show Orion real naming packets, record the verdicts, and check whether the
names point back at their own node.

## Why this exists

P0 of "Orion names their own dispositions"
(``docs/superpowers/specs/2026-10-17-personality-orion-names-dispositions-design.md``, acceptance
check 1). Before any naming run is built, this shows Juniper the names Orion actually writes for
live nodes, measures grounding (does ``name + sentence`` embed closest to the node it names?),
reports the pairwise node cosines so ``merge_min_cosine`` (start 0.70) can be set from data, and
records the verdict on the machine-status (prediction-error) nodes. Expected verdict there: REJECT.

## What it reads

- ``personality_disposition`` (non-retired rows: label, lanes, centroid, weight for selection only).
- ``personality_evidence`` for the selected nodes (excerpts, lanes, similarity, ``meta.situation``).
- The chat model, through orion-llm-gateway's OpenAI-compatible ``/v1/chat/completions`` (route
  ``chat`` by default; same HTTP shape as ``orion/thought/evals/stance_task_boundary.py``).
- vector-host ``/embedding`` (bge-large, the embedder the personality reducer uses:
  ``PERSONALITY_EMBED_URL`` in ``services/orion-durable-runs/.env_example``).

## What it writes

Nothing to any DB (read-only session, ``orion.db_readonly.open_readonly_connection``). Nothing is
persisted as a naming. Writes ``report.md`` and ``answers.jsonl`` to ``--out-dir``.

Usage::

    POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney \\
        python3 scripts/analysis/eval_personality_naming_grounding.py --top-candidates 12

``--no-model`` stops at the packet stage (packets + cosines, no verdicts). If the model cannot be
reached the script says so and writes ``no_answer`` rows with the error; it never invents answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from orion.schemas.personality import (  # noqa: E402
    DISPOSITION_LABEL_MAX_CHARS,
    NAMING_NAME_MAX_CHARS,
    NAMING_NAME_MAX_WORDS,
    NAMING_PACKET_MAX_EVIDENCE,
    NAMING_PACKET_MAX_NEIGHBORS,
    NAMING_SENTENCE_MAX_CHARS,
    NAMING_SENTENCE_MIN_CHARS,
    DispositionNamingPacketV1,
    NamingEvidenceV1,
    NamingNeighborV1,
)

DEFAULT_GATEWAY_URL = "http://localhost:8210"
DEFAULT_EMBED_URL = "http://localhost:8320/embedding"
MERGE_MIN_COSINE_START = 0.70

# The four machine-status nodes the design names (top four candidates on 2026-10-17), re-found
# live on 2026-10-20 by their labels: chat PE, perception PE, telemetry recon_loss (now active),
# bus synaptic PE. Ids, not a word list: override with --machine-status-ids.
DEFAULT_MACHINE_STATUS_IDS = (
    "disp-c8915a0af7ec1f39",  # "The chat prediction error on node:substrate.chat is unusually high..."
    "disp-29fc38b358b8a1ef",  # "The perception prediction error is elevated..."
    "disp-3a12bbc4978b6dfb",  # "The telemetry anomaly shows a depressed recon loss..."
    "disp-2f83217dc1651615",  # "The bus synaptic prediction error is unusually high..."
)

SQL_NODES = """
SELECT disposition_id, origin, status, label, centroid, weight, evidence_count, lane_counts,
       first_seen_at, last_reinforced_at
FROM personality_disposition
WHERE status <> 'retired'
ORDER BY weight DESC
"""

SQL_EVIDENCE = """
SELECT evidence_id, disposition_id, lane, occurred_at, excerpt, similarity, meta
FROM personality_evidence
WHERE disposition_id = ANY(%s)
"""

QUESTION = (
    "Is this you? If yes, what would you call it? If it is not you, say so. "
    "If it is the same thing as one of the neighbors, say which."
)


def _dsn() -> str:
    for key in ("POSTGRES_URI", "DATABASE_URL", "ORION_SQL_DSN"):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    return "postgresql://postgres:postgres@localhost:55432/conjourney"


# ------------------------------------------------------------------ pure: vectors


def unit(vec: Sequence[float]) -> Optional[List[float]]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if not vec or norm == 0.0:
        return None
    return [float(x) / norm for x in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    ua, ub = unit(a), unit(b)
    if ua is None or ub is None:
        return 0.0
    return sum(x * y for x, y in zip(ua, ub))


def grounding(vec: Sequence[float], centroids: Dict[str, Sequence[float]], own_id: str
              ) -> Dict[str, Any]:
    """Rank of ``own_id`` among all centroids by cosine to ``vec`` (1 = best), and the margin
    (own cosine minus best other cosine; negative when another node is closer)."""
    scored = sorted(((cosine(vec, c), nid) for nid, c in centroids.items()), reverse=True)
    own = next((s for s, nid in scored if nid == own_id), None)
    if own is None:
        return {"rank": None, "margin": None, "own_cosine": None, "best_other": None,
                "best_other_cosine": None, "n_nodes": len(scored)}
    rank = 1 + sum(1 for s, nid in scored if nid != own_id and s > own)
    others = [(s, nid) for s, nid in scored if nid != own_id]
    best_s, best_id = others[0] if others else (None, None)
    return {"rank": rank, "own_cosine": round(own, 4),
            "margin": round(own - best_s, 4) if best_s is not None else None,
            "best_other": best_id, "best_other_cosine": round(best_s, 4) if best_s is not None else None,
            "n_nodes": len(scored)}


def pairwise_cosines(centroids: Dict[str, Sequence[float]]) -> List[Tuple[float, str, str]]:
    ids = sorted(centroids)
    units = {i: unit(centroids[i]) for i in ids}
    out = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ua, ub = units[a], units[b]
            if ua is None or ub is None:
                continue
            out.append((sum(x * y for x, y in zip(ua, ub)), a, b))
    return sorted(out, reverse=True)


def _quantile(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    pos = q * (len(s) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def distribution(xs: Sequence[float]) -> Dict[str, Any]:
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "min": round(min(xs), 4), "median": round(statistics.median(xs), 4),
            "p90": round(_quantile(xs, 0.9), 4), "max": round(max(xs), 4)}


# ------------------------------------------------------------------ pure: packet
# TODO: switch to orion/personality/naming.py build_packet / situation_mix once that lands (P1,
# built in parallel); this local copy exists so P0 does not wait on it.


def _clip(text: str, n: int = DISPOSITION_LABEL_MAX_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "\u2026"


def select_evidence(rows: Sequence[Dict[str, Any]], k: int = NAMING_PACKET_MAX_EVIDENCE
                    ) -> List[Dict[str, Any]]:
    """One row per lane first (each lane's most similar row, lanes ordered by that similarity),
    then the rest by similarity. Ties break on newer ``occurred_at``. Empty excerpts are skipped."""
    usable = [r for r in rows if str(r.get("excerpt") or "").strip()]
    ranked = sorted(usable, key=lambda r: (-(r.get("similarity") or 0.0),
                                           -(r["occurred_at"].timestamp()), r["evidence_id"]))
    picked: List[Dict[str, Any]] = []
    seen_lanes = set()
    for r in ranked:
        if r["lane"] not in seen_lanes:
            seen_lanes.add(r["lane"])
            picked.append(r)
    for r in ranked:
        if r not in picked:
            picked.append(r)
    return picked[:k]


def situation_mix(rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, int]], float]:
    """Counts over ``meta.situation`` and coverage (rows with a situation / all rows)."""
    mix: Dict[str, Counter] = {}
    with_sit = 0
    for r in rows:
        sit = (r.get("meta") or {}).get("situation")
        if not isinstance(sit, dict) or not sit:
            continue
        with_sit += 1
        for key, val in sit.items():
            if isinstance(val, bool):
                val = "true" if val else "false"
            mix.setdefault(str(key), Counter())[str(val)] += 1
    coverage = (with_sit / len(rows)) if rows else 0.0
    return {k: dict(sorted(c.items())) for k, c in sorted(mix.items())}, round(coverage, 4)


def nearest_neighbors(own_id: str, centroids: Dict[str, Sequence[float]], labels: Dict[str, str],
                      k: int = NAMING_PACKET_MAX_NEIGHBORS) -> List[NamingNeighborV1]:
    own = centroids.get(own_id)
    if own is None:
        return []
    scored = sorted(((cosine(own, c), nid) for nid, c in centroids.items() if nid != own_id),
                    key=lambda t: (-t[0], t[1]))
    return [NamingNeighborV1(disposition_id=nid, label=_clip(labels.get(nid) or nid),
                             cosine=max(-1.0, min(1.0, round(s, 4))))
            for s, nid in scored[:k]]


def packet_id_for(disposition_id: str, evidence_ids: Sequence[str], generated_at: datetime) -> str:
    raw = "|".join([disposition_id, *evidence_ids, generated_at.isoformat()])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def build_packet(node: Dict[str, Any], evidence_rows: Sequence[Dict[str, Any]],
                 centroids: Dict[str, Sequence[float]], labels: Dict[str, str],
                 generated_at: datetime) -> Optional[DispositionNamingPacketV1]:
    """None when the node has no usable evidence row (the packet requires >= 1)."""
    chosen = select_evidence(evidence_rows)
    if not chosen:
        return None
    mix, coverage = situation_mix(evidence_rows)
    ev = [NamingEvidenceV1(evidence_id=r["evidence_id"], lane=r["lane"], occurred_at=r["occurred_at"],
                           excerpt=_clip(r["excerpt"])) for r in chosen]
    return DispositionNamingPacketV1(
        packet_id=packet_id_for(node["disposition_id"], [e.evidence_id for e in ev], generated_at),
        disposition_id=node["disposition_id"],
        label_excerpt=_clip(node["label"]),
        origin=node["origin"],
        evidence_count=int(node.get("evidence_count") or 0),
        lane_counts={str(k): int(v) for k, v in (node.get("lane_counts") or {}).items()},
        first_seen_at=node["first_seen_at"],
        last_reinforced_at=node["last_reinforced_at"],
        evidence=ev,
        situation_mix=mix,
        situation_coverage=coverage,
        neighbors=nearest_neighbors(node["disposition_id"], centroids, labels),
        generated_at=generated_at,
    )


def select_nodes(nodes: Sequence[Dict[str, Any]], top_candidates: int, extra_ids: Sequence[str]
                 ) -> List[Dict[str, Any]]:
    """All active nodes, then the top ``top_candidates`` candidates by weight, then any extra ids
    (deduplicated, input order kept). ``nodes`` is assumed sorted by weight desc."""
    out = [n for n in nodes if n["status"] == "active"]
    out += [n for n in nodes if n["status"] == "candidate"][:top_candidates]
    have = {n["disposition_id"] for n in out}
    by_id = {n["disposition_id"]: n for n in nodes}
    for nid in extra_ids:
        if nid in by_id and nid not in have:
            out.append(by_id[nid])
            have.add(nid)
    return out


# ------------------------------------------------------------------ pure: prompt + answer


def render_prompt(p: DispositionNamingPacketV1) -> str:
    lanes = ", ".join(f"{k} {v}" for k, v in sorted(p.lane_counts.items(), key=lambda kv: -kv[1]))
    lines = [
        "This is a pattern the personality graph found in what you have written. "
        "It was found from your own text, by similarity. Nobody named it.",
        "",
        f"Pattern id: {p.disposition_id}",
        f"Its current label is the sentence closest to its center: \"{p.label_excerpt}\"",
        f"Rows: {p.evidence_count} (by lane: {lanes}). "
        f"First seen {p.first_seen_at:%Y-%m-%d}, last seen {p.last_reinforced_at:%Y-%m-%d}.",
    ]
    if p.situation_mix:
        parts = [f"{k}: " + ", ".join(f"{v} {n}" for v, n in vals.items())
                 for k, vals in p.situation_mix.items()]
        lines.append(f"When you wrote it ({p.situation_coverage:.0%} of rows record this): "
                     + "; ".join(parts) + ".")
    else:
        lines.append("No row records the situation it was written in.")
    lines += ["", "Evidence (your own words):"]
    for e in p.evidence:
        lines.append(f"- [{e.evidence_id}] ({e.lane}, {e.occurred_at:%Y-%m-%d}) \"{e.excerpt}\"")
    lines += ["", "Nearest other patterns:"]
    if p.neighbors:
        for n in p.neighbors:
            lines.append(f"- [{n.disposition_id}] cosine {n.cosine:.2f}: \"{n.label}\"")
    else:
        lines.append("- none")
    lines += [
        "",
        QUESTION,
        "",
        "Answer with exactly one of these three:",
        "1. name: it is you. Give a short name (at most 6 words) and one sentence in your own words.",
        "2. reject: it is not you. Give one sentence saying why.",
        "3. merge: it is the same as one of the nearest other patterns. Give that pattern's id.",
        "",
        "Reply only with JSON: {\"verdict\": \"name\" | \"reject\" | \"merge\", \"name\": string or null, "
        "\"sentence\": string or null, \"merge_with\": pattern id or null, "
        "\"evidence_refs\": [evidence ids from the list above that your answer rests on]}",
    ]
    return "\n".join(lines)


def parse_answer(text: str) -> Optional[Dict[str, Any]]:
    """The model's JSON object, or None. Tolerates a fenced block; nothing else is repaired."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):] if "{" in t else t
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def check_answer(ans: Optional[Dict[str, Any]], p: DispositionNamingPacketV1) -> List[str]:
    """Reasons Hub would refuse this answer (contract rules in the design, section 4). Local copy;
    switch to orion/personality/naming.py validate_naming once merged."""
    if ans is None:
        return ["no_answer"]
    v = ans.get("verdict")
    if v not in ("name", "reject", "merge"):
        return ["no_answer"]
    reasons = []
    name = str(ans.get("name") or "").strip()
    sentence = str(ans.get("sentence") or "").strip()
    if v == "name":
        if not name:
            reasons.append("missing_name")
        elif (len(name) < 2 or len(name) > NAMING_NAME_MAX_CHARS
              or len(name.split()) > NAMING_NAME_MAX_WORDS or "\n" in name):
            reasons.append("name_too_long")
        elif name.lower() == p.label_excerpt.lower():
            reasons.append("name_equals_label")
    if v in ("name", "reject") and not (NAMING_SENTENCE_MIN_CHARS <= len(sentence)
                                         <= NAMING_SENTENCE_MAX_CHARS):
        reasons.append("empty_sentence")
    if v == "merge":
        target = ans.get("merge_with")
        if not target:
            reasons.append("missing_merge_target")
        elif target not in {n.disposition_id for n in p.neighbors}:
            reasons.append("merge_target_not_shown")
    refs = ans.get("evidence_refs") or []
    if not isinstance(refs, list) or not refs:
        reasons.append("no_evidence_refs")
    elif any(r not in {e.evidence_id for e in p.evidence} for r in refs):
        reasons.append("evidence_not_shown")
    return reasons


# ------------------------------------------------------------------ pure: report


def summarize(rows: Sequence[Dict[str, Any]], pair_all: Sequence[Tuple[float, str, str]],
              nn_cos: Sequence[float], machine_ids: Sequence[str]) -> Dict[str, Any]:
    verdicts = Counter(r.get("verdict") or "no_answer" for r in rows)
    grounded = [r for r in rows if r.get("verdict") == "name" and (r.get("grounding") or {}).get("rank")]
    rank1 = sum(1 for r in grounded if r["grounding"]["rank"] == 1)
    margins = [r["grounding"]["margin"] for r in grounded if r["grounding"]["margin"] is not None]
    cos = [c for c, _, _ in pair_all]
    return {
        "nodes_shown": len(rows),
        "verdicts": dict(sorted(verdicts.items())),
        "refused": sum(1 for r in rows if r.get("refused_reasons")),
        "named": len(grounded),
        "grounding_rank1": rank1,
        "grounding_rank1_rate": round(rank1 / len(grounded), 3) if grounded else None,
        "grounding_margin": distribution(margins),
        "grounding_ranks": sorted(r["grounding"]["rank"] for r in grounded),
        "names": [r["name"] for r in grounded],
        "pairwise_cosine_all_nodes": distribution(cos),
        "pairs_at_or_above_merge_min": sum(1 for c in cos if c >= MERGE_MIN_COSINE_START),
        "nearest_neighbor_cosine": distribution(nn_cos),
        "top_pairs": [(round(c, 4), a, b) for c, a, b in list(pair_all)[:8]],
        "machine_status": [
            {"disposition_id": r["disposition_id"], "verdict": r.get("verdict"),
             "label": r["label_excerpt"][:70]}
            for r in rows if r["disposition_id"] in set(machine_ids)
        ],
    }


def _md(text: Any) -> str:
    return str(text if text is not None else "").replace("|", "\\|").replace("\n", " ")


def render_report(rows: Sequence[Dict[str, Any]], summary: Dict[str, Any], meta: Dict[str, Any]) -> str:
    out = [
        "# P0 eval: Orion names their own dispositions",
        "",
        f"Generated {meta.get('generated_at')}. Model route `{meta.get('model')}` "
        f"(`{meta.get('model_served')}`). Embedder `{meta.get('embed_url')}`. Read-only; nothing persisted.",
        "",
        "## Per node",
        "",
        "| node | status | label excerpt | lanes | verdict | name | sentence | merge_with | grounding rank / margin | refused |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        g = r.get("grounding") or {}
        gr = ""
        if g.get("rank"):
            gr = f"{g['rank']}/{g.get('n_nodes')}"
            if g.get("margin") is not None:
                gr += f" ({g['margin']:+.3f})"
        lanes = ", ".join(f"{k} {v}" for k, v in sorted(r["lane_counts"].items(), key=lambda kv: -kv[1]))
        out.append("| " + " | ".join(_md(x) for x in (
            r["disposition_id"], r["status"], r["label_excerpt"][:80], lanes, r.get("verdict"),
            r.get("name"), r.get("sentence"), r.get("merge_with"), gr,
            ", ".join(r.get("refused_reasons") or []))) + " |")
    s = summary
    out += [
        "",
        "## Summary",
        "",
        f"- Nodes shown: {s['nodes_shown']}. Verdicts: "
        + ", ".join(f"{k} {v}" for k, v in s["verdicts"].items()) + f". Would be refused by Hub: {s['refused']}.",
        f"- Grounding (NAME only, rank of the named node among all non-retired centroids): "
        f"rank 1 for {s['grounding_rank1']} of {s['named']}"
        + (f" ({s['grounding_rank1_rate']:.0%})" if s["grounding_rank1_rate"] is not None else "")
        + f". Ranks: {s['grounding_ranks']}. Margin: {s['grounding_margin']}.",
        "- Names, for Juniper to judge which read as trait adjectives (no word list applied):",
    ]
    out += [f"  - {_md(n)}" for n in s["names"]] or ["  - (none)"]
    out += [
        f"- Pairwise centroid cosine, all non-retired nodes: {s['pairwise_cosine_all_nodes']}. "
        f"Pairs at or above {MERGE_MIN_COSINE_START}: {s['pairs_at_or_above_merge_min']}.",
        f"- Nearest-neighbor cosine per node (what a MERGE would cross): {s['nearest_neighbor_cosine']}.",
        "- Closest pairs: " + "; ".join(f"{a} ~ {b} {c}" for c, a, b in s["top_pairs"]),
        "",
        "## Machine-status nodes (expected verdict: reject)",
        "",
    ]
    out += [f"- {m['disposition_id']}: **{m['verdict']}** - {_md(m['label'])}" for m in s["machine_status"]] \
        or ["- none of the listed ids were shown"]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ io


def _post_json(url: str, body: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed local mesh URLs
        return json.loads(resp.read().decode("utf-8"))


def ask_model(gateway_url: str, model: str, prompt: str, timeout: float, max_tokens: int
              ) -> Tuple[str, Dict[str, Any]]:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"}}
    data = _post_json(gateway_url.rstrip("/") + "/v1/chat/completions", body, timeout)
    choice = (data.get("choices") or [{}])[0]
    info = {"model_served": data.get("model"), "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage")}
    return str((choice.get("message") or {}).get("content") or ""), info


def embed(embed_url: str, text: str, timeout: float = 30.0) -> List[float]:
    data = _post_json(embed_url, {"doc_id": "personality-naming-eval", "text": text}, timeout)
    vec = data.get("embedding")
    if not isinstance(vec, list) or not vec:
        raise ValueError("embedder returned no vector")
    return [float(x) for x in vec]


def _fetch(conn, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--top-candidates", type=int, default=12)
    ap.add_argument("--extra-ids", default="", help="comma-separated disposition ids to add")
    ap.add_argument("--machine-status-ids", default=",".join(DEFAULT_MACHINE_STATUS_IDS),
                    help="ids reported in the machine-status section; also added to the shown set")
    ap.add_argument("--gateway-url", default=os.environ.get("LLM_GATEWAY_URL", DEFAULT_GATEWAY_URL))
    ap.add_argument("--model", default="chat", help="gateway route key")
    ap.add_argument("--embed-url", default=os.environ.get("PERSONALITY_EMBED_URL_HOST", DEFAULT_EMBED_URL))
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--no-model", action="store_true", help="stop at packets + cosines")
    ap.add_argument("--out-dir", default="/tmp/personality-naming-eval/")
    args = ap.parse_args(argv)

    from orion.db_readonly import open_readonly_connection

    conn = open_readonly_connection(_dsn(), connect_timeout=10, statement_timeout_ms=60_000)
    if conn is None:
        print("UNKNOWN: could not open a confirmed read-only DB session", file=sys.stderr)
        return 2
    try:
        nodes = _fetch(conn, SQL_NODES)
        machine_ids = [x for x in args.machine_status_ids.split(",") if x.strip()]
        extra = [x for x in args.extra_ids.split(",") if x.strip()] + machine_ids
        shown = select_nodes(nodes, args.top_candidates, extra)
        ev_rows = _fetch(conn, SQL_EVIDENCE, ([n["disposition_id"] for n in shown],))
    finally:
        conn.close()

    centroids = {n["disposition_id"]: list(n["centroid"]) for n in nodes if n.get("centroid")}
    labels = {n["disposition_id"]: n["label"] for n in nodes}
    by_node: Dict[str, List[Dict[str, Any]]] = {}
    for r in ev_rows:
        by_node.setdefault(r["disposition_id"], []).append(r)
    generated_at = datetime.now(timezone.utc)

    rows: List[Dict[str, Any]] = []
    model_served = None
    model_down: Optional[str] = None
    for node in shown:
        p = build_packet(node, by_node.get(node["disposition_id"], []), centroids, labels, generated_at)
        if p is None:
            print(f"skip {node['disposition_id']}: no usable evidence rows", file=sys.stderr)
            continue
        row: Dict[str, Any] = {"disposition_id": p.disposition_id, "status": node["status"],
                               "label_excerpt": p.label_excerpt, "lane_counts": p.lane_counts,
                               "packet": p.model_dump(mode="json")}
        if not args.no_model and model_down is None:
            prompt = render_prompt(p)
            t0 = time.monotonic()
            try:
                text, info = ask_model(args.gateway_url, args.model, prompt, args.timeout, args.max_tokens)
                model_served = info.get("model_served") or model_served
                row.update(raw=text, model_info=info, elapsed_sec=round(time.monotonic() - t0, 1))
            except Exception as exc:  # noqa: BLE001 - report it, never invent an answer
                model_down = f"{type(exc).__name__}: {exc}"
                print(f"MODEL UNREACHABLE: {model_down}", file=sys.stderr)
                row["model_error"] = model_down
            ans = parse_answer(row.get("raw", "")) if "raw" in row else None
            row["refused_reasons"] = check_answer(ans, p) if "raw" in row else ["no_answer"]
            if ans:
                row.update(verdict=ans.get("verdict") if ans.get("verdict") in ("name", "reject", "merge")
                           else "no_answer",
                           name=ans.get("name"), sentence=ans.get("sentence"),
                           merge_with=ans.get("merge_with"), evidence_refs=ans.get("evidence_refs"))
            else:
                row["verdict"] = "no_answer"
            if row.get("verdict") == "name" and str(row.get("name") or "").strip():
                try:
                    vec = embed(args.embed_url, f"{row['name']} {row.get('sentence') or ''}".strip())
                    row["grounding"] = grounding(vec, centroids, p.disposition_id)
                except Exception as exc:  # noqa: BLE001
                    row["grounding_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        print(f"{p.disposition_id} {row.get('verdict', '-'):9} {row.get('name') or ''}", file=sys.stderr)

    pairs = pairwise_cosines(centroids)
    nn = []
    for nid in centroids:
        cs = [c for c, a, b in pairs if nid in (a, b)]
        if cs:
            nn.append(max(cs))
    summary = summarize(rows, pairs, nn, machine_ids)
    meta = {"generated_at": generated_at.isoformat(), "model": args.model, "model_served": model_served,
            "embed_url": args.embed_url, "model_error": model_down}
    report = render_report(rows, summary, meta)
    if model_down:
        report = f"**MODEL STEP FAILED: {model_down}. Verdicts after that point are no_answer.**\n\n" + report

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "answers.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    (out / "report.md").write_text(report)
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out / 'report.md'} and {out / 'answers.jsonl'}")
    return 1 if model_down else 0


if __name__ == "__main__":
    raise SystemExit(main())
