#!/usr/bin/env python3
"""Live, read-only eval: does the dream-search floor separate related from unrelated questions?

    POSTGRES_URI=postgresql+psycopg2://postgres:postgres@127.0.0.1:55432/conjourney \
    DREAM_SEARCH_EMBED_URL=http://127.0.0.1:8320/embedding \
    python services/orion-dream/evals/run_dream_search_calibration.py [--min-similarity X] \
        [--related "q=>dream:16,dream:15" ...] [--unrelated "cookie recipe" ...]

Reads the same rows the index loop would index (narratives + offered hypotheses)
inside a READ ONLY transaction, embeds each through vector-host HTTP /embedding
(which persists nothing), and scores cosine similarity locally -- no Chroma
write, no bus. For each related question it reports the best hit, whether it
is one of the expected ids, and the lowest similarity among the expected ids.
Exit 0 = the floor separates (every related best >= floor, every unrelated best
< floor, and each related question's best hit is expected); 1 = it does not;
2 = floor not configured, or Postgres / DB driver / embedder unavailable.

A second, KNOWN_WEAKNESS section scores questions phrased with the word
"dream". The framing lifts every narrative, so these do not separate; they are
reported for tracking only and never set the floor or the exit code. The tool
description asks Orion for topic-only queries for this reason.
Defaults target the 2026-09-29 corpus; pass --related/--unrelated when it drifts.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
from pathlib import Path

import httpx

_DREAM_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_DREAM_ROOT), str(_DREAM_ROOT.parents[1])]

from app.dream_search import _DOC_PREFIX, document_text  # noqa: E402
from app.introspect_dreams import doc_id, index_rows  # noqa: E402
from orion.introspect.semantic_index import (  # noqa: E402
    HTTP_TIMEOUT_SEC, SearchConfig, SearchUnavailableError, embed,
)

ENV_EXAMPLE = _DREAM_ROOT / ".env_example"
RELATED = {
    "seeing through a camera, vision and perception": {"dream:15", "dream:16"},
    "a library of recent pull requests and silent failures": {"dream:17"},
    "prediction errors felt as tectonic pressure": {"dream:18"},
    "infrastructure, connectivity and the strain of social engagement": {"dream:19"},
}
UNRELATED = ["chocolate chip cookie recipe", "taking my cat to the vet", "medieval poetry"]
FRAMED_RELATED = {
    "did you dream about pull requests?": {"dream:17"},
    "did you dream about your eyes / vision?": {"dream:15", "dream:16"},
    "a dream about seeing through a camera, vision and perception": {"dream:15", "dream:16"},
    "a dream about a library of recent pull requests and silent failures": {"dream:17"},
    "a dream about prediction errors felt as tectonic pressure": {"dream:18"},
    "a dream about infrastructure, connectivity and the strain of social engagement": {"dream:19"},
}
FRAMED_UNRELATED = [
    "did you dream about the ocean?",
    "a dream about my grandmother",
    "did you dream about a wedding?",
    "a dream about cooking pasta",
]


def _default(key: str, fallback: str) -> str:
    if os.environ.get(key):
        return os.environ[key]
    for line in ENV_EXAMPLE.read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return fallback


def _floor(cli_value: float | None) -> float:
    if cli_value is not None:
        return cli_value
    try:
        return float(_default("DREAM_SEARCH_MIN_SIMILARITY", "nan"))
    except ValueError:
        return math.nan


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0)


def _load_docs(uri: str) -> list[tuple[str, str]]:
    from sqlalchemy import create_engine, text

    with create_engine(uri).connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        pairs = index_rows(conn)
    docs = [(doc_id(k, r), document_text(k, r)) for k, r in pairs]
    return [(i, t) for i, t in docs if t]


def _parse_related(values: list[str] | None) -> dict[str, set[str]]:
    if not values:
        return RELATED
    out = {}
    for value in values:
        question, _, ids = value.partition("=>")
        out[question.strip()] = {i.strip() for i in ids.split(",") if i.strip()}
    return out


async def _score_set(client, cfg, vectors, related, unrelated, prefix=""):
    """Print one line per question; return (related bests, unrelated bests, failures)."""
    best: dict[str, list[float]] = {"related": [], "unrelated": []}
    failures: list[str] = []
    questions = [("related", q, ids) for q, ids in related.items()] + [("unrelated", q, set()) for q in unrelated]
    for group, question, expected in questions:
        qv, _ = await embed(client, cfg, question, doc_prefix=_DOC_PREFIX)
        ranked = sorted(((i, _cos(qv, v)) for i, v in vectors.items()), key=lambda s: -s[1])
        top = ranked[:3]
        score = top[0][1] if top else 0.0
        best[group].append(score)
        shown = ", ".join(f"{i}={s:.3f}" for i, s in top)
        line = f"{prefix}{group:9} {score:.3f}"
        if group == "related":
            hit = bool(top) and top[0][0] in expected
            scores = dict(ranked)
            present = [(scores[i], i) for i in expected if i in scores]
            lowest = f"{min(present)[1]}={min(present)[0]:.3f}" if present else "none-indexed"
            line += f" {'HIT' if hit else 'MISS'}  {question!r}  [{shown}]  expected_min={lowest}"
            if not hit:
                failures.append(f"related {question!r} best hit {top[0][0] if top else None} not in {sorted(expected)}")
        else:
            line += f"  {question!r}  [{shown}]"
        print(line)
        if (group == "related") != (score >= cfg.min_similarity):
            failures.append(f"{group} {question!r} best={score:.3f}")
    return best["related"], best["unrelated"], failures


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--related", action="append")
    parser.add_argument("--unrelated", action="append")
    parser.add_argument("--min-similarity", type=float)
    args = parser.parse_args()
    floor = _floor(args.min_similarity)
    if math.isnan(floor):
        print("UNKNOWN: floor not configured (DREAM_SEARCH_MIN_SIMILARITY missing or not a number)",
              file=sys.stderr)
        return 2
    cfg = SearchConfig(chroma_url="unused", embed_url=_default("DREAM_SEARCH_EMBED_URL", ""),
                       collection="unused", min_similarity=floor)
    uri = os.environ.get("POSTGRES_URI", "")
    if not uri or not cfg.embed_url.startswith("http"):
        print("UNKNOWN: POSTGRES_URI and DREAM_SEARCH_EMBED_URL (http) are required", file=sys.stderr)
        return 2
    try:
        from sqlalchemy.exc import NoSuchModuleError
    except ModuleNotFoundError:
        print("UNKNOWN: sqlalchemy is not installed in this interpreter", file=sys.stderr)
        return 2
    try:
        docs = _load_docs(uri)
    except (ModuleNotFoundError, NoSuchModuleError) as exc:
        print(f"UNKNOWN: DB driver missing for {uri.split('://', 1)[0]!r} ({exc}); install it or "
              "use postgresql+psycopg2://", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 -- any DB failure means unknown
        print(f"UNKNOWN: postgres unavailable ({type(exc).__name__})", file=sys.stderr)
        return 2
    related, unrelated = _parse_related(args.related), args.unrelated or UNRELATED
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
        try:
            vectors = {i: (await embed(client, cfg, t, doc_prefix=_DOC_PREFIX))[0] for i, t in docs}
            rel, unrel, failures = await _score_set(client, cfg, vectors, related, unrelated)
            low, high = min(rel), max(unrel)
            print(f"docs={len(docs)} floor={floor:.2f} related_min={low:.3f} unrelated_max={high:.3f} "
                  f"gap={low - high:.3f} midpoint={(low + high) / 2:.3f}")
            print("--- KNOWN_WEAKNESS: dream-worded questions (informational; "
                  "does not set the floor or the exit code) ---")
            f_rel, f_unrel, f_failures = await _score_set(
                client, cfg, vectors, FRAMED_RELATED, FRAMED_UNRELATED, prefix="KNOWN_WEAKNESS ")
        except SearchUnavailableError as exc:
            print(f"UNKNOWN: {exc}", file=sys.stderr)
            return 2
    f_low, f_high = min(f_rel), max(f_unrel)
    print(f"KNOWN_WEAKNESS framed_related_min={f_low:.3f} framed_unrelated_max={f_high:.3f} "
          f"gap={f_low - f_high:.3f} misses_or_floor_crossings={len(f_failures)}")
    sys.stdout.flush()
    if failures:
        print("FAIL: " + "; ".join(failures), file=sys.stderr)
        return 1
    print("OK: floor separates related from unrelated questions", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
