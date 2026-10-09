#!/usr/bin/env python3
"""Live, read-only eval: does the curiosity-search floor separate related from unrelated questions?

    POSTGRES_URI=postgresql://postgres:postgres@127.0.0.1:55432/conjourney \
    HUB_CURIOSITY_SEARCH_EMBED_URL=http://127.0.0.1:8320/embedding \
    python services/orion-hub/evals/run_curiosity_search_calibration.py [--min-similarity X] \
        [--related "q=>run1,run2" ...] [--unrelated "cookie recipe" ...]

Reads exactly the documents the Hub index loop would index (newest write-up per
run, its Answer section or opening, 1800 chars) inside a READ ONLY transaction,
embeds each through vector-host HTTP /embedding (which persists nothing), and
scores cosine similarity locally -- no Chroma write, no bus.

For each related question it reports the best hit, whether it is one of the
expected runs, and precision among the top 5 hits at or above the floor
(expected / above-floor). Expected sets are the write-ups that name each topic
(2026-10-09 corpus, found by text match), plus top hits read by hand and
confirmed on-topic; a hit not in a set is not necessarily off-topic. Exit 0 = the floor separates (every
related best >= floor and expected, every unrelated best < floor); 1 = it does
not; 2 = floor not configured, or Postgres / driver / embedder unavailable.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
from pathlib import Path

import httpx

_HUB_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_HUB_ROOT), str(_HUB_ROOT.parents[1])]

from orion.introspect.semantic_index import (  # noqa: E402
    HTTP_TIMEOUT_SEC, SearchConfig, SearchUnavailableError, embed,
)
from scripts.curiosity_introspect_listener import (  # noqa: E402
    _DOC_PREFIX, INDEX_ROWS_SQL, index_docs_from_rows,
)

ENV_EXAMPLE = _HUB_ROOT / ".env_example"
TOP = 5
RELATED = {
    "why were so many stances rejected, and is the crystallization gate biased toward concrete content": {
        "23a78070d32a", "32b42392f495", "6a189691ad99", "9fd8cf09163b", "a394430e781e",
        "c8bbf2b9b3c0", "d034f854569d", "d05ef10b303a", "d1223ad8e353", "cf6f68ac804c",
        "7736d5271d97", "54c4a3a3e801",
    },
    "reverie thoughts about unresolved prediction errors that never discharge": {
        "20260921T235425Z-f129a0", "20260922T025855Z-b88ac9", "20260922T060916Z-a3b2d8",
        "282fbb9a08e4", "3d6a3400aaee", "5e7578dfcffa", "62d0548ee14c", "8ae15f17febe",
        "9e911d60c111", "d59b680598af", "57a98162032b",
    },
    "the people on Juniper's team I have never been told the names of": {
        "453e5abdef75", "82f859eaa837", "8998e10d39ff", "d30da0e42ff0", "eb5e10a8ee76", "44912d89079c",
    },
    "substrate prediction-error nodes isolated at degree zero in the atlas": {
        "3b2d038cf18e", "4b9621bab74b", "5c4a40eaf076", "c67b1a10fb93", "15ac6b45eca1",
    },
    "do temporary files I write survive from one turn to the next": {"0b8b04cdc81a", "c9d94022ae20"},
    "do misaligned turns cluster on one of the brains spliced into me": {
        "393c6cf88fe3", "5be237da9365", "5dbb2df26321",
    },
}
UNRELATED = [
    "chocolate chip cookie recipe", "taking my cat to the vet", "medieval poetry",
    "the rules of cricket", "how to change a car tire",
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
        return float(_default("HUB_CURIOSITY_SEARCH_MIN_SIMILARITY", "nan"))
    except ValueError:
        return math.nan


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0)


async def _load_docs(uri: str) -> list[tuple[str, str]]:
    import asyncpg

    conn = await asyncpg.connect(uri)
    try:
        async with conn.transaction(readonly=True):
            rows = [dict(r) for r in await conn.fetch(INDEX_ROWS_SQL)]
    finally:
        await conn.close()
    return [(run_id, text) for run_id, text, _ in index_docs_from_rows(rows)]


def _parse_related(values: list[str] | None) -> dict[str, set[str]]:
    if not values:
        return RELATED
    out = {}
    for value in values:
        question, _, ids = value.partition("=>")
        out[question.strip()] = {i.strip() for i in ids.split(",") if i.strip()}
    return out


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--related", action="append")
    parser.add_argument("--unrelated", action="append")
    parser.add_argument("--min-similarity", type=float)
    args = parser.parse_args()
    floor = _floor(args.min_similarity)
    if math.isnan(floor):
        print("UNKNOWN: floor not configured (HUB_CURIOSITY_SEARCH_MIN_SIMILARITY missing or not a number)",
              file=sys.stderr)
        return 2
    cfg = SearchConfig(chroma_url="unused", embed_url=_default("HUB_CURIOSITY_SEARCH_EMBED_URL", ""),
                       collection="unused", min_similarity=floor)
    uri = os.environ.get("POSTGRES_URI", "")
    if not uri or not cfg.embed_url.startswith("http"):
        print("UNKNOWN: POSTGRES_URI and HUB_CURIOSITY_SEARCH_EMBED_URL (http) are required", file=sys.stderr)
        return 2
    try:
        docs = await _load_docs(uri)
    except ModuleNotFoundError as exc:
        print(f"UNKNOWN: DB driver missing ({exc})", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 -- any DB failure means unknown
        print(f"UNKNOWN: postgres unavailable ({type(exc).__name__})", file=sys.stderr)
        return 2
    related, unrelated = _parse_related(args.related), args.unrelated or UNRELATED
    failures: list[str] = []
    best: dict[str, list[float]] = {"related": [], "unrelated": []}
    kept = relevant = 0
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
        try:
            vectors = {i: (await embed(client, cfg, t, doc_prefix=_DOC_PREFIX))[0] for i, t in docs}
            questions = [("related", q, ids) for q, ids in related.items()] + [("unrelated", q, set()) for q in unrelated]
            for group, question, expected in questions:
                qv, _ = await embed(client, cfg, question, doc_prefix=_DOC_PREFIX)
                ranked = sorted(((i, _cos(qv, v)) for i, v in vectors.items()), key=lambda s: -s[1])
                top = ranked[:3]
                score = top[0][1] if top else 0.0
                best[group].append(score)
                shown = ", ".join(f"{i}={s:.3f}" for i, s in top)
                line = f"{group:9} {score:.3f}"
                if group == "related":
                    hit = bool(top) and top[0][0] in expected
                    above = [i for i, s in ranked[:TOP] if s >= floor]
                    good = sum(1 for i in above if i in expected)
                    kept, relevant = kept + len(above), relevant + good
                    line += f" {'HIT' if hit else 'MISS'} p@floor={good}/{len(above)}  {question!r}  [{shown}]"
                    if not hit:
                        failures.append(f"related {question!r} best hit {top[0][0] if top else None} not expected")
                else:
                    line += f"  {question!r}  [{shown}]"
                print(line)
                if (group == "related") != (score >= floor):
                    failures.append(f"{group} {question!r} best={score:.3f}")
        except SearchUnavailableError as exc:
            print(f"UNKNOWN: {exc}", file=sys.stderr)
            return 2
    low, high = min(best["related"]), max(best["unrelated"])
    precision = f"{relevant}/{kept}={relevant / kept:.2f}" if kept else "0/0"
    print(f"docs={len(docs)} floor={floor:.2f} related_min={low:.3f} unrelated_max={high:.3f} "
          f"gap={low - high:.3f} midpoint={(low + high) / 2:.3f} precision_top{TOP}_at_floor={precision}")
    sys.stdout.flush()
    if failures:
        print("FAIL: " + "; ".join(failures), file=sys.stderr)
        return 1
    print("OK: floor separates related from unrelated questions", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
