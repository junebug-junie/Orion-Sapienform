#!/usr/bin/env python3
"""Live eval: does the reading-search floor separate related from unrelated questions?

    python services/orion-hub/evals/run_reading_search_calibration.py \
        [--related "graphics cards" ...] [--unrelated "cookie recipe" ...] [--min-similarity X]

Embeds each question once (vector-host HTTP /embedding, which persists nothing),
asks Chroma for the nearest indexed readings, and prints the top similarities.
Exit 0 = every related question's best hit clears the floor and every unrelated
one stays below it; 1 = the floor does not separate them; 2 = index or embedder
unavailable. Defaults target the 2026-09-28 corpus (mostly GPU/Nvidia coverage);
pass --related/--unrelated when the corpus drifts.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import httpx

from orion.world_pulse_read.search import (
    HTTP_TIMEOUT_SEC, ReadingSearchConfig, SearchUnavailableError, embed, nearest,
)

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env_example"
RELATED = ["graphics cards", "AI chip export controls", "websites blocking automated fetchers"]
UNRELATED = ["chocolate chip cookie recipe", "taking my cat to the vet", "medieval poetry"]


def _default(key: str, fallback: str) -> str:
    if os.environ.get(key):
        return os.environ[key]
    for line in ENV_EXAMPLE.read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return fallback


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--related", action="append")
    parser.add_argument("--unrelated", action="append")
    parser.add_argument("--min-similarity", type=float)
    args = parser.parse_args()
    cfg = ReadingSearchConfig(
        chroma_url=_default("HUB_READING_SEARCH_CHROMA_URL", ""),
        embed_url=_default("HUB_READING_SEARCH_EMBED_URL", ""),
        collection=_default("HUB_READING_SEARCH_COLLECTION", "orion_reading_results"),
        min_similarity=args.min_similarity if args.min_similarity is not None
        else float(_default("HUB_READING_SEARCH_MIN_SIMILARITY", "nan")),
    )
    groups = [("related", args.related or RELATED), ("unrelated", args.unrelated or UNRELATED)]
    failures = []
    best: dict[str, list[float]] = {"related": [], "unrelated": []}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
        try:
            for group, questions in groups:
                for question in questions:
                    vector, _ = await embed(client, cfg, question)
                    top = await nearest(client, cfg, vector, n=3)
                    score = top[0][1] if top else 0.0
                    best[group].append(score)
                    shown = ", ".join(f"{sid[:12]}={sim:.3f}" for sid, sim in top)
                    print(f"{group:9} {score:.3f}  {question!r}  [{shown}]")
                    if (group == "related") != (score >= cfg.min_similarity):
                        failures.append(f"{group} {question!r} top={score:.3f}")
        except SearchUnavailableError as exc:
            print(f"UNKNOWN: {exc}", file=sys.stderr)
            return 2
    low, high = min(best["related"]), max(best["unrelated"])
    print(f"floor={cfg.min_similarity:.2f} related_min={low:.3f} unrelated_max={high:.3f} gap={low - high:.3f}")
    if failures:
        print("FAIL: floor does not separate: " + "; ".join(failures), file=sys.stderr)
        return 1
    print("OK: floor separates related from unrelated questions", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
