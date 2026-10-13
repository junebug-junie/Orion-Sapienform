"""Live eval: what would the arXiv source hand Orion today?

Read-only. Hits the real arXiv export API (spaced >= 3 s, honest User-Agent)
and, when a DSN is given, reads the real reading queue for already-seen ids.
Writes nothing, publishes nothing.

    PYTHONPATH=.:services/orion-world-pulse python \
        services/orion-world-pulse/evals/run_arxiv_source_eval.py \
        --dsn postgresql://postgres:postgres@localhost:55432/conjourney

Exit 0 only when: the configured cap is met, every pick is a bare
https://arxiv.org/abs/<id> page, no pick was already in the reading queue,
picks are distinct, and requests were spaced >= MIN_REQUEST_SPACING_SECONDS.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="", help="reading-queue DSN; blank = no dedupe (reported)")
    ap.add_argument("--sources", default=str(REPO_ROOT / "config" / "world_pulse" / "sources.yaml"))
    args = ap.parse_args()

    from app.services.ingest import arxiv_api_adapter as ax
    from app.services.ingest.base import http_get_text
    from app.services.source_registry import load_source_registry

    reg = load_source_registry(args.sources)
    source = next(s for s in reg.sources if s.source_id == "arxiv_research")

    seen: set[str] | None = None
    if args.dsn:
        from sqlalchemy import create_engine, text

        with create_engine(args.dsn).connect() as conn:
            seen = ax.seen_ids_from_urls(r[0] for r in conn.execute(text(ax.SEEN_ARXIV_URLS_SQL)))
    print(f"already-seen arXiv ids in reading queue: {'UNAVAILABLE' if seen is None else len(seen)}")

    stamps: list[float] = []

    def timed_get(url, **kw):
        stamps.append(time.monotonic())
        return http_get_text(url, **kw)

    picks = ax.fetch(source, 30, seen_ids_loader=lambda: seen, http_get=timed_get)
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]

    print(f"requests={len(stamps)} min_gap_s={min(gaps) if gaps else 0:.2f} cap={source.max_articles_per_day}")
    for p in picks:
        print(f"- [{p.metadata['query_category']}] {p.metadata['arxiv_id']}  {p.title}\n    {p.url}")

    failures: list[str] = []
    if len(picks) != source.max_articles_per_day:
        failures.append(f"picked {len(picks)} != cap {source.max_articles_per_day}")
    if any(g < ax.MIN_REQUEST_SPACING_SECONDS - 0.05 for g in gaps):
        failures.append(f"request gap below {ax.MIN_REQUEST_SPACING_SECONDS}s: {gaps}")
    ids = [p.metadata["arxiv_id"] for p in picks]
    if len(set(ids)) != len(ids):
        failures.append("duplicate picks")
    if seen and set(ids) & seen:
        failures.append(f"re-picked already-seen ids: {sorted(set(ids) & seen)}")
    if any(p.url != f"https://arxiv.org/abs/{p.metadata['arxiv_id']}" for p in picks):
        failures.append("a pick is not a bare abs URL")
    if seen is None:
        print("NOTE: dedupe not exercised (no --dsn)")
    print("PASS" if not failures else "FAIL: " + "; ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
