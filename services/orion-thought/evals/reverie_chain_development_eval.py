"""Does a reverie chain develop, or does it repeat itself? (read-only, live data)

A chain is up to N reverie steps on one theme. Before 2026-10-16 every step got
the same input, so step 4 mostly restated step 1. This eval measures that from
the stored thoughts in ``substrate_reverie_thought`` (no writes):

- chains and the steps-per-chain distribution;
- median word-bigram overlap (Jaccard) between the first and the last step of
  each chain with 2+ steps -- high means the chain restated itself;
- the same-subject cross-chain baseline: the same overlap between the first
  steps of two consecutive chains on the same selected loop. A chain that
  develops should sit BELOW this line; one that repeats sits at or above it;
- % of thoughts that open "The coalition";
- next_focus fill rate;
- count of thoughts that say "unattended for".

    python services/orion-thought/evals/reverie_chain_development_eval.py --hours 24

DSN: ``REVERIE_EVAL_DSN`` (default the host-published athena Postgres,
localhost:55432). ``--psql-container`` reads through ``docker exec <c> psql``
instead, for hosts where the port is not published.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from typing import Any, Iterable

_SQL = (
    "SELECT thought_json->>'chain_id', "
    "COALESCE((thought_json->>'thought_index')::int, -1), "
    "COALESCE(thought_json->'coalition'->>'selected_open_loop_id', ''), "
    "interpretation, COALESCE(thought_json->>'next_focus', ''), created_at "
    "FROM substrate_reverie_thought "
    "WHERE created_at > now() - make_interval(hours => {hours}) "
    "ORDER BY created_at"
)

_WORD = re.compile(r"[a-z0-9']+")


def bigrams(text: str) -> set[tuple[str, str]]:
    words = _WORD.findall((text or "").lower())
    return set(zip(words, words[1:]))


def bigram_overlap(a: str, b: str) -> float | None:
    """Jaccard overlap of word bigrams; None when either side has no bigram."""
    ba, bb = bigrams(a), bigrams(b)
    if not ba or not bb:
        return None
    return len(ba & bb) / len(ba | bb)


def _median(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def summarize(rows: list[tuple[Any, ...]]) -> dict[str, Any]:
    """rows: (chain_id, thought_index, subject, interpretation, next_focus, created_at)."""
    chains: dict[str, list[tuple[int, str, str, str]]] = defaultdict(list)
    for chain_id, idx, subject, interp, next_focus, _created in rows:
        if chain_id:
            chains[chain_id].append((int(idx), subject or "", interp or "", next_focus or ""))
    for steps in chains.values():
        steps.sort(key=lambda s: s[0])

    dist = Counter(len(s) for s in chains.values())
    within = [bigram_overlap(s[0][2], s[-1][2]) for s in chains.values() if len(s) >= 2]

    # Cross-chain baseline: first step of consecutive chains on the same subject.
    # Chains are ordered by the created_at of their first stored step.
    first_by_subject: dict[str, list[str]] = defaultdict(list)
    first_seen: dict[str, Any] = {}
    for chain_id, _i, _s, _t, _n, created in rows:
        if chain_id and chain_id not in first_seen:
            first_seen[chain_id] = created
    for chain_id in sorted(chains, key=lambda c: str(first_seen.get(c))):
        steps = chains[chain_id]
        if steps and steps[0][1]:
            first_by_subject[steps[0][1]].append(steps[0][2])
    cross: list[float | None] = []
    for texts in first_by_subject.values():
        cross.extend(bigram_overlap(a, b) for a, b in zip(texts, texts[1:]))

    n = len(rows)
    opens_coalition = sum(1 for r in rows if (r[3] or "").lstrip().lower().startswith("the coalition"))
    # Fill rate over steps that were asked for next_focus: every chain step but
    # the chain's last stored one (the last is either the final step, never
    # asked, or the step that ended the chain by leaving it empty -- counted
    # separately as chains_ended_without_next_focus).
    asked = [s for steps in chains.values() for s in steps[:-1]]
    next_focus_filled = sum(1 for s in asked if s[3].strip())
    ended_empty = sum(1 for steps in chains.values() if steps and not steps[-1][3].strip())
    unattended = sum(1 for r in rows if "unattended for" in (r[3] or "").lower())
    return {
        "thoughts": n,
        "chains": len(chains),
        "steps_per_chain": dict(sorted(dist.items())),
        "chains_with_2plus_steps": len(within),
        "median_bigram_overlap_first_vs_last_step": _median(within),
        "same_subject_cross_chain_baseline": _median(cross),
        "cross_chain_pairs": sum(1 for v in cross if v is not None),
        "pct_opening_the_coalition": round(100.0 * opens_coalition / n, 1) if n else None,
        "next_focus_fill_rate_pct": round(100.0 * next_focus_filled / len(asked), 1) if asked else None,
        "next_focus_asked_steps": len(asked),
        "chains_last_step_without_next_focus": ended_empty,
        "thoughts_saying_unattended_for": unattended,
    }


def _fetch_psql(container: str, hours: int) -> list[tuple[Any, ...]]:
    sql = (
        "SELECT json_build_array(thought_json->>'chain_id', "
        "COALESCE((thought_json->>'thought_index')::int, -1), "
        "COALESCE(thought_json->'coalition'->>'selected_open_loop_id', ''), "
        "interpretation, COALESCE(thought_json->>'next_focus', ''), created_at::text) "
        f"FROM substrate_reverie_thought WHERE created_at > now() - make_interval(hours => {int(hours)}) "
        "ORDER BY created_at"
    )
    out = subprocess.run(
        ["docker", "exec", container, "psql", "-U", "postgres", "-d", "conjourney", "-Atc", sql],
        check=True, capture_output=True, text=True,
    ).stdout
    return [tuple(json.loads(line)) for line in out.splitlines() if line.strip()]


def _fetch_dsn(dsn: str, hours: int) -> list[tuple[Any, ...]]:
    from sqlalchemy import create_engine, text

    eng = create_engine(dsn)
    with eng.connect() as c:
        return [tuple(r) for r in c.execute(text(_SQL.format(hours=int(hours))))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--psql-container", default="")
    args = ap.parse_args()
    if args.psql_container:
        rows = _fetch_psql(args.psql_container, args.hours)
    else:
        dsn = os.getenv("REVERIE_EVAL_DSN", "postgresql://postgres:postgres@localhost:55432/conjourney")
        rows = _fetch_dsn(dsn, args.hours)
    report = summarize(rows)
    report["window_hours"] = args.hours
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
