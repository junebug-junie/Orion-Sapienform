"""Eval: tune the mesh-edge shingle run length (issue #3554).

Two parts:

1. **Fixture set** (always runs; deterministic, no I/O). Made-up guarded sentences and outgoing
   texts (``mesh_edge_paraphrase_set.py``). For each run length it reports how many near-verbatim
   edits are refused (want: all), how many real rewordings are refused (expected: few; the stated
   limit), and how many unrelated texts are refused (want: none).

2. **Live false-positive check** (``--live``; read-only). Loads the real guarded sentences from
   Postgres (the same SQL the guard uses) and fetches recent public GitHub PR and issue bodies and
   comments with ``gh`` (read-only). Every one of those was already posted publicly, so each refusal
   is either a false positive or a past leak. It prints counts and ids only (``pr#N`` / ``issue#N``
   and the guarded source id): never the matched text. Also reports the guard build time and the
   per-check latency.

Pass rule (fixture part, at the shipped ``SHINGLE_WORDS``): every near-verbatim case refused except
those whose longest shared run is shorter than the threshold, and zero unrelated cases refused.

Usage::

    python orion/memory/evals/run_mesh_edge_shingle_eval.py
    python orion/memory/evals/run_mesh_edge_shingle_eval.py --live --dsn postgresql://... --limit 200
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.memory.evals.mesh_edge_paraphrase_set import CASES, GUARDED, LABELS  # noqa: E402
from orion.memory.mesh_edge import (  # noqa: E402
    CORE_PAGES_SQL, GUARD_MEMORY_SQL, SHINGLE_WORDS, MeshEdgeGuard, fold_words, normalize, sentences,
)

THRESHOLDS = (5, 6, 7, 8, 9, 10, 12)


def longest_shared_run(text: str, guarded: Sequence[Tuple[str, str]]) -> int:
    """Longest run of consecutive folded words ``text`` shares with any one guarded sentence."""
    out = fold_words(normalize(text))
    best = 0
    for _src, body in guarded:
        for sent in sentences(body):
            g = fold_words(normalize(sent))
            # Classic longest-common-substring over words.
            prev = [0] * (len(g) + 1)
            for a in out:
                cur = [0] * (len(g) + 1)
                for j, b in enumerate(g, 1):
                    if a == b:
                        cur[j] = prev[j - 1] + 1
                        best = max(best, cur[j])
                prev = cur
    return best


def fixture_table() -> Dict[int, Dict[str, Tuple[int, int]]]:
    table: Dict[int, Dict[str, Tuple[int, int]]] = {}
    for n in (0,) + THRESHOLDS:
        guard = MeshEdgeGuard.from_texts(GUARDED, shingle_n=n)
        row: Dict[str, Tuple[int, int]] = {}
        for label in LABELS:
            texts = [t for lab, t in CASES if lab == label]
            refused = sum(1 for t in texts if guard.find(t))
            row[label] = (refused, len(texts))
        table[n] = row
    return table


def fixture_passes(n: int = SHINGLE_WORDS) -> Tuple[bool, List[str]]:
    guard = MeshEdgeGuard.from_texts(GUARDED, shingle_n=n)
    problems: List[str] = []
    for i, (label, text) in enumerate(CASES):
        refused = bool(guard.find(text))
        if label == "unrelated" and refused:
            problems.append(f"case {i}: unrelated text refused")
        if label == "near_verbatim" and not refused and longest_shared_run(text, GUARDED) >= n:
            problems.append(f"case {i}: near-verbatim edit with a {n}+ word run passed")
    return not problems, problems


# --- live part --------------------------------------------------------------------------------


def _load_live_items(dsn: str) -> List[Tuple[str, str]]:
    import psycopg2  # type: ignore
    import psycopg2.extras  # type: ignore

    conn = psycopg2.connect(dsn, connect_timeout=5)
    try:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(CORE_PAGES_SQL)
            items = [(str(r["concept_id"]), str(r["content"] or "")) for r in cur.fetchall()]
            cur.execute(GUARD_MEMORY_SQL)
            items.extend((str(r["source"]), str(r["text"] or "")) for r in cur.fetchall())
    finally:
        conn.close()
    return items


def _gh_json(args: List[str]) -> list:
    out = subprocess.run(["gh", *args], check=True, capture_output=True, text=True, timeout=300).stdout
    return json.loads(out or "[]")


def fetch_public_writes(limit: int) -> List[Tuple[str, str]]:
    """(id, text) for recent PR and issue bodies and their comments. Read-only ``gh`` calls."""
    docs: List[Tuple[str, str]] = []
    fields = "number,body,comments"
    for kind in ("pr", "issue"):
        for item in _gh_json([kind, "list", "--state", "all", "--limit", str(limit), "--json", fields]):
            num = item.get("number")
            docs.append((f"{kind}#{num}", str(item.get("body") or "")))
            for k, c in enumerate(item.get("comments") or []):
                docs.append((f"{kind}#{num}/comment{k}", str(c.get("body") or "")))
    return [(i, t) for i, t in docs if t.strip()]


def live_report(dsn: str, limit: int, show_ids: bool) -> Dict[str, object]:
    t0 = time.perf_counter()
    items = _load_live_items(dsn)
    load_ms = (time.perf_counter() - t0) * 1000
    docs = fetch_public_writes(limit)
    words = [len(normalize(t).split()) for _i, t in docs]
    report: Dict[str, object] = {
        "guard_rows": len(items), "db_load_ms": round(load_ms, 1),
        "public_docs": len(docs), "public_words_median": int(statistics.median(words)) if words else 0,
        "public_words_max": max(words) if words else 0, "by_threshold": {},
    }
    for n in (0,) + THRESHOLDS:
        t1 = time.perf_counter()
        guard = MeshEdgeGuard.from_texts(items, shingle_n=n)
        build_ms = (time.perf_counter() - t1) * 1000
        lat: List[float] = []
        refused: List[Tuple[str, List[str], List[str]]] = []
        for doc_id, text in docs:
            t2 = time.perf_counter()
            hits = guard.find(text)
            lat.append((time.perf_counter() - t2) * 1000)
            if hits:
                refused.append((doc_id, sorted({h.source for h in hits}), sorted({h.kind for h in hits})))
        lat_sorted = sorted(lat)
        entry: Dict[str, object] = {
            "build_ms": round(build_ms, 1), "phrases": len(guard.phrases), "shingles": len(guard.shingles),
            "refused": len(refused), "refused_rate": round(len(refused) / max(1, len(docs)), 4),
            "check_ms_median": round(statistics.median(lat), 3) if lat else 0.0,
            "check_ms_p95": round(lat_sorted[int(0.95 * (len(lat_sorted) - 1))], 3) if lat else 0.0,
            "check_ms_max": round(max(lat), 3) if lat else 0.0,
        }
        if show_ids:
            # Ids only: the public doc id, the guarded row id and the match kind. Never text.
            entry["refused_ids"] = [f"{d} <- {','.join(s)} ({','.join(k)})" for d, s, k in refused]
        report["by_threshold"][str(n)] = entry
    return report


def run(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--live", action="store_true", help="also run the live false-positive check")
    ap.add_argument("--dsn", default=os.environ.get("ORION_MESH_EDGE_PG_DSN", ""), help="read-only Postgres DSN")
    ap.add_argument("--limit", type=int, default=200, help="recent PRs and issues to fetch (each)")
    ap.add_argument("--show-ids", action="store_true", help="list refused public doc ids and guarded row ids")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    table = fixture_table()
    ok, problems = fixture_passes()
    result: Dict[str, object] = {
        "shipped_threshold": SHINGLE_WORDS,
        "fixture": {str(n): {lab: f"{r}/{t}" for lab, (r, t) in row.items()} for n, row in table.items()},
        "fixture_pass": ok, "fixture_problems": problems,
    }
    if args.live:
        if not args.dsn:
            print("--live needs --dsn or ORION_MESH_EDGE_PG_DSN", file=sys.stderr)
            return 2
        result["live"] = live_report(args.dsn, args.limit, args.show_ids)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"shipped threshold: {SHINGLE_WORDS} words (0 = verbatim only)")
        print("fixture: refused/total   " + "  ".join(f"{lab:>14}" for lab in LABELS))
        for n, row in table.items():
            print(f"  n={n:<3}                 " + "  ".join(f"{r:>2}/{t:<11}" for r, t in row.values()))
        print(f"fixture pass at n={SHINGLE_WORDS}: {ok} {problems or ''}")
        live = result.get("live")
        if live:
            print(f"live: {live['guard_rows']} guarded rows (db load {live['db_load_ms']} ms), "
                  f"{live['public_docs']} public docs (median {live['public_words_median']} words, "
                  f"max {live['public_words_max']})")
            for n, e in live["by_threshold"].items():
                print(f"  n={n:<3} refused {e['refused']}/{live['public_docs']} ({e['refused_rate']:.2%})  "
                      f"build {e['build_ms']} ms  shingles {e['shingles']}  check median {e['check_ms_median']} ms "
                      f"p95 {e['check_ms_p95']} ms max {e['check_ms_max']} ms")
                for line in e.get("refused_ids", []):
                    print(f"        {line}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
