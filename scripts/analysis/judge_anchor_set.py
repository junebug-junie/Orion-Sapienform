#!/usr/bin/env python3
"""Juniper's hand-labelled anchor for the finding-register judge: build the unlabelled set, then score.

Juniper, 2026-11-25 (self-reflection drift spec, Missing question 3): the judge's anchor becomes a set of
about 100 items that SHE labels by hand. Until she has labelled it, the frozen 10-20 fixture
(``scripts/analysis/fixtures/curiosity_register/reference_labels_2026-10-20.jsonl``) stays the agreement
check. No outside model labels or pre-labels this set, and the file never shows the judge's own label (so
nothing anchors her).

``build`` (read-only): a balanced sample across organs and lines, ``--per-organ`` items each (deterministic,
md5 of the row id; an organ with fewer items gives what it has and the rest is filled evenly from the others,
up to ``--total``, default 100), from local data only (``reflection_register_sources``). Writes a tab-separated file:

    label <TAB> n <TAB> organ <TAB> ref <TAB> question <TAB> text

``label`` is empty. Juniper writes ONE word per row: ``philosophical``, ``operational`` or ``no_finding``
(``p`` / ``o`` / ``n`` also accepted). Rows left empty are skipped by ``score``. The file lives outside the
repo (it holds Orion's own writing, including messages to Juniper): default
``~/.orion/judge-anchor/anchor_set_v1.tsv``. It refuses to overwrite a file that already holds labels.

``score``: calls the judge through ONE GPU pool operator hold on ``--lane`` (orion/evals/leased_llm.py,
default ``agent``; every call goes to orion-llm-gateway under the hold, #3921) for every labelled row, on the exact question and text
in the file (the item as she read it), and prints three-way agreement and kappa, philosophical-or-not
kappa, and phil-agree (share of her philosophical rows the judge also calls philosophical), per organ and
overall. ``--export`` writes the labels without text (``ref``, ``organ``, ``label_juniper``) as a jsonl
fixture that can be committed, as the 10-20 fixture is.

    python3 scripts/analysis/judge_anchor_set.py build [--start 2026-10-01 --end 2026-11-25] [--per-organ 15]
    ORION_BUS_URL=redis://<tailnet-ip>:6379/0 python3 scripts/analysis/judge_anchor_set.py score [--lane agent] [--export <path>]

A recall of the hold prints the rows scored so far and exits 3; ``--export`` is not written then.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reflection_register_sources import DEFAULT_LANE, ORGANS, Item, judge_one, read_items, sample  # noqa: E402

from orion.evals.leased_llm import (EXIT_RECALLED, EXIT_UNAVAILABLE, OPERATOR_LANES, LaneRecalled,  # noqa: E402
                                    LaneUnavailable, operator_lane_sync)

HOLDER = "operator:judge-anchor-set"

DEFAULT_PATH = Path.home() / ".orion" / "judge-anchor" / "anchor_set_v1.tsv"
HEADER = ("label", "n", "organ", "ref", "question", "text")
LABELS = {"philosophical": "philosophical", "p": "philosophical", "operational": "operational", "o": "operational",
          "no_finding": "no_finding", "n": "no_finding", "nofinding": "no_finding", "no-finding": "no_finding"}


def _cell(text: str) -> str:
    return " ".join(str(text or "").replace("\t", " ").split())


def write_tsv(path: Path, items: list[Item]) -> None:
    lines = ["\t".join(HEADER)]
    for n, it in enumerate(items, 1):
        lines.append("\t".join(("", str(n), it.organ, it.ref, _cell(it.question), _cell(it.text[:700]))))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_tsv(path: Path) -> list[dict]:
    rows = []
    for k, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if k == 0 or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != len(HEADER):
            raise ValueError(f"row {k}: {len(parts)} columns, expected {len(HEADER)}")
        rows.append(dict(zip(HEADER, parts)))
    return rows


def parse_label(raw: str) -> str | None:
    """Her one word, normalised; None when empty. Raises on a word that is not a label."""
    word = str(raw or "").strip().lower()
    if not word:
        return None
    if word not in LABELS:
        raise ValueError(f"not a label: {raw!r} (use philosophical, operational or no_finding)")
    return LABELS[word]


def kappa(a: list[str], b: list[str]) -> tuple[float | None, float | None]:
    n = len(a)
    if not n:
        return None, None
    agree = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    return agree, (None if pe >= 1.0 else (agree - pe) / (1 - pe))


def score_rows(pairs: list[tuple[str, str]]) -> dict:
    """(juniper, judge) pairs with both labels -> agreement numbers. Pure."""
    jun = [p[0] for p in pairs]
    jud = [p[1] for p in pairs]
    agree, k3 = kappa(jun, jud)
    _, k2 = kappa(["p" if x == "philosophical" else "x" for x in jun], ["p" if x == "philosophical" else "x" for x in jud])
    phil = [(a, b) for a, b in pairs if a == "philosophical"]
    return {"n": len(pairs), "agree": agree, "kappa3": k3, "kappa_phil_vs_not": k2,
            "phil_agree": (sum(1 for _, b in phil if b == "philosophical") / len(phil)) if phil else None,
            "n_phil": len(phil)}


def pick_balanced(pools: dict[str, list[Item]], *, per_organ: int, total: int) -> list[Item]:
    """``per_organ`` from each organ (each pool already in md5 order); an organ with fewer gives what it has,
    and the shortfall is filled round-robin from the organs that have more, up to ``total``. Pure."""
    taken = {o: min(per_organ, len(p)) for o, p in pools.items()}
    while sum(taken.values()) < total:
        grew = False
        for o, p in pools.items():
            if sum(taken.values()) >= total:
                break
            if taken[o] < len(p):
                taken[o] += 1
                grew = True
        if not grew:
            break
    return [it for o, p in pools.items() for it in p[: taken[o]]]


def cmd_build(args: argparse.Namespace) -> int:
    path: Path = args.out
    if path.exists() and any(parse_label(r["label"]) for r in read_tsv(path)):
        print(f"refused: {path} already holds labels; build to a new path", file=sys.stderr)
        return 2
    end = args.end or datetime.now(timezone.utc).isoformat()
    start = args.start or (datetime.fromisoformat(end.replace("Z", "+00:00")) - timedelta(days=args.days)).isoformat()
    pools = {organ: sample(read_items(organ, start=start, end=end, with_text=True), 10_000) for organ in ORGANS}
    picked = pick_balanced(pools, per_organ=args.per_organ, total=args.total)
    for organ in ORGANS:
        print(f"{organ:22s} {sum(1 for it in picked if it.organ == organ)} of {len(pools[organ])} items", file=sys.stderr)
    # Interleaved by md5 so the file does not read organ by organ (no run of one kind primes the next).
    from reflection_register_sources import md5_key

    picked.sort(key=lambda it: md5_key("order:" + it.ref))
    write_tsv(path, picked)
    print(f"wrote {len(picked)} unlabelled rows to {path} (window {start} .. {end})")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    rows = [r for r in read_tsv(args.path) if parse_label(r["label"])]
    if not rows:
        print("no labelled rows yet: the frozen 10-20 fixture stays the agreement check", file=sys.stderr)
        return 1
    results: list[dict] = []
    recalled = None
    try:
        with operator_lane_sync(args.lane, holder=HOLDER) as lane:
            for r in rows:
                it = Item(r["organ"], r["ref"], r["question"], r["text"])
                label, _reason, err, model = judge_one(lane, it)
                results.append({**r, "juniper": parse_label(r["label"]), "judge": label, "error": err, "model": model})
    except LaneRecalled as exc:
        recalled = exc   # score what was judged, then stop (leased_llm: nothing resumes on its own)
    except LaneUnavailable as exc:
        print(f"REFUSED on {args.lane}: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    if recalled is not None:
        print(f"RECALLED on {args.lane} after {len(results)}/{len(rows)} rows: {recalled} (partial score below)",
              file=sys.stderr)
    errors = Counter(x["organ"] for x in results if not x["judge"])
    print(f"labelled rows: {len(results)}; judge errors: {sum(errors.values())} (left out)")
    for organ in [None, *sorted({x['organ'] for x in results})]:
        pairs = [(x["juniper"], x["judge"]) for x in results if x["judge"] and (organ is None or x["organ"] == organ)]
        s = score_rows(pairs)
        f = (lambda v: "None" if v is None else f"{v:.3f}")
        print(f"{organ or 'ALL':22s} n={s['n']:3d} agree={f(s['agree'])} kappa3={f(s['kappa3'])} "
              f"kappa_phil_vs_not={f(s['kappa_phil_vs_not'])} phil_agree={f(s['phil_agree'])} (n_phil={s['n_phil']})")
    if recalled is not None:
        return EXIT_RECALLED
    if args.export:
        args.export.write_text("".join(json.dumps({"ref": x["ref"], "organ": x["organ"], "label_juniper": x["juniper"]}) + "\n"
                                       for x in results))
        print(f"exported {len(results)} labels (no text) to {args.export}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, default=DEFAULT_PATH)
    b.add_argument("--days", type=float, default=56.0)
    b.add_argument("--start")
    b.add_argument("--end")
    b.add_argument("--per-organ", type=int, default=14)
    b.add_argument("--total", type=int, default=100)
    s = sub.add_parser("score")
    s.add_argument("--path", type=Path, default=DEFAULT_PATH)
    s.add_argument("--lane", default=DEFAULT_LANE, choices=sorted(OPERATOR_LANES),
                   help="operator lane to judge on (one GPU pool hold, through the gateway)")
    s.add_argument("--export", type=Path)
    args = ap.parse_args(argv)
    return cmd_build(args) if args.cmd == "build" else cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
