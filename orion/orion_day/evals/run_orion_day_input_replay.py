#!/usr/bin/env python3
"""Replay Orion's day letter input, old vs new, side by side (living Prime design #3224, constraint 7).

For each date, gathers the day READ-ONLY from Postgres once, then builds:

* old -- ``budget.build_llm_view`` (the pile the letter was written from until this patch);
* new -- ``day_input.render_day_input`` (Juniper protected, reveries by theme, self-sense once,
  readings and news as a trimmed tail, the temporal core).

Prints, per date and side by side: total size, the share each source got, how much of Juniper's
own words are present, reverie theme spread, self-sense answers shown per question, and whether
every [arc:] / [event:] ref resolves. A review aid for Juniper's read, not a metric: nothing here
feeds a model or a gate. Exit 1 when a new-side check fails (Juniper's words missing, a question
shown twice, a ref that does not resolve).

``--dump DIR`` also writes each side's digest and the letter turn's rendered user message
(``orion/orion_day/letter_turn.py``)
to DIR, so the letters can be generated from them.

Usage:
    python orion/orion_day/evals/run_orion_day_input_replay.py --dsn postgresql://... \\
        --date 2026-11-09 --date 2026-11-10 [--dump /tmp/orion-day-replay]
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


from orion.orion_day.budget import build_llm_view, material_refs  # noqa: E402
from orion.orion_day.day_input import SOURCES, render_day_input  # noqa: E402
from orion.orion_day.gather import gather_orion_day  # noqa: E402
from orion.schemas.orion_day import OrionDayLlmViewV1, OrionDayMaterialV1  # noqa: E402

# Old digest sections by their "## " heading prefix (budget._full_text_blocks / _render_reveries).
OLD_HEADINGS = (
    ("## My day as my own chronology", "chronology"), ("## Curiosity", "curiosity"),
    ("## Self-sense answers", "self_sense"), ("## Readings", "readings"), ("## Dreams", "dreams"),
    ("## Corrections from conversations with Juniper", "juniper"), ("## Conversations with Juniper", "juniper"),
    ("## Changes to my code", "github"), ("## World pulse digest", "world_pulse"),
    ("## The street outside", "walkway"), ("## Visual reveries", "visual_reveries"), ("## Reveries", "reveries"),
)
_REVERIE_REF = re.compile(r"\[reverie:([0-9a-f-]{1,8})\]")
_SELF_SENSE_HEAD = re.compile(r"^### \[self_sense:[^:\]]+:([^\]]+)\]", re.MULTILINE)


def old_shares(view: OrionDayLlmViewV1) -> dict[str, int]:
    """Characters per source in an old digest, by splitting it at its "## " headings."""
    chars: Counter = Counter()
    text = view.digest_md
    starts = [m.start() for m in re.finditer(r"^## ", text, flags=re.MULTILINE)]
    chars["frame"] += starts[0] if starts else len(text)
    for i, at in enumerate(starts):
        chunk = text[at: starts[i + 1] if i + 1 < len(starts) else len(text)]
        source = next((s for prefix, s in OLD_HEADINGS if chunk.startswith(prefix)), "frame")
        chars[source] += len(chunk)
    return dict(chars)


def juniper_words(material: OrionDayMaterialV1, digest: str) -> tuple[int, int]:
    """(characters of Juniper's own words present verbatim in the digest, characters she wrote)."""
    turns = [t.prompt for t in material.juniper_chat if t.prompt]
    return sum(len(p) for p in turns if p in digest), sum(len(p) for p in turns)


def reverie_spread(material: OrionDayMaterialV1, digest: str) -> dict[str, object]:
    theme_of = {c.chain_id: (c.theme_key or "unthemed") for c in material.reverie_chains}
    by_short = {t.thought_id[:8]: t for t in material.reverie_thoughts}
    shown = [by_short[s] for s in dict.fromkeys(_REVERIE_REF.findall(digest)) if s in by_short]
    themes_all = {theme_of.get(t.chain_id or "", "unthemed") for t in material.reverie_thoughts}
    themes = Counter(theme_of.get(t.chain_id or "", "unthemed") for t in shown)
    pe = sum(1 for t in shown if "prediction error" in t.interpretation.lower())
    return {"shown": len(shown), "themes_shown": len(themes), "themes_total": len(themes_all),
            "largest_theme_pct": round(100 * max(themes.values()) / len(shown)) if shown else 0,
            "prediction_error_pct": round(100 * pe / len(shown)) if shown else 0}


def self_sense_shown(digest: str) -> dict[str, int]:
    return dict(Counter(_SELF_SENSE_HEAD.findall(digest)))


def unresolved(view: OrionDayLlmViewV1, material: OrionDayMaterialV1) -> list[str]:
    known = material_refs(material)
    return [r for r in view.included_refs if r not in known]


def _render_prompt(view: OrionDayLlmViewV1, material: OrionDayMaterialV1) -> str:
    from orion.orion_day.letter_turn import build_letter_prompt
    from orion.schemas.orion_day import OrionDayRunBriefV1

    brief = OrionDayRunBriefV1(letter_date=material.letter_date, timezone=material.timezone,
                               window_start=material.window_start, window_end=material.window_end,
                               material=material, llm_view=view)
    return build_letter_prompt(brief)


def report(material: OrionDayMaterialV1, dump: Path | None = None) -> bool:
    old, new = build_llm_view(material), render_day_input(material)
    old_chars, new_chars = old_shares(old), dict(new.source_chars)
    print(f"\n=== {material.letter_date}  (old | new)")
    print(f"  digest chars          {len(old.digest_md):>9,} | {len(new.digest_md):>9,}")
    print(f"  approx tokens         {old.approx_tokens:>9,} | {new.approx_tokens:>9,}")
    print("  share by source (% of digest):")
    for name in SOURCES:
        o, n = old_chars.get(name, 0), new_chars.get(name, 0)
        if o or n:
            print(f"    {name:16s} {100 * o / max(1, len(old.digest_md)):6.1f}% ({o:>7,}) | "
                  f"{100 * n / max(1, len(new.digest_md)):6.1f}% ({n:>7,})")
    jo, total = juniper_words(material, old.digest_md)
    jn, _ = juniper_words(material, new.digest_md)
    talks = sum(1 for t in material.juniper_chat if t.prompt)
    unprompted = sum(1 for t in material.juniper_chat if not t.prompt)
    print(f"  Juniper's own words   {jo:>5,} of {total:,} chars | {jn:>5,} of {total:,} chars "
          f"({talks} turns she wrote, {unprompted} messages I sent unprompted)")
    ro, rn = reverie_spread(material, old.digest_md), reverie_spread(material, new.digest_md)
    print(f"  reveries shown        {ro['shown']:>9} | {rn['shown']:>9}   (of {len(material.reverie_thoughts)} thoughts)")
    print(f"  reverie themes shown  {ro['themes_shown']:>5} of {ro['themes_total']:<3}| {rn['themes_shown']:>5} of {rn['themes_total']}")
    print(f"  largest theme share   {ro['largest_theme_pct']:>8}% | {rn['largest_theme_pct']:>8}%")
    print(f"  'prediction error'    {ro['prediction_error_pct']:>8}% | {rn['prediction_error_pct']:>8}%   "
          "(shown reveries naming it; review aid only)")
    so, sn = self_sense_shown(old.digest_md), self_sense_shown(new.digest_md)
    print(f"  self-sense answers    {sum(so.values()):>9} | {sum(sn.values()):>9}   "
          f"({len(material.self_sense)} answers, {len({a.question_key for a in material.self_sense})} questions)")
    print(f"    per question old: {dict(sorted(so.items()))}")
    print(f"    per question new: {dict(sorted(sn.items()))}")
    arcs_o = sum(1 for r in old.included_refs if r.startswith("arc:"))
    arcs_n = sum(1 for r in new.included_refs if r.startswith(("arc:", "event:")))
    bad_o, bad_n = unresolved(old, material), unresolved(new, material)
    print(f"  arc/event refs        {arcs_o:>9} | {arcs_n:>9}")
    print(f"  unresolved refs       {len(bad_o):>9} | {len(bad_n):>9}")
    print(f"  curiosity clip cap    {old.condensed.full_text_clip_chars or 'none':>9} | "
          f"{new.condensed.full_text_clip_chars or 'none':>9}")
    if dump is not None:
        dump.mkdir(parents=True, exist_ok=True)
        for side, view in (("old", old), ("new", new)):
            (dump / f"{material.letter_date}-{side}-digest.md").write_text(view.digest_md)
            (dump / f"{material.letter_date}-{side}-note-prompt.txt").write_text(_render_prompt(view, material))
        print(f"  wrote {dump}/{material.letter_date}-{{old,new}}-{{digest.md,note-prompt.txt}}")
    checks = {
        "juniper_in_full": jn == total,
        "self_sense_once": all(n == 1 for n in sn.values()),
        "refs_resolve": not bad_n,
    }
    print("  checks (new): " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in checks.items()))
    return all(checks.values())


async def _gather(dsn: str, dates: list[date]) -> list[OrionDayMaterialV1]:
    import asyncpg

    conn = await asyncpg.connect(dsn, server_settings={"default_transaction_read_only": "on"})
    try:
        return [await gather_orion_day(conn, d) for d in dates]
    finally:
        await conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--date", action="append", required=True)
    parser.add_argument("--dump", type=Path)
    args = parser.parse_args()
    materials = asyncio.run(_gather(args.dsn, [date.fromisoformat(d) for d in args.date]))
    ok = True
    for m in materials:
        errors = {k: s.error for k, s in m.sources.items() if s.status == "error"}
        if errors:
            print(f"\n{m.letter_date}: sources that could not be read: {errors}")
        ok = report(m, args.dump) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
