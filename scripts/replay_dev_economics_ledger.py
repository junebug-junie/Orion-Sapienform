#!/usr/bin/env python3
"""Dev-economics ledger replay -- the "Recommended next patch" from
docs/superpowers/specs/2026-07-30-dev-economics-signal-design.md: run
claude_code_ingest.py's session-usage parser against this machine's real
local Claude Code transcript history and produce a normalized table Juniper
can eyeball directly (token totals, model mix, rough session count) --
before pricing, before Cursor, before any coupling decision to
structural_mass, per that doc's own explicit phasing.

Read-only. Writes only aggregate/per-session numeric summaries (token
counts, word counts, timestamps, model/effort labels) -- never raw message
text, same privacy discipline as the affective-state replay
(orion/cocreation/affective_signals.py's own docstring).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.dev_economics.claude_code_ingest import (  # noqa: E402
    DEFAULT_PROJECTS_ROOT,
    iter_all_session_usage_records,
)
from orion.dev_economics.pricing import estimate_session_cost_usd  # noqa: E402


def _record_cost_usd(record):
    """One record's estimated cost, or None if unpriceable (unknown model,
    or a date this table's known windows don't reach). A record touching
    more than one distinct model (rare -- e.g. a mid-session model switch)
    is priced against its first-seen model only; the real total then
    slightly under- or over-states cost for that one record. Accepted
    simplification -- splitting cost within a record would need re-parsing
    per line, out of scope for this replay."""
    if not record.models or record.started_at is None:
        return None
    return estimate_session_cost_usd(
        model=record.models[0],
        observed_at=record.started_at,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        cache_creation_input_tokens=record.cache_creation_input_tokens,
        cache_read_input_tokens=record.cache_read_input_tokens,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts-root", default=str(DEFAULT_PROJECTS_ROOT))
    parser.add_argument("--out-dir", default="/tmp/dev-economics-ledger-replay")
    args = parser.parse_args()

    transcripts_root = Path(args.transcripts_root).expanduser()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = list(iter_all_session_usage_records(transcripts_root))
    top_level = [r for r in records if not r.is_subagent]
    subagent = [r for r in records if r.is_subagent]

    model_counter: Counter[str] = Counter()
    effort_counter: Counter[str] = Counter()
    for r in records:
        model_counter.update(r.models)
        effort_counter.update(r.effort_tiers)

    def _sum(field: str, pool) -> int:
        return sum(getattr(r, field) for r in pool)

    def _durations(pool) -> list[float]:
        return [r.duration_sec for r in pool if r.duration_sec is not None and r.duration_sec > 0]

    summary = {
        "transcripts_root": str(transcripts_root),
        "top_level_session_count": len(top_level),
        "subagent_transcript_count": len(subagent),
        "total_input_tokens": _sum("input_tokens", records),
        "total_output_tokens": _sum("output_tokens", records),
        "total_cache_creation_input_tokens": _sum("cache_creation_input_tokens", records),
        "total_cache_read_input_tokens": _sum("cache_read_input_tokens", records),
        "total_assistant_word_count": _sum("assistant_word_count", records),
        "total_human_word_count": _sum("human_word_count", top_level),
        "model_mix": dict(model_counter.most_common()),
        "effort_mix": dict(effort_counter.most_common()),
    }
    top_level_durations = _durations(top_level)
    if top_level_durations:
        summary["top_level_session_duration_sec_mean"] = statistics.fmean(top_level_durations)
        summary["top_level_session_duration_sec_median"] = statistics.median(top_level_durations)
        summary["top_level_session_duration_sec_max"] = max(top_level_durations)

    costed = [(r, _record_cost_usd(r)) for r in records]
    cost_by_id = {id(r): c for r, c in costed}

    per_session = [
        {
            "session_id": r.session_id,
            "is_subagent": r.is_subagent,
            "models": list(r.models),
            "effort_tiers": list(r.effort_tiers),
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "cache_creation_input_tokens": r.cache_creation_input_tokens,
            "cache_read_input_tokens": r.cache_read_input_tokens,
            "total_tokens": r.total_tokens,
            "estimated_cost_usd": (
                round(cost_by_id[id(r)].total_usd, 4) if cost_by_id.get(id(r)) else None
            ),
            "assistant_turn_count": r.assistant_turn_count,
            "assistant_word_count": r.assistant_word_count,
            "human_turn_count": r.human_turn_count,
            "human_word_count": r.human_word_count,
            "duration_sec": r.duration_sec,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "ended_at": r.ended_at.isoformat() if r.ended_at else None,
        }
        for r in records
    ]
    per_session.sort(key=lambda r: r["started_at"] or "")

    (out_dir / "session_records.json").write_text(json.dumps(per_session, indent=2), encoding="utf-8")

    top_by_tokens = sorted(records, key=lambda r: r.total_tokens, reverse=True)[:10]

    total_cost_usd = sum(c.total_usd for _, c in costed if c is not None)
    priced_count = sum(1 for _, c in costed if c is not None)
    unpriced_models = sorted(
        {r.models[0] for r, c in costed if c is None and r.models}
    )

    lines = [
        "# Dev-economics ledger replay",
        "",
        f"Transcripts root: `{transcripts_root}`",
        f"Top-level session transcripts: {summary['top_level_session_count']}",
        f"Subagent transcripts: {summary['subagent_transcript_count']}",
        "",
        "## Real token totals (all transcripts, top-level + subagent combined)",
        "",
        f"- input_tokens: {summary['total_input_tokens']:,}",
        f"- output_tokens: {summary['total_output_tokens']:,}",
        f"- cache_creation_input_tokens: {summary['total_cache_creation_input_tokens']:,}",
        f"- cache_read_input_tokens: {summary['total_cache_read_input_tokens']:,}",
        "",
        "## Word counts (reading-burden proxy)",
        "",
        f"- total assistant visible word count: {summary['total_assistant_word_count']:,}",
        f"- total human (top-level sessions only) word count: {summary['total_human_word_count']:,}",
        "",
        "## Model mix (transcript count per model seen)",
        "",
    ]
    for model, count in summary["model_mix"].items():
        lines.append(f"- {model}: {count}")

    lines += [
        "",
        "## Real $ cost estimate (orion/dev_economics/pricing.py, real sourced rates)",
        "",
        f"- estimated total: ${total_cost_usd:,.2f}",
        f"- records priced: {priced_count} / {len(records)}",
    ]
    if unpriced_models:
        lines.append(f"- unpriced models (no rate entry, cost excluded): {', '.join(unpriced_models)}")

    lines += ["", "## Effort mix", ""]
    for effort, count in summary["effort_mix"].items():
        lines.append(f"- {effort}: {count}")

    if "top_level_session_duration_sec_mean" in summary:
        lines += [
            "",
            "## Top-level session duration (real elapsed time, first to last timestamp)",
            "",
            f"- mean: {summary['top_level_session_duration_sec_mean']/60:.1f} min",
            f"- median: {summary['top_level_session_duration_sec_median']/60:.1f} min",
            f"- max: {summary['top_level_session_duration_sec_max']/3600:.1f} hr",
        ]

    lines += ["", "## Top 10 transcripts by total_tokens", ""]
    lines += [
        "| session_id | subagent | model | total_tokens | est. cost | duration |",
        "|---|---|---|---|---|---|",
    ]
    for r in top_by_tokens:
        dur = f"{r.duration_sec/60:.1f}min" if r.duration_sec else "n/a"
        cost = cost_by_id.get(id(r))
        cost_str = f"${cost.total_usd:,.2f}" if cost else "n/a"
        lines.append(
            f"| {r.session_id[:8]} | {r.is_subagent} | {','.join(r.models) or '?'} | "
            f"{r.total_tokens:,} | {cost_str} | {dur} |"
        )

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote {report_path} and {out_dir / 'session_records.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
