#!/usr/bin/env python3
"""Measure-before-minting replay for the Juniper affective-state signal
(docs/superpowers/specs/2026-07-30-juniper-affective-state-signal-proposal.md,
"What trace proves it worked"). Read-only: parses real local Claude Code
transcripts, scores each session's real typed turns, and reports whether the
resulting swear_frequency / typo_rate signals are non-degenerate and read as
genuinely calm most of the time -- the same two checks root CLAUDE.md's
metric-quality-gate section requires before any new signal is trusted.

Does not write to Postgres, the bus, or any Orion-facing surface -- per the
proposal doc, this producer is not live-wired until this replay is real and
reviewed. Output is a markdown report + a per-session JSON dump (aggregate
scores only, never raw text) under the given --out-dir.

Usage:
    python scripts/replay_juniper_affective_state.py \\
        --transcripts-root ~/.claude/projects \\
        --out-dir /tmp/juniper-affective-replay
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.cocreation.affective_signals import aggregate_scores, score_message  # noqa: E402
from orion.dev_economics.claude_code_ingest import (  # noqa: E402
    DEFAULT_PROJECTS_ROOT,
    iter_all_human_messages,
)


def _session_summaries(transcripts_root: Path) -> list[dict]:
    by_session: dict[str, list] = defaultdict(list)
    for msg in iter_all_human_messages(transcripts_root):
        by_session[msg.session_id].append(msg)

    summaries = []
    for session_id, messages in by_session.items():
        messages.sort(key=lambda m: m.timestamp)
        per_message_scores = [score_message(m.text) for m in messages]
        agg = aggregate_scores(per_message_scores)
        summaries.append(
            {
                "session_id": session_id,
                "message_count": len(messages),
                "first_timestamp": messages[0].timestamp.isoformat(),
                "last_timestamp": messages[-1].timestamp.isoformat(),
                "word_count": agg.word_count,
                "swear_count": agg.swear_count,
                "swear_frequency": agg.swear_frequency,
                "checked_word_count": agg.checked_word_count,
                "unknown_word_count": agg.unknown_word_count,
                "typo_rate": agg.typo_rate,
            }
        )
    summaries.sort(key=lambda s: s["first_timestamp"])
    return summaries


def _non_degenerate_report(values: list[float], label: str) -> dict:
    if not values:
        return {"label": label, "n": 0, "verdict": "NO_DATA"}
    zero_count = sum(1 for v in values if v == 0.0)
    zero_pct = zero_count / len(values)
    report = {
        "label": label,
        "n": len(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "zero_count": zero_count,
        "zero_pct": zero_pct,
        "nonzero_count": len(values) - zero_count,
    }
    if report["stdev"] == 0.0:
        report["verdict"] = "DEGENERATE_FLAT"
    elif zero_pct < 0.05:
        # Every session reading nonzero would itself be suspicious for a
        # frustration/fatigue proxy -- most collaborative work is calm.
        report["verdict"] = "SUSPICIOUS_NEVER_CALM"
    else:
        report["verdict"] = "NON_DEGENERATE"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts-root", default=str(DEFAULT_PROJECTS_ROOT))
    parser.add_argument("--out-dir", default="/tmp/juniper-affective-replay")
    args = parser.parse_args()

    transcripts_root = Path(args.transcripts_root).expanduser()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = _session_summaries(transcripts_root)

    swear_values = [s["swear_frequency"] for s in summaries if s["swear_frequency"] is not None]
    typo_values = [s["typo_rate"] for s in summaries if s["typo_rate"] is not None]

    swear_report = _non_degenerate_report(swear_values, "swear_frequency")
    typo_report = _non_degenerate_report(typo_values, "typo_rate")

    top_swear = sorted(
        (s for s in summaries if s["swear_frequency"]),
        key=lambda s: s["swear_frequency"],
        reverse=True,
    )[:10]
    top_typo = sorted(
        (s for s in summaries if s["typo_rate"]),
        key=lambda s: s["typo_rate"],
        reverse=True,
    )[:10]

    (out_dir / "session_summaries.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# Juniper affective-state signal replay",
        "",
        f"Transcripts root: `{transcripts_root}`",
        f"Sessions with at least one real typed turn: {len(summaries)}",
        "",
        "## swear_frequency",
        "",
        f"- n (sessions with word_count > 0): {swear_report['n']}",
    ]
    if swear_report["n"]:
        report_lines += [
            f"- mean: {swear_report['mean']:.4f}",
            f"- stdev: {swear_report['stdev']:.4f}",
            f"- min/max: {swear_report['min']:.4f} / {swear_report['max']:.4f}",
            f"- calm (== 0.0) sessions: {swear_report['zero_count']} "
            f"({swear_report['zero_pct']*100:.1f}%)",
            f"- verdict: **{swear_report['verdict']}**",
        ]
    report_lines += ["", "## typo_rate", ""]
    report_lines += [f"- n (sessions with checked_word_count > 0): {typo_report['n']}"]
    if typo_report["n"]:
        report_lines += [
            f"- mean: {typo_report['mean']:.4f}",
            f"- stdev: {typo_report['stdev']:.4f}",
            f"- min/max: {typo_report['min']:.4f} / {typo_report['max']:.4f}",
            f"- calm (== 0.0) sessions: {typo_report['zero_count']} "
            f"({typo_report['zero_pct']*100:.1f}%)",
            f"- verdict: **{typo_report['verdict']}**",
        ]

    report_lines += ["", "## Top 10 sessions by swear_frequency", ""]
    report_lines += ["| session_id | first_timestamp | word_count | swear_frequency |", "|---|---|---|---|"]
    for s in top_swear:
        report_lines.append(
            f"| {s['session_id'][:8]} | {s['first_timestamp']} | {s['word_count']} | "
            f"{s['swear_frequency']:.3f} |"
        )

    report_lines += ["", "## Top 10 sessions by typo_rate", ""]
    report_lines += ["| session_id | first_timestamp | checked_word_count | typo_rate |", "|---|---|---|---|"]
    for s in top_typo:
        report_lines.append(
            f"| {s['session_id'][:8]} | {s['first_timestamp']} | {s['checked_word_count']} | "
            f"{s['typo_rate']:.3f} |"
        )

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nWrote {report_path} and {out_dir / 'session_summaries.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
