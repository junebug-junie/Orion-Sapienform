"""Generate a single Markdown weekly report from daily rollups."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

from .metrics import DailyMetricsV1


def _load_day(runs_dir: Path, day: date) -> DailyMetricsV1 | None:
    target = runs_dir / f"{day.isoformat()}.json"
    if not target.exists():
        return None
    return DailyMetricsV1.model_validate(json.loads(target.read_text(encoding="utf-8")))


def _date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current = current + timedelta(days=1)


def _verdict(start_score: float | None, end_score: float | None) -> str:
    if start_score is None or end_score is None:
        return "Insufficient data — substrate did not record a full week."
    delta = end_score - start_score
    if delta > 0.05:
        return (
            f"Yes — substrate health rose by {delta:+.2f}. Shared grammar is "
            "outperforming bespoke organ state."
        )
    if delta < -0.05:
        return (
            f"No — substrate health fell by {delta:+.2f}. Shared substrate is "
            "currently noisier than the bespoke alternative."
        )
    return (
        f"Inconclusive — substrate health barely moved ({delta:+.2f}). Run "
        "longer or wire more organs."
    )


def generate_week_report(
    *,
    start_date: date,
    end_date: date,
    runs_dir: str | Path,
    out_path: str | Path | None = None,
) -> str:
    """Render a markdown report over the [start_date, end_date] inclusive range.

    Writes to ``out_path`` if provided; always returns the markdown body.
    """

    runs_path = Path(runs_dir)
    days = list(_date_range(start_date, end_date))
    loaded: list[tuple[date, DailyMetricsV1 | None]] = [
        (day, _load_day(runs_path, day)) for day in days
    ]
    present = [(day, metrics) for day, metrics in loaded if metrics is not None]

    lines: list[str] = []
    lines.append(f"# Substrate Experiment Report — {start_date} → {end_date}")
    lines.append("")
    lines.append("**Question:** Did the shared substrate become more useful than bespoke organ state?")
    lines.append("")

    if not present:
        lines.append("_No daily rollups were found for this window._")
        body = "\n".join(lines) + "\n"
        if out_path:
            Path(out_path).write_text(body, encoding="utf-8")
        return body

    start_score = present[0][1].substrate_health_score
    end_score = present[-1][1].substrate_health_score
    lines.append(f"**Verdict:** {_verdict(start_score, end_score)}")
    lines.append("")

    lines.append("## Daily snapshot")
    lines.append("")
    lines.append(
        "| date | molecules | organs | resonance | cross-organ | reinforce | decay | orphan% | health |"
    )
    lines.append("|------|-----------|--------|-----------|-------------|-----------|-------|---------|--------|")
    for day, metrics in loaded:
        if metrics is None:
            lines.append(f"| {day} | — | — | — | — | — | — | — | — |")
            continue
        organs = ", ".join(sorted(metrics.organ_coverage.by_organ)) or "—"
        lines.append(
            f"| {day} | {metrics.molecule_count} | {organs} | "
            f"{metrics.resonance_hits} | {metrics.cross_organ_reuse} | "
            f"{metrics.reinforcement_count} | {metrics.decay_count} | "
            f"{metrics.orphan_molecule_rate * 100:.0f}% | "
            f"{metrics.substrate_health_score:+.2f} |"
        )
    lines.append("")

    last = present[-1][1]
    lines.append("## End-of-window gradient distribution")
    lines.append("")
    lines.append("| gradient | min | mean | max |")
    lines.append("|----------|-----|------|-----|")
    for stat in last.gradient_distribution:
        lines.append(f"| {stat.key} | {stat.min:.2f} | {stat.mean:.2f} | {stat.max:.2f} |")
    lines.append("")

    if last.contradiction_clusters:
        lines.append("## Contradiction clusters (end of window)")
        lines.append("")
        for cluster in last.contradiction_clusters[:5]:
            lines.append(
                f"- atoms={cluster.shared_atoms} sum={cluster.contradiction_sum:.2f} "
                f"members={len(cluster.molecule_ids)}"
            )
        lines.append("")

    body = "\n".join(lines) + "\n"
    if out_path:
        Path(out_path).write_text(body, encoding="utf-8")
    return body
