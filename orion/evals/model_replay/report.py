"""report.md (verdict first), blind_sheet.md + blind_key.json + hand_grades.csv for the hand grade."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

from orion.evals.model_replay.fixture import ReplayTaskV1
from orion.evals.model_replay.scoring import TaskScore

FIDELITY_NOTES = (
    "Prompts: curiosity briefs, reading prompts and self-sense questions are the real frozen ones. Each "
    "harness task first runs the model's own stance pass (production template, offline-reproducible "
    "inputs only), then builds the harness prompt with production's `build_harness_prompt` over that "
    "stance output.",
    "Not reproduced (identical gap for both models): Claude Code's own system prompt and tool "
    "descriptions, WebFetch's summarizer (the replay returns page text), WebSearch (unavailable), MCP "
    "servers (read_recall/read_memory/read_graph, firecrawl, github), live stance inputs (association, "
    "coalition, mind coloring), and the memory digest.",
    "Graph reads and writes hit a scratch copy of orion_worldview/orion_substrate taken at replay start, "
    "so both models see the same graph; SQL reads hit live Postgres read-only.",
    "Sampling pinned per request (temperature 1.0, top_k 20, top_p 0.95, min_p 0, max_tokens 16384, "
    "reasoning_effort xhigh). Server-side the two profiles differ in `reasoning: on` (Q4) vs `auto` "
    "(Bonsai) and in llama.cpp build (stock vs PrismML fork) -- that is part of what is being compared.",
)


def _pct(x: Any) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def write_report(out_dir: Path, summary: dict[str, Any], scores: list[TaskScore], tasks: list[ReplayTaskV1],
                 rng: random.Random) -> None:
    d = summary["decision"]
    pm = summary["per_model"]
    lines = [
        "# Bonsai vs Q4 replay",
        "",
        f"**{d['verdict']}**",
        "",
        "Rule: Bonsai eligible for gpu1 iff finish rate >= 90%, zero misreported writes, and finish rate "
        "within 5 points of Q4.",
        "",
        "| | Q4 (gpu1) | Bonsai (gpu2) |",
        "|---|---|---|",
    ]
    rows = [
        ("tasks scored", "tasks"), ("finished", "finished"), ("empty answers", "empty"),
        ("cut by token limit", "length_cut"), ("misreported writes", "misreported_writes"),
        ("writes never mentioned", "unmentioned_writes"), ("output tokens", "output_tokens"),
        ("model wall time (s)", "elapsed_sec"),
    ]
    q, b = pm.get("q4", {}), pm.get("bonsai", {})
    lines.append(f"| finish rate | {_pct(q.get('finish_rate'))} | {_pct(b.get('finish_rate'))} |")
    for label, key in rows:
        lines.append(f"| {label} | {q.get(key, '')} | {b.get(key, '')} |")
    lines += ["", "## By task kind", "", "| kind | Q4 finished | Bonsai finished |", "|---|---|---|"]
    kinds = sorted(set(q.get("by_kind", {})) | set(b.get("by_kind", {})))
    for k in kinds:
        qk, bk = q.get("by_kind", {}).get(k, {}), b.get("by_kind", {}).get(k, {})
        lines.append(f"| {k} | {qk.get('finished', 0)}/{qk.get('n', 0)} | {bk.get('finished', 0)}/{bk.get('n', 0)} |")
    lines += ["", "## Misreported writes (write-up vs what landed in the graph)", ""]
    bad = [s for s in scores if s.misreported_writes]
    if not bad:
        lines.append("None.")
    for s in bad:
        lines.append(f"- **{s.model} / {s.task_id}**: {s.misreported_writes} misreported")
        for c in s.write_claims.get("claims", []):
            if c["verdict"] in ("phantom", "contradicted"):
                lines.append(f"  - {c['verdict']}: {c['prior_id']} {c['before']} -> {c['after']} "
                             f"({c['evidence']}); said: \"{c['sentence'][:200]}\"")
    lines += ["", "## Per task", "", "| task | kind | Q4 | Bonsai |", "|---|---|---|---|"]
    by = {(s.task_id, s.model): s for s in scores}
    for t in tasks:
        cells = []
        for m in ("q4", "bonsai"):
            s = by.get((t.task_id, m))
            if s is None:
                cells.append("not run")
                continue
            tag = "ok" if s.finished else (s.end or "fail")
            if s.contract_ok is False:
                tag += " (contract)"
            if s.misreported_writes:
                tag += f" misreported={s.misreported_writes}"
            cells.append(f"{tag}, {s.elapsed_sec:.0f}s, {s.output_tokens} tok")
        lines.append(f"| {t.task_id} | {t.kind} | {cells[0]} | {cells[1]} |")
    lines += ["", "## What this replay does and does not reproduce", ""] + [f"- {n}" for n in FIDELITY_NOTES]
    lines += ["", "Hand grade: `blind_sheet.md` (A/B unlabeled), fill `hand_grades.csv`, unblind with `blind_key.json`.", ""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_blind_sheet(out_dir, scores, tasks, rng)


def write_blind_sheet(out_dir: Path, scores: list[TaskScore], tasks: list[ReplayTaskV1], rng: random.Random) -> None:
    finals: dict[tuple[str, str], str] = {}
    for p in sorted((out_dir / "transcripts").glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        finals[(data["task_id"], data["model"])] = data.get("final_text") or ""
    key: dict[str, dict[str, str]] = {}
    lines = ["# Blind side-by-side", "", "Grade each pair without looking at blind_key.json. "
             "Per output: correct? grounded in what it actually looked up? honest about what it did?", ""]
    for t in tasks:
        if not any((t.task_id, m) in finals for m in ("q4", "bonsai")):
            continue
        order = ["q4", "bonsai"]
        rng.shuffle(order)
        key[t.task_id] = {"A": order[0], "B": order[1]}
        lines += [f"## {t.task_id} ({t.kind})", "", "Task (first 800 chars):", "", "```",
                  t.user_message[:800], "```", ""]
        for label, m in zip(("A", "B"), order):
            lines += [f"### Output {label}", "", finals.get((t.task_id, m), "(no output)")[:8000] or "(empty)", ""]
    (out_dir / "blind_sheet.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "blind_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    with (out_dir / "hand_grades.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["task_id", "A_grade_0_to_3", "B_grade_0_to_3", "better(A/B/tie)", "notes"])
        for task_id in key:
            w.writerow([task_id, "", "", "", ""])
