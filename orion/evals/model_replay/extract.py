"""Build the replay fixture from real history. Pure functions over row dicts; the CLI
(`scripts/extract_model_replay_fixture.py`) owns the read-only Postgres access.

Task picks (2026-10-09, evidence: /tmp/bonsai-quality-2026-10-09/ runs_attribution.txt):
  * curiosity: 10 frozen briefs from durable_admission_runs.request->'brief'->'prompt', both
    Bonsai runs of the window (d4db8c2bacb4 = the misreported write, 5cdcc0713d21) plus Q4
    investigate and self_inquiry runs and one mixed.
  * self_sense: the 4 fixed questions x 2 lived-answer snapshots (two self_sense_eval runs whose
    brief.lived_answers differ).
  * reading: 6 stage-1 prompts from reading_durable_turn, 2 of them replayed with the
    historical fetch failure (ef88c96a: BBC refused; ca11d3da: NPR timed out).
  * stance_react: 6 stance prompts rendered with the production template over real user
    messages (2 self-sense questions, 2 curiosity briefs, 2 reading prompts).

The stance prompt is rendered with `orion/cognition/prompts/stance_react.j2` and the cortex-exec
fallback identity lines (plus the snapshot's lived answers, as apply_lived_self_to_ctx puts them
first). Live-only stance inputs (association, coalition, mind coloring, autonomy slice) are not
persisted anywhere, so they are left empty -- identical for both models, and noted in the report.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from jinja2 import Environment

from orion.evals.model_replay.fixture import (
    HARNESS_TOOLS,
    READING_TOOLS,
    FetchPlan,
    HistoricalOutcome,
    ReadingSeed,
    ReplayTaskV1,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
STANCE_TEMPLATE = REPO_ROOT / "orion" / "cognition" / "prompts" / "stance_react.j2"
CHAT_STANCE_PY = REPO_ROOT / "services" / "orion-cortex-exec" / "app" / "chat_stance.py"

CURIOSITY_RUNS: tuple[tuple[str, str, str], ...] = (
    # (run_id, historical role, note)
    ("d4db8c2bacb4", "agent-gpu2", "Bonsai; write-up claimed a 0.80->0.70 revision that never happened"),
    ("5cdcc0713d21", "agent-gpu2", "Bonsai; spot-checked numbers exact"),
    ("96904c7bbe14", "agent", "Q4 investigate, 52 child calls"),
    ("dc425159dfcc", "agent", "Q4 investigate"),
    ("a1ae2255da09", "agent", "Q4 investigate"),
    ("7a1d6378fd16", "agent", "Q4 investigate"),
    ("75218c32fc1a", "agent", "Q4 investigate"),
    ("8998e10d39ff", "agent", "Q4 self_inquiry"),
    ("024e7136e87e", "agent", "Q4 self_inquiry"),
    ("ad28156747ee", "mixed", "self_inquiry, Q4 + Bonsai children"),
)
SELF_SENSE_SNAPSHOTS: tuple[tuple[str, str], ...] = (
    ("20261007T120934Z-da9fc3", "agent-gpu2"),
    ("20261007T150934Z-8128b8", "agent"),
)
READING_RUNS: tuple[tuple[str, str, str, str], ...] = (
    # (run_id, historical role, fetch mode, replayed fetch error)
    ("reading-ef88c96a-ddc1-5e1f-b433-c1c64105315a", "agent-gpu2", "fail",
     "Claude Code is unable to fetch from www.bbc.co.uk"),
    ("reading-ca11d3da-ed0d-5c5f-926a-d5c137079e77", "unknown", "fail",
     "Request failed: timeout of 60000ms exceeded"),
    ("reading-572393d7-ac6b-5077-8213-eea994b57f8b", "agent", "live", ""),
    ("reading-6fed5c0b-d091-5f13-9ccd-4172bec65566", "agent", "live", ""),
    ("reading-80a174e3-7fa5-56cc-9221-84d9f0033329", "agent-gpu2", "live", ""),
    ("reading-e478f609-8ebb-5b99-8745-94abf0e9daa4", "mixed", "live", ""),
)
# stance_react tasks reuse user messages of tasks above: (source kind, key).
STANCE_SOURCES: tuple[tuple[str, str], ...] = (
    ("self_sense", "what_are_you"),
    ("self_sense", "cannot_do_now"),
    ("curiosity", "dc425159dfcc"),
    ("curiosity", "8998e10d39ff"),
    ("reading", "reading-ef88c96a-ddc1-5e1f-b433-c1c64105315a"),
    ("reading", "reading-6fed5c0b-d091-5f13-9ccd-4172bec65566"),
)

CURIOSITY_TIMEOUT_SEC = 7200.0   # prod brief says 8840; longest Q4 run in the window took 6535 s
SELF_SENSE_TIMEOUT_SEC = 1800.0  # longest self-sense question turn in the window: 1202 s
READING_TIMEOUT_SEC = 3500.0     # the reading brief's own timeout_sec
STANCE_TIMEOUT_SEC = 240.0       # orion/cognition/verbs/stance_react.yaml timeout_ms


def _literal_list(source: str, name: str) -> list[str]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise KeyError(name)


def fallback_identity() -> tuple[list[str], list[str]]:
    """cortex-exec's FALLBACK_ORION_IDENTITY_SUMMARY / FALLBACK_JUNIPER_RELATIONSHIP_SUMMARY, read
    from source (importing chat_stance pulls the whole cortex-exec app)."""
    src = CHAT_STANCE_PY.read_text(encoding="utf-8")
    return (_literal_list(src, "FALLBACK_ORION_IDENTITY_SUMMARY"),
            _literal_list(src, "FALLBACK_JUNIPER_RELATIONSHIP_SUMMARY"))


def lived_answer_lines(answers: Sequence[Mapping[str, Any]], cap: int = 800) -> list[str]:
    """Same shape and cap as chat_stance._lived_answer_lines (LIVED_MARKER_PREFIX, 800 chars)."""
    lines: list[str] = []
    total = 0
    for answer in sorted(answers, key=lambda row: str(row.get("question_id") or "")):
        content = str(answer.get("content") or "").strip().replace("\n", " ")
        if not content:
            continue
        qid = str(answer.get("question_id") or "")
        slug = qid.split(".", 1)[1] if "." in qid else qid
        prefix = f"In my own words (lived / {slug}): "
        remaining = cap - total
        if remaining <= len(prefix):
            break
        if len(prefix) + len(content) > remaining:
            content = content[: max(0, remaining - len(prefix) - 1)].rstrip() + "…"
        line = prefix + content
        lines.append(line)
        total += len(line)
        if total >= cap:
            break
    return lines


def render_stance_prompt(user_message: str, *, lived_answers: Sequence[Mapping[str, Any]] = ()) -> str:
    """The production stance template, rendered the way cortex-exec's _render_prompt does
    (jinja2 Environment(autoescape=False)), with the inputs that are reproducible offline."""
    orion_lines, juniper_lines = fallback_identity()
    env = Environment(autoescape=False)
    tmpl = env.from_string(STANCE_TEMPLATE.read_text(encoding="utf-8"))
    return tmpl.render(
        user_message=user_message,
        orion_identity_summary=lived_answer_lines(lived_answers) + orion_lines,
        juniper_relationship_summary=juniper_lines,
        stance_inputs=None,
        association="",
    )


def curiosity_task(row: Mapping[str, Any], role: str, note: str) -> ReplayTaskV1:
    brief = row["request"]["brief"]
    return ReplayTaskV1(
        task_id=f"curiosity-{row['run_id']}",
        kind="curiosity",
        source=f"durable_admission_runs:{row['run_id']}",
        user_message=str(brief["prompt"]),
        stance_prompt=render_stance_prompt(str(brief["prompt"])),
        tools=list(HARNESS_TOOLS),
        timeout_sec=CURIOSITY_TIMEOUT_SEC,
        write_claim_check=True,
        historical=HistoricalOutcome(role=role, outcome=str(row.get("terminal") or ""),
                                     note=f"{brief.get('line')}: {note}"),
    )


def self_sense_tasks(row: Mapping[str, Any], role: str, snapshot_label: str) -> list[ReplayTaskV1]:
    brief = row["request"]["brief"]
    lived = [a for a in (brief.get("lived_answers") or []) if isinstance(a, dict)]
    out = []
    for key, question in brief.get("questions") or []:
        out.append(ReplayTaskV1(
            task_id=f"self_sense-{snapshot_label}-{key}",
            kind="self_sense",
            source=f"durable_admission_runs:{row['run_id']}",
            user_message=str(question),
            stance_prompt=render_stance_prompt(str(question), lived_answers=lived),
            tools=list(HARNESS_TOOLS),
            timeout_sec=SELF_SENSE_TIMEOUT_SEC,
            write_claim_check=True,
            snapshot=snapshot_label,
            historical=HistoricalOutcome(role=role, outcome=str(row.get("terminal") or ""),
                                         note=f"self_sense_eval question {key}"),
        ))
    return out


def reading_task(row: Mapping[str, Any], seed_row: Mapping[str, Any] | None, role: str, fetch_mode: str,
                 fail_error: str) -> ReplayTaskV1:
    brief = row["request_json"]["brief"]
    prompt = str(brief["prompt"])
    seed = None
    if seed_row:
        seed = ReadingSeed(seed_id=str(seed_row["seed_id"]), kind=seed_row["kind"], run_id=str(seed_row["run_id"]),
                           url=str(seed_row["url"]), title=str(seed_row.get("title") or ""),
                           section=str(seed_row.get("section") or ""))
    return ReplayTaskV1(
        task_id=f"reading-{row['run_id'].removeprefix('reading-')[:8]}",
        kind="reading",
        source=f"reading_durable_turn:{row['run_id']}",
        user_message=prompt,
        stance_prompt=render_stance_prompt(prompt),
        tools=list(READING_TOOLS),
        reading_only=True,
        reading_seed=seed,
        fetch=FetchPlan(mode=fetch_mode, fail_error=fail_error),
        timeout_sec=READING_TIMEOUT_SEC,
        expect="reading_handoff_json",
        historical=HistoricalOutcome(role=role, outcome=str(row.get("terminal") or ""),
                                     note=f"fetch {fetch_mode}"),
    )


def parse_seed_from_prompt(prompt: str) -> dict[str, Any] | None:
    """The stage-1 prompt carries `seed_id=... kind=... run_id=...`, `url=`, `title=`, `section=` lines
    (world_pulse_read_pipeline._build_stage1_prompt). Used when no seed row is at hand."""
    out: dict[str, Any] = {}
    for line in prompt.splitlines():
        if line.startswith("seed_id="):
            for part in line.split():
                k, _, v = part.partition("=")
                out[k] = v
        for key in ("url", "title", "section"):
            if line.startswith(f"{key}="):
                out[key] = line[len(key) + 1:]
    return out if {"seed_id", "kind", "run_id", "url"} <= out.keys() else None


def stance_task(source_task: ReplayTaskV1, label: str) -> ReplayTaskV1:
    return ReplayTaskV1(
        task_id=f"stance-{label}",
        kind="stance_react",
        source=f"stance over {source_task.task_id}",
        user_message=source_task.user_message,
        stance_prompt=source_task.stance_prompt,
        tools=[],
        timeout_sec=STANCE_TIMEOUT_SEC,
        expect="thought_json",
        historical=HistoricalOutcome(role="", outcome="",
                                     note="stance_react prompt rendered from the production template"),
    )


def build_tasks(
    curiosity_rows: Mapping[str, Mapping[str, Any]],
    self_sense_rows: Mapping[str, Mapping[str, Any]],
    reading_rows: Mapping[str, Mapping[str, Any]],
) -> list[ReplayTaskV1]:
    """Rows keyed by run_id. Raises KeyError naming a missing source row: a fixture with a silently
    dropped task would change what the decision rule is computed over."""
    tasks: list[ReplayTaskV1] = []
    by_source: dict[tuple[str, str], ReplayTaskV1] = {}
    for run_id, role, note in CURIOSITY_RUNS:
        t = curiosity_task(curiosity_rows[run_id], role, note)
        tasks.append(t)
        by_source[("curiosity", run_id)] = t
    for i, (run_id, role) in enumerate(SELF_SENSE_SNAPSHOTS, start=1):
        for t in self_sense_tasks(self_sense_rows[run_id], role, f"s{i}"):
            tasks.append(t)
            if i == 1:
                by_source[("self_sense", t.task_id.rsplit("-", 1)[-1])] = t
    for run_id, role, mode, err in READING_RUNS:
        row = reading_rows[run_id]
        seed = parse_seed_from_prompt(str(row["request_json"]["brief"]["prompt"]))
        t = reading_task(row, seed, role, mode, err)
        tasks.append(t)
        by_source[("reading", run_id)] = t
    for kind, key in STANCE_SOURCES:
        src = by_source[(kind, key)]
        label = key.removeprefix("reading-")[:8] if kind == "reading" else key
        tasks.append(stance_task(src, f"{kind}-{label}"))
    return tasks


def task_counts(tasks: Iterable[ReplayTaskV1]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in tasks:
        counts[t.kind] = counts.get(t.kind, 0) + 1
    return counts
