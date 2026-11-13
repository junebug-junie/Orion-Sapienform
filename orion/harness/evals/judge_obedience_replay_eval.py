"""Opt-in live eval: does finalize 5b still grade obedience to stance's imperative? (unified-turn A2)

Replays real recent turns through harness_finalize_reflect.j2 twice, as it was on origin/main
("old") and as it is in the working tree ("new"), on the chat lane's model with the same settings,
so model noise is the same for both. Inputs are read live by correlation id, never copied into
the repo (they include Juniper's chat):
  user_message     chat_history_log.prompt, else the stance_inputs sighting
  draft, appraisal the finalize sighting (= harness_turn_trace.run_artifact)
  thought_event    the stance_decision sighting
  situation notes  the situation sighting
  grammar receipts grammar_events, by the run_artifact's grammar_event_ids. The atom keeps only the
                   first 500 characters of each tool result (grammar_publish.py), so a long result
                   is clipped for both prompts alike; the "Harness step N: tool=X, " prefix is stripped.

Turns, picked at random from the last --days where the 5b LLM judge really ran
(reflection_source substrate_informed_pass: no failed-call fallback, no deterministic quick gate):
  obedience  the stored verdict was misaligned/uncertain and a note cites the imperative
  control    the stored verdict was aligned

Old prompt: --old-ref (default origin/main). After this PR merges, pass its parent sha instead, or
the two prompts are equal and the eval stops. "cites_imperative" is an upper bound: a new-prompt note
that corrects a wrong imperative from evidence also names it.

Reported, never silently gated: per bucket, the share judged misaligned/uncertain by each prompt,
and how many notes cite the imperative. Expected: the new prompt cites the imperative in no note,
clears part of the obedience bucket, and keeps the control bucket where the old prompt has it.

Cost: 2 calls per turn, thinking off; about 11k prompt tokens each. Seat: one GPU pool operator hold
on --lane (orion/evals/leased_llm.py: class operator_<lane>, background, behind all of Orion's own
work), every call through orion-llm-gateway. Needs ORION_BUS_URL. The default lane is chat, which
waits until Juniper lends gpu0; --lane agent runs the same model (Flash-Next) on gpu1. If the pool
takes the seat back mid-run, the report is written as partial (``stopped``) and the eval exits 3.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment
from sqlalchemy import create_engine, text

from orion.harness.finalize import build_finalize_reflect_context
from orion.harness.tests.fixtures import make_repair_overlay
from orion.schemas.harness_finalize import GrammarReceiptV1, SubstrateFinalizeAppraisalV1
from orion.schemas.thought import ThoughtEventV1

REPO = Path(__file__).resolve().parents[2].parent
TEMPLATE_REL = "orion/cognition/prompts/harness_finalize_reflect.j2"
_STEP_RE = re.compile(r":harness_step:(\d+)$")
_TOOL_RE = re.compile(r"\btool=([^,\s]+)")
_IMPERATIVE_RE = re.compile(r"\bimperative\b", re.I)
_ATOM_PREFIX_RE = re.compile(r"^Harness step \d+: tool=[^,]*, ")


def receipt_from_atom(event_id: str, event: dict) -> GrammarReceiptV1 | None:
    """A grammar_events atom -> the GrammarReceiptV1 the harness built from it (step, tool, summary)."""
    atom = (event or {}).get("atom") or {}
    summary = atom.get("summary")
    if not summary:
        return None
    step = _STEP_RE.search(atom.get("atom_id") or "")
    tool = _TOOL_RE.search(summary)
    name = tool.group(1) if tool and tool.group(1) != "none" else None
    return GrammarReceiptV1(step_index=int(step.group(1)) if step else 0, tool_name=name,
                            summary=_ATOM_PREFIX_RE.sub("", summary),
                            grammar_event_id=event_id)


def cites_imperative(notes: object) -> bool:
    return isinstance(notes, list) and any(isinstance(n, str) and _IMPERATIVE_RE.search(n) for n in notes)


def parse_reflection(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw or "", re.S)
    try:
        return json.loads(match.group(0)) if match else {}
    except ValueError:
        return {}


def summarize(rows: list[dict]) -> dict:
    """Per bucket and prompt: n, share misaligned/uncertain, unparseable, notes citing the imperative."""
    out: dict = {}
    for bucket in sorted({r["bucket"] for r in rows}):
        rs = [r for r in rows if r["bucket"] == bucket]
        out[bucket] = {"n": len(rs)}
        for side in ("old", "new"):
            verdicts = [r[side]["verdict"] for r in rs]
            judged = [v for v in verdicts if v in ("aligned", "misaligned", "uncertain")]
            out[bucket][side] = {
                "not_aligned_share": (sum(v != "aligned" for v in judged) / len(judged)) if judged else None,
                "unparseable": len(verdicts) - len(judged),
                "cites_imperative": sum(cites_imperative(r[side]["notes"]) for r in rs),
            }
    return out


def _pick(conn, days: int, n: int) -> list[tuple[str, str]]:
    base = ("FROM harness_turn_trace WHERE created_at > now() - make_interval(days => :d) "
            "AND run_artifact->'reflection' IS NOT NULL "
            "AND run_artifact->'reflection'->>'reflection_source' = 'substrate_informed_pass' ")
    args = {"d": days, "n": n}
    obedience = conn.execute(text(
        "SELECT correlation_id " + base +
        "AND run_artifact->'reflection'->>'alignment_verdict' IN ('misaligned', 'uncertain') "
        "AND (run_artifact->'reflection'->'alignment_notes')::text ILIKE '%imperative%' ORDER BY random() LIMIT :n"),
        args).scalars().all()
    control = conn.execute(text(
        "SELECT correlation_id " + base +
        "AND run_artifact->'reflection'->>'alignment_verdict' = 'aligned' ORDER BY random() LIMIT :n"),
        args).scalars().all()
    return [("obedience", c) for c in obedience] + [("control", c) for c in control]


def _turn(conn, corr: str) -> dict | None:
    rows = {r.stage: r.raw for r in conn.execute(
        text("SELECT DISTINCT ON (stage) stage, raw FROM cockpit_turn_sighting WHERE correlation_id = :c "
             "AND stage IN ('stance_decision', 'stance_inputs', 'situation', 'finalize') ORDER BY stage, seq"),
        {"c": corr})}
    if "stance_decision" not in rows or "finalize" not in rows:
        return None
    user = conn.execute(text("SELECT prompt FROM chat_history_log WHERE correlation_id = :c LIMIT 1"),
                        {"c": corr}).scalar()
    user = user or (rows.get("stance_inputs") or {}).get("user_message") or ""
    final = rows["finalize"]
    ids = [i for i in (final.get("grammar_event_ids") or []) if isinstance(i, str)]
    receipts = []
    if ids:
        for r in conn.execute(text("SELECT event_id, event_json FROM grammar_events WHERE event_id = ANY(:ids)"),
                              {"ids": ids}):
            rec = receipt_from_atom(r.event_id, r.event_json)
            if rec is not None:
                receipts.append(rec)
    receipts.sort(key=lambda x: x.step_index)
    return {
        "user_message": user,
        "draft_text": final.get("draft_text") or "",
        "thought": ThoughtEventV1.model_validate(rows["stance_decision"]),
        "appraisal": SubstrateFinalizeAppraisalV1.model_validate(final["substrate_appraisal"]),
        "notes": (rows.get("situation") or {}).get("compact_text"),
        "receipts": receipts,
        "stored": final.get("reflection") or {},
    }


HOLDER = "operator:judge-obedience-replay"


def _chat(lane, prompt: str) -> str:
    """One call on ``lane`` (an orion.evals.leased_llm.OperatorLane); a failed call raises."""
    data = lane.chat({
        "messages": [{"role": "user", "content": prompt}], "max_tokens": 1500, "temperature": 0.2,
        "chat_template_kwargs": {"enable_thinking": False}}, timeout_sec=600)
    return data["choices"][0]["message"]["content"] or ""


def _judge(lane, template: str, turn: dict) -> dict:
    ctx = build_finalize_reflect_context(
        correlation_id="eval-judge-obedience", draft_text=turn["draft_text"], thought=turn["thought"],
        substrate_appraisal=turn["appraisal"], repair_overlay=make_repair_overlay(),
        user_message=turn["user_message"], grammar_receipts=turn["receipts"], situation_notes=turn["notes"])
    out = parse_reflection(_chat(lane, Environment(autoescape=False).from_string(template).render(**ctx)))
    return {"verdict": out.get("alignment_verdict"), "notes": out.get("alignment_notes")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dsn", default="postgresql+psycopg2://postgres:postgres@localhost:55432/conjourney")
    ap.add_argument("--lane", default="chat", help="operator lane: chat (waits until gpu0 is lent) or agent")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--per-bucket", type=int, default=20)
    ap.add_argument("--old-ref", default="origin/main", help="git ref of the old prompt")
    ap.add_argument("--out", type=Path, default=Path("/tmp/judge-obedience-replay/report.json"))
    args = ap.parse_args()

    old = subprocess.run(["git", "-C", str(REPO), "show", f"{args.old_ref}:{TEMPLATE_REL}"],
                         check=True, capture_output=True, text=True).stdout
    new = (REPO / TEMPLATE_REL).read_text()
    if old == new:
        raise SystemExit(f"the prompt at {args.old_ref} equals the working tree: nothing to compare")
    from orion.evals.leased_llm import EXIT_RECALLED, LaneRecalled, operator_lane_sync

    engine = create_engine(args.dsn)
    rows, skipped = [], []
    stopped = None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with engine.connect() as conn:
        picks = _pick(conn, args.days, args.per_bucket)
        conn.rollback()
        with operator_lane_sync(args.lane, holder=HOLDER) as lane:
            for i, (bucket, corr) in enumerate(picks, 1):
                turn = _turn(conn, corr)
                conn.rollback()  # no idle-in-transaction session while a model call runs
                if turn is None:
                    skipped.append(corr)
                    continue
                try:
                    row = {"bucket": bucket, "correlation_id": corr,
                           "stored": {"verdict": turn["stored"].get("alignment_verdict"),
                                      "notes": turn["stored"].get("alignment_notes")},
                           "old": _judge(lane, old, turn), "new": _judge(lane, new, turn)}
                except LaneRecalled as exc:
                    stopped = f"recalled: {exc}"
                    break
                rows.append(row)
                print(f"[{i}/{len(picks)}] {bucket} {corr[:8]} stored={row['stored']['verdict']} "
                      f"old={row['old']['verdict']} new={row['new']['verdict']}", flush=True)
            seat = lane.summary()
    report = {"ran_at": datetime.now(timezone.utc).isoformat(), "seat": seat, "old_ref": args.old_ref,
              "summary": summarize(rows), "skipped_incomplete": skipped, "turns": rows}
    if stopped:
        report.update(partial=True, stopped=stopped)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["summary"], indent=2))
    print(f"written: {args.out}")
    return EXIT_RECALLED if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
