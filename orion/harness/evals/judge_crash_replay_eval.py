"""Opt-in live eval: does the finalize 5b judge still fail to write a judgement? (judge-crash fix)

Replays the real turns whose 5b judge fell back to "reflect_llm_failed" (2026-11-03..11, listed by
correlation id in fixtures/judge_crash_cases.json) on the lane each one failed on, two ways:

  old  the request as cortex-exec sent it before this fix: the prompt at --old-ref as one system
       message, no response_format, the lane's own thinking default, max_tokens 8000.
  new  the request as it is now: the working-tree prompt as the system message, the grade turn
       (FINALIZE_REFLECT_USER_TURN) as the user message, the gateway's own response_format for the
       FinalizeReflectionV1 judgement schema (json_object_schema, rewritten for Strata exactly as the
       gateway does), thinking off.

Each answer goes through the harness's own parse: parse_json_object, the server-owned fields, then
FinalizeReflectionV1. An engine refusal (Strata returns HTTP 502 for an answer that breaks the schema)
counts as a failure. Turns that failed on transport (gpu_pool_unavailable, timeouts) are not replayed:
no prompt change can move them; they are counted in the report from the fixture.

A control set of turns the judge graded fine is replayed too, so a drop in failures is not bought
by a judge that now says something different: the report gives each side's agreement with the
stored verdict.

Turn inputs are read live (they include Juniper's chat), never stored: the draft, appraisal and
grammar event ids from harness_turn_trace.run_artifact; thought_event from the stance_decision
sighting; situation notes from the situation sighting; user_message from chat_history_log, else the
stance_inputs sighting. Recall is off for this verb now, so neither side gets a memory digest.

Cost: thinking-on old calls run minutes each on agent-gpu2. Bypasses the GPU pool like the other
replay evals: run it when the lanes are quiet. --sides new skips the old calls.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
from jinja2 import Environment
from sqlalchemy import create_engine, text

from orion.cognition.cortex_payload_extract import looks_like_error_text
from orion.core.llm_json import parse_json_object
from orion.harness.evals.judge_obedience_replay_eval import TEMPLATE_REL, receipt_from_atom
from orion.harness.finalize import (
    FINALIZE_REFLECT_HARNESS_FIELDS,
    FINALIZE_REFLECT_SCHEMA_NAME,
    FINALIZE_REFLECT_USER_TURN,
    build_finalize_reflect_context,
    finalize_reflect_json_schema,
)
from orion.harness.tests.fixtures import make_repair_overlay
from orion.schemas.harness_finalize import (
    FinalizeReflectionV1,
    GrammarReceiptV1,
    SubstrateFinalizeAppraisalV1,
)
from orion.schemas.thought import ThoughtEventV1

REPO = Path(__file__).resolve().parents[2].parent
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "judge_crash_cases.json"
# cortex-exec's _build_hop_messages appends this to every hop prompt.
HOP_GUARDRAIL = "If no memory is provided, do not claim prior work; say you don't know."
OLD_MAX_TOKENS = 8000
TEMPERATURE = 0.7  # cortex-exec's default when the context sets none
LANES = {  # lane -> (url, gateway backend name), config/gpu_pool.yaml roles
    "chat": ("http://100.112.254.99:8011", "strata"),
    "agent": ("http://100.112.254.99:8015", "strata"),
    "agent-gpu2": ("http://100.112.254.99:8016", "llamacpp"),
}


def _gateway_structured_output():
    """The gateway's own response_format builders (services/orion-llm-gateway/app/structured_output.py)."""
    path = REPO / "services/orion-llm-gateway/app/structured_output.py"
    spec = importlib.util.spec_from_file_location("gateway_structured_output", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def new_response_format(backend: str) -> dict:
    so = _gateway_structured_output()
    rf = so.build_response_format("json_object_schema", finalize_reflect_json_schema(),
                                  schema_name=FINALIZE_REFLECT_SCHEMA_NAME)
    return so.response_format_for_backend(rf, backend, schema_name=FINALIZE_REFLECT_SCHEMA_NAME)


def classify(raw: str | None, error: str | None = None) -> dict:
    """The harness's parse path (finalize.run_finalize_reflection) on one answer."""
    if error is not None:
        return {"ok": False, "failure": f"engine:{error[:120]}"}
    if raw and looks_like_error_text(raw):  # extract_finalize_reflection_payload refuses these
        return {"ok": False, "failure": "error_text"}
    try:
        payload = parse_json_object(raw or "")
    except Exception:
        head = (raw or "").lstrip()[:40]
        return {"ok": False, "failure": "unparseable:empty" if not head else "unparseable:text"}
    if not isinstance(payload, dict):
        return {"ok": False, "failure": "unparseable:not_object"}
    keys = sorted(payload)
    payload = {**payload, "correlation_id": "eval", "thought_event_id": "eval",
               "substrate_appraisal_id": "eval", "draft_hash": "eval", **FINALIZE_REFLECT_HARNESS_FIELDS}
    try:
        reflection = FinalizeReflectionV1.model_validate(payload)
    except Exception:
        return {"ok": False, "failure": "validation", "keys": keys[:8]}
    return {"ok": True, "verdict": reflection.alignment_verdict, "notes": reflection.alignment_notes[:8]}


def summarize(rows: list[dict], sides: tuple[str, ...]) -> dict:
    out: dict = {}
    for bucket in ("failed", "control", "perception"):
        rs = [r for r in rows if r["bucket"] == bucket]
        out[bucket] = {"n": len(rs)}
        for side in sides:
            res = [r[side] for r in rs if side in r]
            fails = [x for x in res if not x["ok"]]
            entry = {"replayed": len(res), "failures": len(fails),
                     "failure_classes": dict(Counter(x["failure"].split(":")[0] for x in fails))}
            if bucket == "control":
                agree = [r for r in rs if side in r and r[side]["ok"] and r[side]["verdict"] == r["stored_verdict"]]
                entry["agrees_with_stored_verdict"] = len(agree)
            if bucket == "perception":
                # The stored judge caught an unsupported perception claim; does this side still?
                caught = [r for r in rs if side in r and r[side]["ok"] and r[side]["verdict"] != "aligned"
                          and any("unsupported claim" in str(n).lower() for n in r[side].get("notes") or [])]
                entry["still_flags_unsupported_claim"] = len(caught)
                entry["not_aligned"] = sum(1 for r in rs if side in r and r[side]["ok"] and r[side]["verdict"] != "aligned")
            entry["by_lane"] = {lane: {"replayed": sum(1 for r in rs if side in r and r["lane"] == lane),
                                       "failures": sum(1 for r in rs if side in r and r["lane"] == lane
                                                       and not r[side]["ok"])}
                                for lane in sorted({r["lane"] for r in rs})}
            out[bucket][side] = entry
    return out


def _turn(conn, corr: str) -> dict | None:
    art = conn.execute(text("SELECT run_artifact FROM harness_turn_trace WHERE correlation_id = :c"),
                       {"c": corr}).scalar()
    rows = {r.stage: r.raw for r in conn.execute(
        text("SELECT DISTINCT ON (stage) stage, raw FROM cockpit_turn_sighting WHERE correlation_id = :c "
             "AND stage IN ('stance_decision', 'stance_inputs', 'situation') ORDER BY stage, seq"),
        {"c": corr})}
    if not art or not art.get("substrate_appraisal") or "stance_decision" not in rows:
        return None
    user = conn.execute(text("SELECT prompt FROM chat_history_log WHERE correlation_id = :c LIMIT 1"),
                        {"c": corr}).scalar()
    user = user or (rows.get("stance_inputs") or {}).get("user_message") or ""
    ids = [i for i in (art.get("grammar_event_ids") or []) if isinstance(i, str)]
    receipts: list[GrammarReceiptV1] = []
    if ids:
        for r in conn.execute(text("SELECT event_id, event_json FROM grammar_events WHERE event_id = ANY(:ids)"),
                              {"ids": ids}):
            rec = receipt_from_atom(r.event_id, r.event_json)
            if rec is not None:
                receipts.append(rec)
    receipts.sort(key=lambda x: x.step_index)
    # The recall digest the live call carried (cortex-exec appended it to the prompt);
    # the old side gets it back, the new side runs with recall off as the verb now does.
    digest = ""
    steps = conn.execute(text("SELECT steps FROM cognition_traces WHERE correlation_id = :c "
                              "AND verb = 'harness_finalize_reflect' ORDER BY timestamp DESC LIMIT 1"),
                         {"c": corr}).scalar()
    for step in steps or []:
        recall = ((step or {}).get("result") or {}).get("RecallService") or {}
        digest = str(recall.get("memory_digest") or digest)
    return {
        "user_message": user,
        "draft_text": art.get("draft_text") or "",
        "thought": ThoughtEventV1.model_validate(rows["stance_decision"]),
        "appraisal": SubstrateFinalizeAppraisalV1.model_validate(art["substrate_appraisal"]),
        "notes": (rows.get("situation") or {}).get("compact_text"),
        "receipts": receipts,
        "stored_verdict": (art.get("reflection") or {}).get("alignment_verdict"),
        "memory_digest": digest.strip(),
    }


def _context(turn: dict) -> dict:
    return build_finalize_reflect_context(
        correlation_id="eval-judge-crash", draft_text=turn["draft_text"], thought=turn["thought"],
        substrate_appraisal=turn["appraisal"], repair_overlay=make_repair_overlay(),
        user_message=turn["user_message"], grammar_receipts=turn["receipts"], situation_notes=turn["notes"])


def append_memory_digest(prompt: str, digest: str) -> str:
    """cortex-exec executor._append_memory_digest, as the live judge call got it before recall was off."""
    if not digest:
        return prompt
    return (f"{prompt}\n\n"
            "# RELEVANT MEMORY (retrieved; supplemental only — prioritize the latest user question; "
            "ignore unrelated older banter unless the user clearly continues that thread; "
            "do not repeat past assistant replies quoted inside this block)\n"
            f"{digest}\n")


def build_request(side: str, template: str, turn: dict, backend: str) -> dict:
    ctx = _context(turn)
    system = Environment(autoescape=False).from_string(template).render(**ctx)
    if side == "old":
        system = append_memory_digest(system, turn.get("memory_digest") or "")
    system = system.strip() + "\n\n" + HOP_GUARDRAIL
    body: dict = {"messages": [{"role": "system", "content": system}], "max_tokens": OLD_MAX_TOKENS,
                  "temperature": TEMPERATURE, "stream": False}
    if side == "new":
        body["messages"] += ctx["messages"]
        assert ctx["messages"][-1]["content"] == FINALIZE_REFLECT_USER_TURN
        body["response_format"] = new_response_format(backend)
        body["chat_template_kwargs"] = dict(ctx["chat_template_kwargs"])
    return body


def _call(url: str, body: dict) -> dict:
    try:
        resp = httpx.post(f"{url}/v1/chat/completions", timeout=900, json=body)
    except httpx.HTTPError as exc:
        return classify(None, error=f"transport {type(exc).__name__}")
    if resp.status_code != 200:
        return classify(None, error=f"http {resp.status_code} {resp.text[:80]}")
    choice = resp.json()["choices"][0]
    out = classify(choice["message"].get("content") or "")
    out["finish_reason"] = choice.get("finish_reason")
    out["completion_tokens"] = (resp.json().get("usage") or {}).get("completion_tokens")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dsn", default="postgresql+psycopg2://postgres:postgres@localhost:55432/conjourney")
    ap.add_argument("--old-ref", default="origin/main", help="git ref of the old prompt")
    ap.add_argument("--sides", default="old,new")
    ap.add_argument("--buckets", default="failed,control,perception")
    ap.add_argument("--lanes", default="chat,agent,agent-gpu2")
    ap.add_argument("--max-per-lane", type=int, default=0, help="per bucket and lane; 0 = all")
    ap.add_argument("--resume", type=Path, help="a partial report: its turns are kept, not re-run")
    ap.add_argument("--out", type=Path, default=Path("/tmp/judge-crash-replay/report.json"))
    args = ap.parse_args()
    sides = tuple(s for s in args.sides.split(",") if s)
    buckets, lanes = set(args.buckets.split(",")), set(args.lanes.split(","))

    templates = {"new": (REPO / TEMPLATE_REL).read_text()}
    if "old" in sides:
        templates["old"] = subprocess.run(["git", "-C", str(REPO), "show", f"{args.old_ref}:{TEMPLATE_REL}"],
                                          check=True, capture_output=True, text=True).stdout
    fixture = json.loads(FIXTURE.read_text())
    cases = [("failed", c) for c in fixture["failed"] if c["family"] != "transport"]
    cases += [("control", c) for c in fixture["control"]]
    cases += [("perception", c) for c in fixture.get("perception", [])]
    cases = [(b, c) for b, c in cases if b in buckets and c["lane"] in lanes]
    if args.max_per_lane:
        seen: Counter = Counter()
        kept = []
        for b, c in cases:
            seen[(b, c["lane"])] += 1
            if seen[(b, c["lane"])] <= args.max_per_lane:
                kept.append((b, c))
        cases = kept
    transport = Counter(c["failure_class"].split(":")[0] for c in fixture["failed"] if c["family"] == "transport")

    rows, skipped = [], []
    if args.resume and args.resume.exists():
        rows = [r for r in json.loads(args.resume.read_text()).get("turns", []) if all(s in r for s in sides)]
    done = {(r["bucket"], r["correlation_id"]) for r in rows}
    cases = [(b, c) for b, c in cases if (b, c["correlation_id"]) not in done]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(args.dsn)
    with engine.connect() as conn:
        for i, (bucket, case) in enumerate(cases, 1):
            turn = _turn(conn, case["correlation_id"])
            if turn is None:
                skipped.append(case["correlation_id"])
                continue
            url, backend = LANES[case["lane"]]
            row = {"bucket": bucket, "correlation_id": case["correlation_id"], "lane": case["lane"],
                   "live_failure": case.get("failure_class"), "stored_verdict": turn["stored_verdict"]}
            for side in sides:
                row[side] = _call(url, build_request(side, templates[side], turn, backend))
            rows.append(row)
            print(f"[{i}/{len(cases)}] {bucket} {case['lane']} {case['correlation_id'][:8]} "
                  f"live={case.get('failure_class') or row['stored_verdict']} "
                  + " ".join(f"{s}={'ok:' + row[s]['verdict'] if row[s]['ok'] else row[s]['failure']}" for s in sides),
                  flush=True)
            args.out.write_text(json.dumps({"partial": True, "turns": rows}, indent=2, default=str))
    report = {"ran_at": datetime.now(timezone.utc).isoformat(), "old_ref": args.old_ref, "sides": sides,
              "transport_failures_not_replayed": dict(transport), "summary": summarize(rows, sides),
              "skipped_incomplete": skipped, "turns": rows}
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["summary"], indent=2))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
