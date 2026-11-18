"""Eval: the live promise concern d8db9d7b1bba4cf7 replayed through the take-on decision frame (choice-as-EFE Step -1).

The concern is real: on 2026-11-12 Orion told Juniper "While you're off in Hyrule I'll take a lazy browse through the
last 24 hours" and Prime has held it since, with no way to carry it out. This replays it through the real code path
of the new primitive, with no model call and no write:

* frame        -- with take-on on, Prime's frame marks the concern "you can take this on" and the system prompt offers
                  the move; with it off, neither appears;
* decision     -- a ``take_on`` answer for it parses and passes the runtime's check while its horizon has room;
* adapter      -- the promise adapter returns the EXACT quote, the originating chat session and horizon = promise time
                  + 72 h (acceptance S-1.2), and the route back into that thread;
* request      -- Hub's carry request validates (``DurableRunRequestV1``, workflow ``intention.carry``), its prompt
                  starts with ``prime_concern:<id>`` (S-1.5) and its admission deadline is the horizon;
* no_adapter   -- the letter-thread concern Prime also holds is refused with a reason (S-1.1: no adapter yet);
* ages_out     -- held past the horizon and never taken on, the concern leaves Prime's list as ``lapsed``.

Default input: the captured fixture (deterministic, used by the test). ``--live`` reads the same rows from Postgres
(read-only) and the concern from Redis instead.

Run: ``PYTHONPATH=. python -m orion.prime.evals.run_intention_take_on_replay [--live]``. Exit 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from orion.prime import intention as it
from orion.prime.intention_sources import turn_promise
from orion.prime.state import PrimeConcernV1, PrimeStateV1
from orion.prime.thought import (
    FrameInputs, PrimeAnswerV1, TakeOnMoveV1, apply_answer, build_frame, summary_note, system_prompt,
)
from orion.schemas.intention import IntentionOrderV1, intention_correlation_id, intention_run_id

FIXTURE = Path(__file__).parent / "fixtures" / "intention_concern_d8db9d7b1bba4cf7.json"
DEFAULT_PSQL = "docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -Atq"
CONCERN_ID = "d8db9d7b1bba4cf7"


def _ts(value: Any) -> Optional[datetime]:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def load_fixture(path: Path = FIXTURE) -> dict[str, Any]:
    fx = json.loads(path.read_text())
    fx["row"]["promised_at"] = _ts(fx["row"]["promised_at"])
    return fx


def load_live(psql: str, redis_url: str) -> dict[str, Any]:
    fx = load_fixture()
    raw = subprocess.run(["redis-cli", "-u", redis_url, "GET", "orion:prime:state"], capture_output=True, text=True,
                         check=True).stdout
    concerns = json.loads(raw or "{}").get("concerns") or []
    live = next((c for c in concerns if c.get("concern_id") == CONCERN_ID), None)
    if live is not None:
        fx["concern"] = live
    corr = turn_promise.correlation_id(fx["concern"]["event_ref"])
    sql = ("select json_build_object('promised_at', t.created_at, 'promises_made', "
           "t.run_artifact->'reflection'->'promises_made', 'session_id', c.session_id, 'prompt', c.prompt, "
           "'response', c.response) from harness_turn_trace t left join chat_history_log c "
           f"on c.correlation_id = t.correlation_id where t.correlation_id = '{corr}' limit 1")
    out = subprocess.run([*shlex.split(psql), "-c", sql], capture_output=True, text=True, check=True).stdout.strip()
    fx["row"] = json.loads(out) if out else None
    if fx["row"] is not None:
        fx["row"]["promised_at"] = _ts(fx["row"]["promised_at"])
    return fx


def run(fx: dict[str, Any]) -> dict[str, Any]:
    concern = PrimeConcernV1.model_validate(fx["concern"])
    letter = PrimeConcernV1.model_validate(fx["letter_concern"])
    expect = fx["expect"]
    state = PrimeStateV1(concerns=[concern, letter])
    intentions = it.PrimeIntentionsStateV1()
    now = concern.opened_at + timedelta(hours=26)       # a day after the promise: room left before the horizon
    checks: dict[str, bool] = {}
    notes: dict[str, Any] = {}

    on = build_frame(state, FrameInputs(now=now, take_on=True, intentions=it.frame_lines(intentions, now)))
    off = build_frame(state, FrameInputs(now=now))
    concern_line = next(line for line in on.splitlines() if CONCERN_ID in line)
    checks["frame"] = ("you can take this on" in concern_line and "you can take this on" not in off
                       and '"kind": "take_on"' in system_prompt(None, take_on=True)
                       and "take_on" not in system_prompt(None)
                       and sum("you can take this on" in line for line in on.splitlines()) == 1)

    def check(move: TakeOnMoveV1, st: PrimeStateV1) -> Optional[str]:
        refusal = it.take_on_refusal(move.concern_id, st, intentions, now, enabled=True)
        if refusal:
            return refusal
        c = next(c for c in st.concerns if c.concern_id == move.concern_id)
        return it.parse_not_before(move.not_before, now, it.horizon_for(c))[1]

    answer = PrimeAnswerV1.model_validate({
        "reading": "I promised Juniper a browse of the last day's PRs and have not done it.",
        "concerns": {"keep": [CONCERN_ID]},
        "move": {"kind": "take_on", "concern_id": CONCERN_ID, "why": "I said I would, and it is still in reach"}})
    applied = apply_answer(state, answer, now=now, arc_ids=[], focus_pending=False, take_on_check=check)
    checks["decision"] = isinstance(applied.move, TakeOnMoveV1) and applied.move_refusal is None
    notes["summary_note"] = summary_note(applied)

    src = turn_promise.adapt(concern.event_ref, fx["row"], why=answer.move.why)
    if isinstance(src, str):
        checks["adapter"] = False
        notes["adapter_refusal"] = src
    else:
        checks["adapter"] = (f'"{expect["quote"]}"' in src.prompt and src.route == "chat_thread"
                             and src.route_target == expect["session_id"]
                             and src.horizon_at == _ts(expect["horizon_at"]) and src.quote_ref == expect["quote_ref"])
        notes.update(horizon_at=src.horizon_at.isoformat(), route_target=src.route_target, prompt_chars=len(src.prompt))
        # Hub's request, exactly as scripts/intention_order.py builds it (imported lazily: Hub's package path).
        from orion.schemas.durable_run import DurableRunRequestV1
        from orion.schemas.intention import IntentionCarryBriefV1
        from orion.schemas.resource_admission import ResourceRequirementV1

        order = IntentionOrderV1(concern_id=CONCERN_ID, event_ref=concern.event_ref, run_id=intention_run_id(CONCERN_ID),
                                 correlation_id=intention_correlation_id(CONCERN_ID), prime_act_id="act_eval",
                                 why=answer.move.why)
        req = DurableRunRequestV1(
            run_id=order.run_id, workflow="intention.carry", correlation_id=order.correlation_id,
            brief=IntentionCarryBriefV1(intention="intention.carry", concern_id=CONCERN_ID, source=src.source,
                                        prime_act_id=order.prime_act_id,
                                        prompt=it.carry_prompt(CONCERN_ID, src.prompt), route=src.route,
                                        route_target=src.route_target, quote_ref=src.quote_ref,
                                        horizon_at=src.horizon_at, timeout_sec=order.timeout_sec),
            admission=ResourceRequirementV1(preferred_lane="agent", resource="llm.route.agent",
                                            deadline_at=src.horizon_at))
        checks["request"] = (req.brief.prompt.startswith(f"prime_concern:{CONCERN_ID}\n")
                             and req.admission.deadline_at == src.horizon_at)

    refused = apply_answer(state, PrimeAnswerV1.model_validate({"move": {
        "kind": "take_on", "concern_id": letter.concern_id, "why": "fix the hire channel"}}),
        now=now, arc_ids=[], focus_pending=False, take_on_check=check)
    checks["no_adapter"] = refused.move is None and "only promises" in (refused.move_refusal or "")
    notes["no_adapter_refusal"] = refused.move_refusal

    later = it.horizon_for(concern) + timedelta(minutes=1)
    aged = it.age_out(state, intentions, later)
    checks["ages_out"] = (CONCERN_ID not in {c.concern_id for c in aged.concerns}
                          and letter.concern_id in {c.concern_id for c in aged.concerns}
                          and aged.concern_record[-1].end == "lapsed")
    return {"checks": checks, "notes": notes, "ok": all(checks.values())}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true", help="read the concern and rows live (read-only)")
    ap.add_argument("--psql", default=DEFAULT_PSQL)
    ap.add_argument("--redis", default="redis://100.92.216.81:6379/0")
    args = ap.parse_args(argv)
    fx = load_live(args.psql, args.redis) if args.live else load_fixture()
    out = run(fx)
    print(json.dumps(out, indent=1, default=str))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["CONCERN_ID", "FIXTURE", "load_fixture", "main", "run"]
