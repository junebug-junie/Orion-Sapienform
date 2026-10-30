"""Eval: what Prime reads before a thought, built from live inputs captured 2026-10-30.

The fixture is the real body (durable-runs GET /physiology/state), attention (GET /temporal-self/frame)
and curiosity (Hub GET /curiosity/api/runs) at capture time. The eval builds the frame the runtime
would send to the model and checks it is something a mind could act on honestly:

* grounded  -- every arc_id in the frame is one Temporal Self returned, and the active arc is named;
* bounded   -- the frame stays under FRAME_CHAR_CAP (a few thousand tokens, design section 8.2);
* honest    -- a failed read shows "(unavailable: <why>)", never an empty section or "None";
* settles   -- the runtime's own settlement rule finds a graded run for a control question opened
               before the newest finished run in the fixture (the eval would catch a settle rule that
               can never close on live run shapes);
* refusals  -- the move check reads the live pending_focus shape (a focus waiting refuses the move);
* refs      -- Prime's live question of 2026-10-30, which wrote "arc 8d87adeca2ce06a5" in its text with
               no way for a curiosity run to open it, is refused; the same question with that arc in
               ``refs`` passes, and the run's focus section prints the dig command that opens it;
* record    -- the frame shows Prime's concern track record, counted the way state.track_record says
               (settled / dropped within an hour / faded), from records shaped like the live concerns.

Run: ``PYTHONPATH=. python orion/prime/evals/run_prime_frame_eval.py`` (exit 1 on any failed check).
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

from orion.curiosity.focus import FocusRow, format_focus_section
from orion.prime.acts import settle_nomination
from orion.prime.state import PrimeConcernV1, PrimeStateV1, close_concern
from orion.prime.thought import FrameInputs, PrimeAnswerV1, apply_answer, build_frame

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "prime_inputs_live_2026-10-30.json"
FRAME_CHAR_CAP = 12000


def _inputs(data: dict, **errors) -> FrameInputs:
    now = datetime.fromisoformat(data["captured_at"])
    temporal = None if "temporal_error" in errors else data["temporal"]
    arc_ids = [a["arc_id"] for a in (temporal or {}).get("arcs_today") or []]
    if temporal and (temporal.get("active_arc") or {}).get("arc_id"):
        arc_ids.append(temporal["active_arc"]["arc_id"])
    return FrameInputs(
        now=now,
        body=None if "body_error" in errors else data["body"], body_error=errors.get("body_error"),
        temporal=temporal, temporal_error=errors.get("temporal_error"),
        curiosity=None if "curiosity_error" in errors else data["curiosity"], curiosity_error=errors.get("curiosity_error"),
        arc_ids=arc_ids,
    )


def run() -> list[tuple[str, bool, str]]:
    data = json.loads(FIXTURE.read_text())
    state = PrimeStateV1(pending_changes=["first wake: I have not thought yet"])
    results: list[tuple[str, bool, str]] = []

    inputs = _inputs(data)
    frame = build_frame(state, inputs)
    shown = set(re.findall(r"arc_id (\w+)", frame))
    unknown = shown - set(inputs.arc_ids)
    active = (data["temporal"].get("active_arc") or {})
    results.append(("grounded", not unknown and (not active or active.get("arc_id") in frame),
                    f"{len(shown)} arc ids shown, unknown={sorted(unknown)}"))
    results.append(("bounded", len(frame) <= FRAME_CHAR_CAP, f"{len(frame)} chars (cap {FRAME_CHAR_CAP})"))
    results.append(("no_shells", "None" not in frame and "nan" not in frame.lower().split(),
                    "no None/nan in the frame"))

    gaps = build_frame(state, _inputs(data, body_error="URLError: refused", temporal_error="timeout",
                                       curiosity_error="HTTPError 502"))
    ok = all(f"(unavailable: {e})" in gaps for e in ("URLError: refused", "timeout", "HTTPError 502"))
    results.append(("honest_gaps", ok, "every failed read is a named gap"))

    runs = data["curiosity"]["runs"]
    finished = sorted((r for r in runs if r.get("outcome_kind") not in ("running", None) and r.get("started_at_iso")),
                      key=lambda r: r["started_at_iso"])
    if finished:
        newest = datetime.fromisoformat(finished[-1]["started_at_iso"])
        s = settle_nomination(arm="control", focus_id="eval", opened_at=newest - timedelta(seconds=1), queued=None,
                              queue_error=None, runs=runs, now=inputs.now)
        results.append(("settles", s is not None and s.run_id == finished[-1]["run_id"],
                        f"control settles on {getattr(s, 'run_id', None)}: {getattr(s, 'note', None)}"))
    else:
        results.append(("settles", False, "fixture has no finished investigate run"))

    move = PrimeAnswerV1.model_validate({"move": {"kind": "curiosity_focus", "question": "q", "why": "w"}})
    pending = (data["curiosity"]["schedule"].get("pending_focus") or {})
    focus_pending = bool(pending.get("focus")) if pending.get("available") else None
    applied = apply_answer(state, move, now=inputs.now, arc_ids=inputs.arc_ids, focus_pending=focus_pending)
    expect_pass = focus_pending is False
    results.append(("move_check_reads_live_shape", (applied.move is not None) == expect_pass,
                    f"pending_focus available={pending.get('available')} focus={bool(pending.get('focus'))} "
                    f"-> move {'sent' if applied.move else 'refused: ' + str(applied.move_refusal)}"))

    # refs: the live question (open nomination in orion:prime:state, 2026-10-30 16:05Z).
    live_q = ("What specific conditions or patterns in arc 8d87adeca2ce06a5 correlate with the high demand on the "
              "agent lane (39 requests for 1 slot), and does this congestion contribute to the observed energy overage?")
    live_why = "The agent lane congestion is an immediate operational risk."
    bare = apply_answer(PrimeStateV1(), PrimeAnswerV1.model_validate(
        {"move": {"kind": "curiosity_focus", "question": live_q, "why": live_why}}),
        now=inputs.now, arc_ids=inputs.arc_ids, focus_pending=False)
    cited = apply_answer(PrimeStateV1(), PrimeAnswerV1.model_validate(
        {"move": {"kind": "curiosity_focus", "question": live_q, "why": live_why,
                  "refs": [{"kind": "arc", "id": "8d87adeca2ce06a5"}]}}),
        now=inputs.now, arc_ids=inputs.arc_ids, focus_pending=False)
    section = "\n".join(format_focus_section(FocusRow(
        focus_id="f", question=live_q, why=live_why, origin="self", status="taken", chat_correlation_id="prime:t",
        refs=tuple(r.model_dump() for r in cited.refs))))
    results.append(("refs", bare.move is None and "8d87adeca2ce06a5" in (bare.move_refusal or "")
                    and cited.move is not None and "python -P -m orion.dig arc 8d87adeca2ce06a5" in section,
                    f"bare: {bare.move_refusal!r}; with refs: {'sent' if cited.move else cited.move_refusal}"))

    # record: three concerns shaped like the live ones, ended three ways.
    rec_state = PrimeStateV1()
    for cid, end, minutes in (("777925ef197d44b1", "settled", 600), ("e474869546ec4641", "dropped", 40),
                              ("a1b2c3d4e5f60718", "faded", 1500)):
        opened = inputs.now - timedelta(minutes=minutes)
        c = PrimeConcernV1(concern_id=cid, text="agent lane congestion", opened_at=opened, last_evidence_at=opened,
                           act_opened_at=opened)
        rec_state = close_concern(rec_state, c, end=end, now=inputs.now, note=end)
    rec_frame = build_frame(rec_state, inputs)
    want = "last 3 concerns: 1 settled"
    results.append(("record", want in rec_frame and "1 dropped (1 of them within an hour of opening)" in rec_frame
                    and "1 faded" in rec_frame and len(rec_frame) <= FRAME_CHAR_CAP,
                    "track record line shown and counted"))
    return results


def main() -> int:
    results = run()
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
