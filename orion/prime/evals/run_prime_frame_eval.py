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
* refusals  -- apply_answer on the live pending_focus payload shape (a focus waiting refuses the move). Since
               #3366 the runtime reads that payload from GET /curiosity/api/focus/pending, not the dashboard;
               the shape is the same (both come from Hub's _read_pending_focus);
* refs      -- Prime's live question of 2026-10-30, which wrote "arc 8d87adeca2ce06a5" in its text with
               no way for a curiosity run to open it, is refused; the same question with that arc in
               ``refs`` passes, and the run's focus section prints the dig command that opens it;
* record    -- the frame shows Prime's concern track record, counted the way state.track_record says
               (settled / dropped within an hour / faded), from records shaped like the live concerns.
* regulate  -- with the receptors read from the real config/physiology/families.yaml, the frame lists only
               curiosity investigate and the dream cycle as signalable (step 4 added dream, slow only); a slow
               signal to each goes; the same signal to self-inquiry (no receptors) is refused with a reason Prime
               can read; quicken to dream is refused; the frame stays under the cap.
* attention_background -- (2026-11-03) a tick where only the active arc moved records no change (attention
               no longer wakes Prime), yet the frame still names the live active arc under ATTENTION.
* focus_read_split -- (#3366) with the dashboard read timed out, apply_answer still decides from the cheap
               pending-focus read's result (a move goes when nothing waits, is refused when a focus waits, and is
               refused with the runtime's "timed out" sentence when that read times out); the frame names both
               gaps. It does not run the runtime's read-to-result mapping: services/orion-mind/tests/test_prime_loop.py
               covers that.
* starved_try  -- after a try that never reached the model, the next frame says so in Prime's own outcome
               lines, and that line is not a change (it cannot wake Prime by itself).
* now          -- (integration patch 3) a full now (12 items, TemporalNowV1-valid, labels at the 200-char cap) adds
               the WHAT YOUR ORGANS LIVED section with every item's [experience:<id>], the frame stays under the
               cap, the dashboard's runs are not in the frame, and apply_answer keeps only shown cites, at most 3;
               and (#3425) the system prompt's JSON reply template itself carries the top-level experience refs.

Since integration patch 3 the frame does not read the curiosity dashboard; the fixture's dashboard copy still feeds
``settles`` (settling reads it) and its pending_focus page stands in for the cheap read (same shape).

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
from orion.prime.thought import FrameInputs, PrimeAnswerV1, apply_answer, build_frame, system_prompt

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
        pending_focus_read="pending_focus_error" not in errors,
        pending_focus=None if "pending_focus_error" in errors else (
            data["curiosity"]["schedule"].get("pending_focus") or {}).get("focus"),
        pending_focus_error=errors.get("pending_focus_error"),
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
                                       pending_focus_error="HTTPError 502"))
    ok = (all(f"(unavailable: {e})" in gaps for e in ("URLError: refused", "timeout"))
          and "(whether a focus is waiting for the next run is unknown: HTTPError 502)" in gaps)
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
    # regulate (step 3): receptors from the real families config, on the live frame.
    from orion.physiology.families import load_families

    fams, err = load_families(Path(__file__).resolve().parents[3] / "config" / "physiology" / "families.yaml")
    receptors = {f: c.receptors for f, c in (fams.families.items() if fams else []) if c.receptors}
    reg_inputs = _inputs(data)
    reg_inputs.receptors = receptors
    reg_frame = build_frame(PrimeStateV1(), reg_inputs)

    def regulate(family):
        return apply_answer(PrimeStateV1(), PrimeAnswerV1.model_validate(
            {"move": {"kind": "regulate", "family": family, "direction": "slow",
                      "why": "curiosity took the agent lane all morning", "hours": 3}}),
            now=inputs.now, arc_ids=inputs.arc_ids, focus_pending=False, receptors=receptors)

    ok_move, refused = regulate("durable.curiosity_investigate"), regulate("durable.self_inquiry")
    ok_dream = regulate("dream.cycle")
    dream_quicken = apply_answer(PrimeStateV1(), PrimeAnswerV1.model_validate(
        {"move": {"kind": "regulate", "family": "dream.cycle", "direction": "quicken", "why": "w", "hours": 3}}),
        now=inputs.now, arc_ids=inputs.arc_ids, focus_pending=False, receptors=receptors)
    results.append(("regulate", err is None and sorted(receptors) == ["dream.cycle", "durable.curiosity_investigate"]
                    and "- durable.curiosity_investigate: slow" in reg_frame and "- dream.cycle: slow" in reg_frame
                    and ok_move.move is not None and ok_dream.move is not None and refused.move is None
                    and dream_quicken.move is None and "not quicken" in (dream_quicken.move_refusal or "")
                    and "declares no receptors" in (refused.move_refusal or "") and len(reg_frame) <= FRAME_CHAR_CAP,
                    f"receptors {receptors}; self_inquiry: {refused.move_refusal!r}"))
    # attention_background (2026-11-03): attention is background, not a trigger.
    from orion.prime.state import PrimeSnapshotV1, note_no_model, observe

    quiet = PrimeStateV1(seen_binding=[], last_thought_at=inputs.now - timedelta(minutes=10))
    ticked = observe(quiet, PrimeSnapshotV1(binding=[], body_ok=True), inputs.now)
    bg_frame = build_frame(ticked, _inputs(data))
    active_id = active.get("arc_id")
    results.append(("attention_background", ticked.pending_changes == [] and "ATTENTION" in bg_frame
                    and (not active_id or active_id in bg_frame),
                    f"changes after an attention-only tick: {ticked.pending_changes}; active arc {active_id} in frame"))

    # focus_read_split (#3366): the dashboard timed out every thought on 11-14; the cheap read decides now.
    live_page = data["curiosity"]["schedule"]["pending_focus"]  # GET /curiosity/api/focus/pending has this shape
    waiting = {"origin": "juniper", "question": "what heats the closet at night?"}
    down: dict = {}  # the dashboard is not read for the frame any more (integration patch 3)
    split_ok, split_why = True, []
    for page_focus, unread, expect_sent in ((live_page["focus"], None, True), (waiting, None, False),
                                            (None, "Hub did not answer in 4 s", False)):
        inp = _inputs(data, **down)
        if unread:
            inp.pending_focus_error = unread
            fp, why = None, "timed out reading whether curiosity already has a focus waiting (no answer from Hub in 4 s)"
        else:
            inp.pending_focus_read, inp.pending_focus = True, page_focus
            fp, why = page_focus is not None, None
        f = build_frame(state, inp)
        a = apply_answer(state, move, now=inp.now, arc_ids=inp.arc_ids, focus_pending=fp, focus_unread=why)
        shown = ((not page_focus or "what heats the closet at night?" in f)
                 and (not unread or f"is unknown: {unread}" in f))
        ok = (a.move is not None) == expect_sent and shown and (not unread or "timed out" in (a.move_refusal or ""))
        split_ok &= ok
        split_why.append(f"{'waiting' if page_focus else 'unread' if unread else 'none'}->"
                         f"{'sent' if a.move else 'refused'}{'' if ok else ' (WRONG)'}")
    results.append(("focus_read_split", split_ok, "; ".join(split_why)))

    # starved_try: the line a no-seat try leaves (orion/mindloop/runtime.py _note_no_model) reaches the next frame.
    line = "a try to think at 12:00 UTC never reached the model (no_seat: deadline); I tried again"
    starved = note_no_model(PrimeStateV1(pending_changes=["quiet: nothing has woken me for 6 hours"]), "no_seat",
                            inputs.now, base=timedelta(seconds=300))
    starved = starved.model_copy(update={"recent_outcomes": [line]})
    s_frame = build_frame(starved, _inputs(data))
    results.append(("starved_try", line in s_frame and starved.pending_changes == ["quiet: nothing has woken me for 6 hours"]
                    and len(s_frame) <= FRAME_CHAR_CAP, "the no-seat line is shown, and is not a change"))
    results.append(_now_check(state, data))
    return results


def _now_check(state: PrimeStateV1, data: dict) -> tuple[str, bool, str]:
    """A full now in the frame (integration patch 3), built through the real contract (TemporalNowV1)."""
    from orion.schemas.temporal_self import NOW_LABEL_MAX, NOW_MAX_ITEMS, TemporalNowV1

    inputs = _inputs(data)
    kinds = ["reading_turn", "curiosity_run", "vision_percept", "prediction_surprise", "reverie_question", "street_event"]
    organs = ["reading", "curiosity", "camera", "prediction", "reverie", "street"]
    items = [{"event_id": f"{kinds[i % 6]}:cortex-{i:02d}-8580502e-e6ac-5b06-90c9-0f244b9d51ab", "organ": organs[i % 6],
              "source_kind": kinds[i % 6], "label": ("x" * NOW_LABEL_MAX), "related_refs": [f"act:act_{i:032x}"] * 8,
              "occurred_at": (inputs.now - timedelta(minutes=5 * i)).isoformat(), "salience": round(1 - i / 20, 2),
              "surprise_nats": 3.25 if i % 6 == 3 else None} for i in range(NOW_MAX_ITEMS)]
    now = TemporalNowV1.model_validate({"shown_id": "eval-shown", "reader_organ": "prime", "since": inputs.now.isoformat(),
                                        "as_of": inputs.now.isoformat(), "items": items, "candidates": 40,
                                        "dropped_by_organ_cap": 20, "dropped_by_total_cap": 8}).model_dump(mode="json")
    inputs.orion_now = now
    frame = build_frame(state, inputs)
    ids = [i["event_id"] for i in now["items"]]
    runs = [r.get("run_id") for r in data["curiosity"]["runs"] if r.get("run_id")]
    shown_all = all(f"[experience:{e}]" in frame for e in ids)
    answer = PrimeAnswerV1.model_validate({"reading": "r", "refs": [{"kind": "experience", "id": e}
                                                                    for e in ["not:shown", *ids[:5]]]})
    applied = apply_answer(state, answer, now=inputs.now, arc_ids=inputs.arc_ids, focus_pending=False,
                           shown_experience=ids)
    # #3425: the cite must be part of the reply template itself (live, a paragraph after it was not followed).
    prompt = system_prompt(None, events=True, take_on=True, experiences=True, needs=True, evidence_clock=True)
    start = prompt.index("Reply with ONE JSON object")
    template = prompt[start:prompt.index("\n\n", start)]
    in_template = ('"refs": [] | [{"kind": "experience"' in template
                   and template.index('"refs": [] |') < template.index('"move": null |')
                   and template.rstrip().endswith('"id": "..."}]}}'))
    ok = (shown_all and len(frame) <= FRAME_CHAR_CAP and "(28 more were recorded and not shown" in frame
          and not any(f"run {r}" in frame for r in runs) and applied.experience_cites == ids[:3] and in_template)
    return ("now", ok, f"{len(frame)} chars (cap {FRAME_CHAR_CAP}); 12 ids shown={shown_all}; "
                       f"cites kept {len(applied.experience_cites)} of 6 (1 not shown, cap 3); "
                       f"cite in reply template={in_template}")


def main() -> int:
    results = run()
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
