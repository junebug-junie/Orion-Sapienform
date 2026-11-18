"""Eval: what the curiosity replica reads before a plan, built from live inputs captured 2026-11-04.

The fixture is the real live-prior pool (FalkorDB ``orion_worldview``, Hub's own LIVE_PRIORS_CYPHER), 90 days
of scored tests (``curiosity_run_outcomes``), Hub's ``/curiosity/api/runs`` (3 days of investigate runs) with
their scores, Prime's concerns view built from the live ``orion:prime:state`` (the view Prime publishes from
this patch on), the live regulate hash on curiosity, and the newest chat-history digest's metadata. The
digest's words are NOT in the fixture (this repo is public; it summarises Juniper's chats): only its lengths,
index and times. The eval puts placeholder text of the same length in their place.

Checks (exit 1 on any failure):

* candidates -- at most 60, distinct, no stale prior, the 30 most uncertain all present, in an order that is
               not the uncertainty order (the replica must not be handed the rule it has to beat);
* request    -- the candidate list fits the contract (``CuriosityOfferPlanRequestV1`` validates);
* grounded   -- every candidate id and every stale id is printed in full in the frame;
* bounded    -- the frame plus system prompt stays under FRAME_TOKEN_CAP (~4 characters a token);
* honest     -- no ``None`` / ``nan`` in the frame; every failed read is a named gap;
* ages       -- the digest is shown with its window end and how long ago it was written; Prime's concerns
               with the view's age;
* plan       -- a plan of three candidates validates against the live pool, and ``with_head`` over the live
               rows puts exactly those three on top with the same layout (6 ordered + 2 coverage, same stale
               bucket) as the rule order;
* reject     -- the same plan with one stale id, or a made-up id, is rejected whole.

Run: ``PYTHONPATH=. python orion/curiosity/replica/evals/run_replica_frame_eval.py``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from orion.curiosity.replica.candidates import UNCERTAIN_N, build_candidates, last_moves
from orion.curiosity.replica.frame import ReplicaFrameInputs, build_frame, frame_tokens, system_prompt
from orion.curiosity.replica.plan import validate_plan
from orion.curiosity.value import build_yield_model, prior_tests_from_rows, valid_confidence
from orion.curiosity.worldview import WorldviewSnapshot, fresh_priors, select_priors, with_head
from orion.prime.state import PrimeConcernsViewV1
from orion.schemas.curiosity_replica import MAX_CANDIDATES, CuriosityOfferPickV1, CuriosityOfferPlanRequestV1

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "replica_inputs_live_2026-11-04.json"
FRAME_TOKEN_CAP = 6000
SAMPLE, STALE_AFTER, COVERAGE = 8, 3, 2  # Hub's live HUB_CURIOSITY_PRIOR_SAMPLE / _STALE_PRIOR_TESTS / _COVERAGE_SLOTS
RUN_ID = "eval0replica"


def run() -> list[tuple[str, bool, str]]:
    data = json.loads(FIXTURE.read_text())
    now = datetime.fromisoformat(data["captured_at"])
    rows = data["prior_rows"]
    results: list[tuple[str, bool, str]] = []

    rule, stale, _ = select_priors(rows, sample=SAMPLE, stale_after=STALE_AFTER, rotate_seed=RUN_ID,
                                   coverage_slots=COVERAGE)
    view = WorldviewSnapshot(live_priors=rule, stale_priors=stale, prior_rows=tuple(rows))
    fresh = fresh_priors(view, stale_after=STALE_AFTER)
    history = [rec for run in data["history_per_prior"] for rec in prior_tests_from_rows(run)]
    model = build_yield_model(history, window=3, pseudo_tests=2.0)
    cands = build_candidates(fresh, now=now, rotate_seed=RUN_ID,
                             expected_nats_for=lambda p: model.expected_nats(p.prior_id, valid_confidence(p.confidence)),
                             moves=last_moves(history))
    ids = [c.prior_id for c in cands]
    stale_ids = sorted({p.prior_id for p in fresh_priors(WorldviewSnapshot(prior_rows=tuple(rows)), stale_after=0)}
                       - {p.prior_id for p in fresh})
    top_unc = [p.prior_id for p in select_priors(rows, sample=UNCERTAIN_N, stale_after=STALE_AFTER,
                                                 rotate_seed=RUN_ID)[0]]
    results.append(("candidates", len(ids) <= MAX_CANDIDATES and len(set(ids)) == len(ids)
                    and not set(ids) & set(stale_ids) and set(top_unc) <= set(ids) and ids[:6] != top_unc[:6],
                    f"{len(ids)} of {len(fresh)} fresh ({len(rows)} live, {len(stale_ids)} stale); top 30 uncertain in"))

    try:
        req = CuriosityOfferPlanRequestV1(run_id=RUN_ID, correlation_id="c", deadline_at=now + timedelta(seconds=90),
                                          candidates=cands, stale_ids=stale_ids)
        results.append(("request", True, "the request validates"))
    except Exception as exc:  # noqa: BLE001
        results.append(("request", False, str(exc)[:200]))
        return results

    meta = data["digest_meta"] or {}
    digest = ({"summary": "s" * int(meta["summary_len"]), "title": "t" * int(meta.get("title_len") or 1),
               "window_end": meta["window_end"], "created_at": meta["created_at"], "index": meta["index"]}
              if meta else None)
    concerns = PrimeConcernsViewV1.model_validate(data["prime_concerns_view"]) if data["prime_concerns_view"] else None
    inp = ReplicaFrameInputs(now=now, request=req, runs=data["runs"],
                             outcomes={o["run_id"]: o for o in data["outcomes"]}, concerns=concerns,
                             concerns_error=None if concerns else "Prime has published no concerns view",
                             digest=digest, record_rows=[], fallback_counts={}, plans_total=0)
    frame = build_frame(inp)
    tokens = frame_tokens(system_prompt(), frame)
    results.append(("grounded", all(f"- {i} |" in frame for i in ids) and all(i in frame for i in stale_ids[:20]),
                    f"{len(ids)} candidate ids and {min(20, len(stale_ids))} stale ids printed in full"))
    results.append(("bounded", tokens <= FRAME_TOKEN_CAP, f"~{tokens} tokens (cap {FRAME_TOKEN_CAP})"))
    results.append(("honest", "None" not in frame and "nan" not in frame.lower().split(), "no None/nan"))
    gaps = build_frame(ReplicaFrameInputs(now=now, request=req, runs_error="URLError: refused", outcomes_error="pg",
                                          concerns_error="Prime has published no concerns view", signal_error="x",
                                          digest_error="pg", record_error="pg"))
    results.append(("honest_gaps", gaps.count("(unavailable:") >= 5 and "None" not in gaps, "every failed read named"))
    results.append(("ages", ("covering up to" in frame and "written" in frame and "h ago" in frame
                             and (concerns is None or "Prime's published view" in frame)),
                    f"digest {meta.get('index')} written {meta.get('created_at')}; concerns "
                    f"{len(concerns.concerns) if concerns else 'none'}"))

    picks = [CuriosityOfferPickV1(prior_id=i, why="a live eval pick") for i in ids[-3:]]
    fresh_ids = [p.prior_id for p in fresh]
    ok = validate_plan(picks, candidate_ids=ids, fresh_ids=fresh_ids) is None
    planned = with_head(view, [p.prior_id for p in picks], sample=SAMPLE, stale_after=STALE_AFTER, rotate_seed=RUN_ID,
                        coverage_slots=COVERAGE)
    got = [p.prior_id for p in planned.live_priors]
    layout = ([p.coverage for p in planned.live_priors] == [p.coverage for p in rule]
              and planned.stale_priors == view.stale_priors)
    results.append(("plan", ok and got[:3] == [p.prior_id for p in picks] and len(got) == len(rule) and layout,
                    f"top 3 = picks; {len(got)} shown on both; same coverage slots and stale bucket"))
    bad_stale = [picks[0], picks[1], CuriosityOfferPickV1(prior_id=stale_ids[0], why="stale")] if stale_ids else None
    bad_made_up = [picks[0], picks[1], CuriosityOfferPickV1(prior_id="self:not_a_real_prior", why="made up")]
    r1 = validate_plan(bad_stale, candidate_ids=ids, fresh_ids=fresh_ids) if bad_stale else "not_a_candidate:3"
    r2 = validate_plan(bad_made_up, candidate_ids=ids, fresh_ids=fresh_ids)
    results.append(("reject", r1 == "not_a_candidate:3" and r2 == "not_a_candidate:3", f"stale -> {r1}; made up -> {r2}"))
    return results


def main() -> int:
    results = run()
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
