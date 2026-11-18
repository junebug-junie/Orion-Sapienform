# Self-monitor patch 1: Orion's track record, read-only

## Summary

- Orion can now answer "how good am I at what?" with evidence. A read-only report gathers every bet
  Orion made in the last 7 days. For each faculty it says reliable, unreliable, broken,
  miscalibrated, or "can't tell yet, and why".
- **One shape for every bet:** `SettledBetV1`. Pure adapters turn five sources into it:
  - the act ledger;
  - stance's turn self-bets;
  - body action bets;
  - temperature forecasts;
  - vision percept expectations.
- **A faculty reducer.** It weights recent bets more than old ones, so a faculty can recover. It
  tracks "didn't run" separately from "was wrong", and it checks Orion's stated confidence against
  what happened. A status only changes after it holds for 3 evaluations.
- **A reader contract for the next patches:** Redis key, stale or absent changes nothing, one
  direction only.
- **The grader moves** from the prediction service into `orion/prediction/grading.py`, so the
  service and the organ share one implementation.
- Nothing in Orion's behaviour changes. This patch publishes nothing and acts on nothing.

## Outcome moved

The first real self-knowledge table (live, 7 days to 2026-11-03 03:30Z, after the review fixes):

- **Broken:** the walkway-grade journal. 22 of its 28 runs never executed, and the 6 that ran all
  worked.
- **Miscalibrated:** stance's guesses about its own turn length. The step range covers the outcome
  6% of the time and the reply-length range 8%, both against a stated 80%.
- **Stance's "will I ask a question" guess** says 7% while it asks 11% of the time, so it is
  miscalibrated. It is also no better than always guessing the base rate.
- **Underconfident act predictions.** Prime's thoughts are predicted to work 72% of the time and
  work 100%. World-pulse reads are predicted at 71% and work 88%.
- **Painting works again:** 84 of 84 runs in 7 days. The town-episode journal works 397 of 397.
  Its stated confidence is a posterior shared with the other journal triggers, so the organ does
  not judge its calibration.
- **The office temperature forecast** covers 66% against a stated 80%. The cabinet forecast is
  reliable, at 84%.
- **Prime's concern acts** work 2% of the time. Orion predicts 12%, so they are rare successes, and
  Orion roughly knows it.
- **Eight faculties are named as ungradeable**, each with the fix that would grade it: reverie,
  dreams, curiosity, attention asks, same-person vision, GPU power, chat quality, and the power
  bill.

## Current architecture

Before this patch, each source graded itself in its own place (orion-prediction's live verdicts,
scripts, health pages), or not at all. Nothing gathered the grades, and Orion knew none of them.
Design: `docs/superpowers/specs/2026-10-20-self-monitor-organ-design.md` (PR #2827). Juniper took
every recommendation on 2026-11-03.

## Architecture touched

- New package `orion/self_monitor/`, which is pure apart from the report's read-only queries.
- New schemas in `orion/schemas/self_monitor.py`, registered in `orion/schemas/registry.py`.
- `orion/prediction/grading.py` (moved). orion-prediction now imports it from there; the image
  already copies `orion/`.

## Files changed

- `orion/schemas/self_monitor.py`: `SettledBetV1`, `FacultyRecordV1`, `SelfMonitorStateV1`.
- `orion/schemas/registry.py`: registers the three schemas.
- `orion/self_monitor/adapters.py`: one pure adapter per bet source, with the live vocabulary
  documented.
- `orion/self_monitor/record.py`: the faculty reducer (decay, Kish effective n, Wilson intervals,
  status rules, `advance` hysteresis).
- `orion/self_monitor/catalog.py`: the faculties that can't be graded yet, and why.
- `orion/self_monitor/view.py`: the reader contract for patches 2–5.
- `orion/self_monitor/report.py`: the read-only live report.
- `orion/self_monitor/tests/test_self_monitor_record.py`: 17 tests.
- `orion/prediction/__init__.py`, `orion/prediction/grading.py`: the shared grader (moved, not
  copied).
- `services/orion-prediction/app/{action_bets,engine,turn_self_bets}.py`,
  `services/orion-prediction/tests/test_grading.py`: import the grader from its new home.
- `.github/workflows/self-monitor-tests.yml`: new CI job.
- `.github/workflows/orion-prediction-tests.yml`: also runs when `orion/prediction/**` changes.

## Schema / bus / API changes

- Added: `SettledBetV1`, `FacultyRecordV1`, `SelfMonitorStateV1` (registry). The Redis key name
  `orion:self_monitor:latest` is reserved in `view.py`; nothing writes it yet.
- Removed: `services/orion-prediction/app/grading.py`, moved to `orion/prediction/grading.py`.
- Renamed: none.
- Behavior changed: none.
- Compatibility notes: none. There is no bus channel and nothing new is published.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest orion/self_monitor/tests                         -> 21 passed (also in a clean CI-like venv: pytest pydantic==2.10.3 numpy sqlalchemy)
cd services/orion-prediction && pytest tests            -> 217 passed, 8 failed; the same 8 fail on main (Prophet/env), so the move breaks nothing
pytest tests/test_agent_trace_schema_registry.py + registry tests -> 164 passed
```

## Evals run

```text
PYTHONPATH=. POSTGRES_URI=... python -P -m orion.self_monitor.report   (read-only, exit 0, no source errors)
  after review fixes: 1 broken, 1 unreliable, 8 miscalibrated, 13 reliable, rest unknown (too few, or catalogued ungradeable)
```

## Docker/build/smoke checks

```text
Not needed: no service runtime changes apart from orion-prediction's import path. Its image copies orion/ (Dockerfile).
```

## Metric quality gate (for the numbers the organ will publish in patch 2)

1. **Provenance.** Every bet traces to a source row via `evidence_ref`. Each adapter's docstring
   names the producing module.
2. **Independence.** Each faculty is reduced only from its own bets, and a bet counts once
   (`bet_id`). Act-ledger `predicted` is the act cell's posterior mean. So "act calibration"
   measures whether that posterior learns, not an independent judgement. This is stated in the
   report wording.
3. **Theory anchor.**
   - Wilson score intervals on Bernoulli shares, with Kish's effective n for weighted samples.
   - Brier score and skill, as orion-prediction already uses.
   - Calibration-in-the-large: mean stated confidence vs the observed rate.
4. **Live sanity.** Pulled above. Statuses vary across faculties, nothing is pinned, and each
   faculty can return to `reliable`. Old evidence decays out of the 7-day window, and a test pins
   recovery.
5. **Existing mechanism.** It reuses orion-prediction's grader (moved, one copy). The
   `self_signal_outcome_association` statistics answer a different question (body state → turn
   failure). Regulation's reader pattern is copied in `view.py`.
6. **Reversibility.** Pure package, no schema in any database, nothing published. Delete the
   package to remove it.

## Review findings fixed

The code review ran in a subagent. It found 3 material and 8 minor findings, and all are fixed.

- **Finding (material):** the faculty key left out the doer. Painting, dream hops and the
  journal-compose run all "make a durable_run", so one failing doer would hide behind the others.
  - **Fix:** the key is now `act.<actor>.<verb>[.<name>]`. The actor is also the act ledger's own
    learner key. Names may contain hyphens, but no digits, so ids and hashes are dropped.
  - **Evidence:** `test_act_faculty_is_keyed_on_the_doer_and_a_plain_name_only`. Live, the three
    doers are now separate faculties.
- **Finding (material):** the town-episode journal was called "miscalibrated" against a confidence
  Orion never stated about it. The act posterior is one per verb|actor|signal, shared by all 8
  journal triggers.
  - **Fix:** bets carry `belief_key`. Calibration and Brier are skipped when that key spans several
    faculties, and the reason says so. Reliability and broken still apply.
  - **Evidence:** `test_calibration_is_not_judged_against_a_belief_shared_with_siblings`.
- **Finding (material):** a flat 0.05 slack let "asks a question, stated 0.07, happened 0.11"
  pass as reliable. Event faculties also had no skill check.
  - **Fix:** the slack is now `min(0.05, 0.25 * min(p, 1 - p))`, and event faculties need Brier
    skill above 0 to be called reliable. The word "within" was removed.
  - **Evidence:** `test_small_probabilities_get_a_proportional_slack` and
    `test_an_event_faculty_without_skill_is_not_reliable`. Live, asks_question is now miscalibrated.
- **Findings (minor), fixed:**
  - "run died" and `no_planner_decisions` are now `failed_to_execute`.
  - An expired act is a real 0, as the ledger defines it: `refuted`, settled by the world.
  - The stated mean is decay-weighted, and calibration compares the same bets on both sides.
  - The report survives empty error messages and engine errors, disposes its engine in `finally`,
    and counts adapter failures once per source.
  - `advance()` refuses a different faculty and heals a missing `status_since`.
  - The production `assert` became a guard.
  - The adapter docstring now states the back-filled `predicted` case.
  - Stale doc references to `app/grading.py` are fixed.

## Restart required

```bash
scripts/safe_docker_build.sh orion-prediction up -d --build
```

## Risks / concerns

- Severity: low
- Concern: act faculty names come from `object_ref`. A new object kind whose name is a plain word
  makes its own faculty, and one with an id is folded into its kind. That is right for the live
  kinds; a future kind could split or merge wrongly.
- Mitigation: the rule is tested on the live kinds, and the report shows every faculty, so a bad
  split is visible.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3041

🤖 Generated with [Claude Code](https://claude.com/claude-code)
