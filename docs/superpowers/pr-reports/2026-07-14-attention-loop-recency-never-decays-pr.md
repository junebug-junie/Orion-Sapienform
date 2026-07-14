## Summary

- Fixes a real, live, confirmed production bug: `orion/substrate/attention/salience.py`'s `_recency()` implements a genuine half-life decay (~6h) but was permanently returning `1.0` for every open loop, forever, because the one piece of state it needs (`SalienceHistory.first_seen_at`) was never populated by its only real caller.
- **Found via a hand-verification of axis-4** (self-model fidelity — see `docs/superpowers/specs/2026-07-14-inner-state-signal-framework-working-doc.md`), not a code audit: a live reverie thought confidently narrated "the coalition is fixated on unresolved transport node anomalies... a struggle to stabilize... requires sustained attention" about `open-loop-9d84d08cddf5`. Checking that claim against real Postgres data found the loop had already been verdicted `resolved` (2026-07-08) and `dismissed` (2026-07-10) in `attention_loop_outcome`, yet was still winning selection on 2026-07-14 with `salience_features` byte-identical, field for field, to the 2026-07-10 snapshot.
- Two fixes: (1) a new `_first_selected_at` dict in `attention_broadcast.py`, threaded into `_current_history()`, same lifecycle as the existing `_recent_selected_counts` (paired eviction); (2) `build_open_loops()` never received a `now` parameter at all — always defaulted to real wall-clock time internally, decoupled from the frame's own `generated_at`, which also made recency impossible to test deterministically. Now threaded through from `build_substrate_attention_frame`'s single resolved `now`.
- 6 new/updated tests. Code review (2 angles) found 2 real test-hygiene gaps — both fixed.

## Outcome moved

`_recency()`'s existing, correctly-implemented decay math now actually runs. A loop that's been stale for days will correctly stop being artificially treated as "maximally fresh" in the salience competition.

## Current architecture

`orion/substrate/attention_broadcast.py` runs the "rung 3" continuous workspace competition over the substrate graph (`build_substrate_attention_frame` → `build_open_loops` → `select_actions`, then `broadcast_projection_from_frame` records the winner). `orion/substrate/attention/salience.py`'s `compute_features()` computes 7 features per loop (`evidence_strength`, `evidence_breadth`, `recurrence`, `recency`, `novelty_vs_known`, `dwell`, `habituation`), combined via a linear weighted sum (`LinearSalienceCombiner`, seed weights: `evidence_strength=0.30`, `novelty_vs_known=0.20`, `habituation=-0.35`, others smaller). Before this patch: `recency` was structurally dead — `_current_history()` (the only producer of `SalienceHistory` for this path) never set `first_seen_at`, so `_recency()`'s `first is None` branch fired every time, returning `1.0` unconditionally.

## Architecture touched

`orion/substrate/attention_broadcast.py`, `orion/substrate/attention/scoring.py` (both live production modules, `orion-substrate-runtime`), plus 3 test files. No schema change (`SalienceHistory.first_seen_at` already existed — added this session, unused until now). No other service, no `.env`/compose changes.

## Files changed

- `orion/substrate/attention_broadcast.py`: `_first_selected_at: dict[str, datetime]` (new module state, mirrors `_recent_selected_counts`'s exact lifecycle including eviction), `_record_selection()` now accepts `now`, `_current_history()` now populates `first_seen_at`. `build_substrate_attention_frame` resolves `now` once (`resolved_now`) and reuses it for both `build_open_loops(now=resolved_now)` and `AttentionFrameV1.generated_at`.
- `orion/substrate/attention/scoring.py`: `build_open_loops()` gained `now: datetime | None = None`, threaded into `compute_salience(..., now=now)`.
- `orion/substrate/tests/test_attention_broadcast_recency.py` (new): 4 tests — first selection stays fresh (unchanged behavior), recency decays after real simulated time passes (the core regression test), recency matches the live incident's actual scale (near-zero after 5 simulated days), eviction stays paired between the two dicts.
- `orion/substrate/tests/test_scoring_salience_wiring.py`: 1 new test — direct, wall-clock-independent proof that `now` reaches the recency calculation (constructs `SalienceHistory` + two fixed, widely-separated `now` values, asserts the exact half-life relationship, no dependency on real execution time).
- `orion/substrate/tests/test_broadcast_habituation.py`, `orion/substrate/tests/test_attention_broadcast_dwell.py`: fixture fixes (see Review findings below).

## Schema / bus / API changes

None. `SalienceHistory.first_seen_at` and `build_open_loops`'s new `now` parameter are both purely additive with safe defaults (`{}` / `None`) — no existing caller's behavior changes unless it opts in.

## Env/config changes

None.

## Tests run

```text
.venv/bin/python -m pytest \
  orion/substrate/tests/test_attention_broadcast_dwell.py \
  orion/substrate/tests/test_attention_broadcast.py \
  orion/substrate/tests/test_scoring_salience_wiring.py \
  orion/substrate/tests/test_salience_combiner.py \
  orion/substrate/tests/test_salience_schema.py \
  orion/substrate/tests/test_attention_salience_contracts.py \
  orion/substrate/tests/test_salience_discrimination_eval.py \
  orion/substrate/tests/test_refit_salience_stub.py \
  orion/substrate/tests/test_attention_broadcast_recency.py \
  orion/substrate/tests/test_broadcast_habituation.py \
  tests/test_attention_frame_schemas.py \
  tests/test_attention_frame_builder.py -q
=> 63 passed

PYTHONPATH=services/orion-cortex-exec:. .venv/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_attention_frame.py -q
=> 9 passed (the chat-turn-scoped sibling path, confirmed unaffected)
```

## Evals run

None applicable — no eval harness exists for `orion/substrate/attention/`. The live diagnostic query used to find and confirm this bug (comparing a live reverie thought's claim against `attention_loop_outcome`/`substrate_attention_frames`/`substrate_attention_broadcast_projection` in Postgres) is the closest thing to an eval here — informal, by hand, not automated. Worth noting as a real gap: this class of bug (a dead feature silently pinned at its default) would benefit from a periodic live check, not just this one-off catch. Not building that here — out of proportionate scope for this fix.

## Docker/build/smoke checks

Not run this session. This is a code-only fix inside `orion-substrate-runtime`'s existing Python path — picked up on next rebuild/restart, no config/dependency changes to validate separately.

## Review findings fixed

2-angle code review run against the diff (correctness/wiring, cross-file/conventions):

- Finding (correctness angle): `orion/substrate/tests/test_broadcast_habituation.py` (a pre-existing test file, not touched by the original patch) has an autouse fixture that resets `_recent_selected_counts` but never the new `_first_selected_at` — this file runs WITH habituation enabled (`monkeypatch.setenv(...HABITUATION_ENABLED, "true")`), so `_current_history()` is genuinely live there and the new dict gets populated by real test execution, creating a real state-leak risk between this file's own tests.
  - Fix: added `_first_selected_at.clear()` to both setup and teardown.
  - Evidence: `orion/substrate/tests/test_broadcast_habituation.py`'s fixture, both tests still pass (`test_broadcast_history_tracks_selection`, `test_rumination_lock_breaks_on_produced_path`).
- Finding (cross-file angle): `orion/substrate/tests/test_attention_broadcast_dwell.py`'s fixture also never reset `_recent_selected_counts`/`_first_selected_at` — currently harmless (habituation is off in that file's tests, so `_current_history()` is never consulted), but a real order-dependent flake risk if that ever changes or if test execution is ever randomized/parallelized.
  - Fix: added both to setup and teardown, for consistency and defensiveness.
- Finding (correctness angle): the `now`-threading fix in `scoring.py` only had incidental test coverage (detection depended on real-clock proximity to a hardcoded test constant, not a direct assertion on `now` reaching the calculation).
  - Fix: added `test_build_open_loops_now_param_reaches_recency_calculation` — fully deterministic, zero dependency on real execution time, asserts the exact half-life relationship between two fixed, widely-separated `now` values.
- Noted, not fixed (confirmed out of scope, not a regression): `orion/substrate/attention_frame.py`'s `build_attention_frame` (the chat-turn-scoped sibling to the fixed broadcast path) never passes `history=`/`now=` to `build_open_loops` at all — it has no persistent habituation state of its own, so recency is always `1.0` there too, by the same mechanical path. Confirmed this is `SalienceHistory`'s own documented intent ("Phase 1 shadow behavior... the broadcast producer fills it in Phase 3") — this patch is exactly that Phase-3 broadcast-producer wiring, scoped to the path that has real persistent state. Not a regression introduced or left behind by this patch.
- Noted, not resolved (real open question, flagged for awareness): `test_broadcast_habituation.py::test_rumination_lock_breaks_on_produced_path` proves the resonance/inhibition-of-return safety valve *can* work in a focused 15-tick unit test (a stuck loop does eventually lose to a competitor). Why that safety valve didn't save the live incident (`open-loop-9d84d08cddf5` won for 9+ minutes straight in production) wasn't fully root-caused this session — plausible candidates include service-restart-driven habituation-memory resets (both `_recent_selected_counts` and the new `_first_selected_at` are in-memory only, wiped on every `orion-substrate-runtime` restart) or the competing candidates never having comparable evidence_strength regardless of habituation penalty. This fix (recency) is real and correct on its own regardless of that open question, but it may not be the sole explanation for the specific incident depth observed.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml up -d --build
```
Not run this session.

## Risks / concerns

- Severity: low
- Concern: `_first_selected_at` is in-memory only (matches `_recent_selected_counts`'s existing, already-accepted behavior) — resets on every service restart, meaning recency briefly "forgets" real history across a restart. Consistent with existing habituation-memory behavior, not a new limitation this patch introduces.
- Severity: low
- Concern: the "why didn't resonance save us" open question above — this fix addresses a real, confirmed bug, but may not be the complete explanation for the specific 9+-minute incident depth. Worth a follow-up investigation if the same class of stuck-coalition behavior recurs after this deploys.

## PR link

`gh` is unauthenticated in this environment — open manually at:
https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/attention-loop-recency-never-decays
