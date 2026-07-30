# CollapseMirror generative triggers: insight + flow gates

Branch: `feat/collapse-mirror-insight-flow-gates`
Design doc: `docs/superpowers/specs/2026-07-28-collapse-mirror-generative-triggers-design.md`

## Summary

- Adds two new `MetacogTriggerV1.trigger_kind` values — **`insight`** (surprise resolution / confidence recovery) and **`flow`** (sustained low-variance regime). These are the first non-error-shaped generative triggers in this codebase; every kind built this arc (`telemetry_anomaly`, `chat_turn`, `transport`, `relational`) fires on a rupture.
- Both read **one already-live field**, `AttentionSelfModelV1.prediction_error_confidence`, from the `substrate_attention_self_model` table (written every ~30s by `orion-substrate-runtime`'s `_attention_self_model_tick()`, PR #1459) as two different windowing functions. No new producer, reducer, schema field, or bus channel. `orion-substrate-runtime` is untouched.
- **`insight` is deliberately not a single-tick threshold crossing** — the one documented exception in this service. PR #1463 measured recoveries as a median 3-tick / max 12-tick gradual climb, so a point check would fire on noise partway up. It requires a sustained multi-tick low→high transition.
- Fixes the design doc's **Missing Question 6**: `_fallback_metacog_draft()` now consults `trigger_kind` first (`insight → epiphany`, `flow → flow`), before its phi-band guess. `trigger_kind` previously drove `CollapseMirrorEntryV2.type` in *no* path at all and `"epiphany"` was unreachable dead code.
- Thresholds calibrated against **2330 real ticks over ~21h**, not guessed — including two measured negative results that changed the shipped config.
- **Both gates ship disabled.** Each gets its own cooldown lane.

## Outcome moved

Orion can now, once enabled, notice and record two positive/neutral self-states it was previously structurally blind to: a surprise resolving, and a stretch of sustained calm. Before this patch, every trigger kind in the metacog family could only fire on something going wrong, and `CollapseMirrorEntryV2`'s own `epiphany → reorientation` entry type was unreachable by any code path.

Secondary, immediately real regardless of the flags: `trigger_kind` now influences downstream entry `type` at all. It previously did not, in any path.

## Current architecture (before this patch)

- A gate module per kind in `orion-equilibrium-service/app/<kind>_metacog_gate.py` evaluates evidence, builds a `MetacogTriggerV1`, and `_publish_metacog_trigger()` applies that kind's cooldown lane and publishes to `orion:equilibrium:metacog:trigger`. `orion-cortex-orch` dispatches a `log_orion_metacognition` plan; `orion-cortex-exec` drafts a `CollapseMirrorEntryV2`; `orion-sql-writer` persists to `orion_metacog`.
- All five existing kinds were rupture-shaped and all were single-tick-crossing gates.
- `orion-equilibrium-service`'s `app/` had no direct Postgres access (though it already read Postgres indirectly via `orion/substrate/felt_state_reader.py`).
- `CollapseMirrorEntryV2.type` was guessed **only** from phi bands, in `_fallback_metacog_draft()` — which despite its name is not fallback-only: the successful-LLM-draft path seeds its `base_entry` from it and the draft prompt forbids the LLM from emitting `type`. So that one heuristic decided every published entry's type, and could only ever produce `idle`/`turbulence`/`flow`.

## Architecture touched

- `orion-equilibrium-service`: new combined poll loop (`_generative_metacog_poll_loop`), first direct psycopg2 read in `app/`, two new cooldown lanes, `_publish_metacog_trigger` now reports whether it published.
- `orion/substrate/metacog_trigger_signals.py`: two new pure detectors (no I/O), keeping the math independently testable and out of the service layer.
- `orion-cortex-exec`: the type-guess heuristic in `_fallback_metacog_draft()`.
- No bus channel or schema-registry change (see below).

## Files changed

- `orion/substrate/metacog_trigger_signals.py`: new `ConfidenceSample`/`ConfidenceRecovery`/`FlowRegime` dataclasses and the two pure detectors `detect_confidence_recovery` / `detect_flow_regime`.
- `services/orion-equilibrium-service/app/attention_self_model_reader.py` (new): read-only psycopg2 reader; enforces `default_transaction_read_only`, fails open to `[]`.
- `services/orion-equilibrium-service/app/insight_metacog_gate.py` (new), `flow_metacog_gate.py` (new): trigger construction with real `upstream` evidence.
- `services/orion-equilibrium-service/app/service.py`: poll loop, per-gate evaluation with de-dupe, freshness guard, fetch-limit invariant, cooldown-lane registration, startup/shutdown wiring, connection cleanup.
- `services/orion-equilibrium-service/app/settings.py`: 17 new settings.
- `services/orion-equilibrium-service/.env_example`, `docker-compose.yml`: the same 17 keys (this service's compose has no blanket `env_file:`, so every key must be listed individually or it never reaches the container — enforced by an existing parity test).
- `services/orion-equilibrium-service/README.md`: two new rows in the trigger-family table plus a full section, including the measured calibration table and both negative results.
- `services/orion-cortex-exec/app/executor.py`: `trigger_kind`-first type mapping.
- `orion/schemas/telemetry/metacog_trigger.py`: `trigger_kind` docstring only.
- Tests: `tests/test_metacog_generative_trigger_signals.py`, `services/orion-equilibrium-service/tests/{test_insight_flow_metacog_gates,test_insight_flow_separate_cooldown,test_attention_self_model_reader,test_generative_metacog_gate_evaluation}.py`, `services/orion-cortex-exec/tests/test_metacog_draft_trigger_kind_type_mapping.py`.

## Schema / bus / API changes

- **Added**: two new `trigger_kind` string values, `"insight"` and `"flow"`.
- **Removed / renamed**: none.
- **Behavior changed**: `CollapseMirrorEntryV2.type` now derives from `trigger_kind` for these two kinds (`epiphany` / `flow`); previously it derived from phi bands for every kind. Every pre-existing kind's behavior is unchanged and regression-tested. `_publish_metacog_trigger()` now returns `bool` instead of `None`; every pre-existing caller ignores the return value.
- **Checked and NOT changed, explicitly** (per the task's requirement to verify rather than skip):
  - `orion/bus/channels.yaml` — no change needed. `orion:equilibrium:metacog:trigger` is already declared with `schema_id: "MetacogTriggerV1"` and `producer_services: ["orion-equilibrium-service"]` (line 829). No new channel; both gates publish on the existing one.
  - `orion/schemas/registry.py` — no change needed. `MetacogTriggerV1` is already registered (line 733). `trigger_kind` is a free-form `str`, so a new kind needs no migration, and `CollapseMirrorEntryV2.type` is likewise a free `str` with `"epiphany"`/`"flow"` already mapped in `DEFAULT_CHANGE_TYPE_BY_ENTRY_TYPE`.
- **Compatibility**: additive. No existing consumer sees a changed payload shape.

## Metric quality gate (CLAUDE.md §0A)

Re-run for this metric rather than inherited from the design doc:

1. **Provenance.** `orion/substrate/attention_self_model.py`'s `_aggregate_prediction_error_confidence()` computes `1.0 - mean(prediction_error)` over `ACTIVE_INFERENCE_DOMAINS`; persisted by `services/orion-substrate-runtime/app/worker.py::_attention_self_model_tick()` to `substrate_attention_self_model`. Traced to code, not schema comment.
2. **Independence.** insight and flow read the *same* field and are explicitly **not** independent of each other — they are two windowing functions over one signal, disclosed as such in code, README and the `upstream` payload (`evidence_source` is identical by design; `detector` differs). Neither is independent of the five prediction-error domains feeding the aggregate.
3. **Theory anchor.** Active-Inference prediction-error confidence: a resolved surprise is a real drop-then-recovery in prediction error, and sustained low prediction error with low variance is the standard characterization of a stable regime. `flow`/`epiphany` were chosen to match v1's existing `change_type` vocabulary rather than inventing new taxonomy.
4. **Live-data sanity.** Pulled real data (2330 ticks / ~21h): range 0.597–0.977, mean 0.893, 100% coverage, non-degenerate. Both bands reachable — 14 ticks at/below 0.70, 1544 at/above 0.90. Two measured negative results changed the config: `floor=0.92` yields **zero** qualifying windows (degenerate, documented as do-not-use), and at `floor=0.90` the variance ceiling is **non-binding** (identical 71 qualifying windows at max_stdev 0.02/0.03/0.05, because the field's ~0.977 ceiling squeezes any `min>=0.90` window into a <0.08 band). The variance ceiling was kept but disclosed as currently not doing work, rather than shipped as though it were.
5. **Existing mechanism.** Reuses the already-live field/table rather than building a new reducer. `FieldStateV1.recent_perturbation_ewma` was considered and set aside in the design doc (distinct causal chain, and that metric family has live-verified rest-point bugs needing its own sanity pass).
6. **Reversibility.** Cheap. Two flags off by default, no schema/manifest/training default baked in; deleting the two gate files, the poll loop and the settings block removes it entirely.

## Env/config changes

- **Added keys (17)**: `EQUILIBRIUM_METACOG_INSIGHT_TRIGGER_ENABLE`, `EQUILIBRIUM_METACOG_FLOW_TRIGGER_ENABLE`, `EQUILIBRIUM_METACOG_GENERATIVE_POLL_INTERVAL_SEC`, `EQUILIBRIUM_METACOG_GENERATIVE_POSTGRES_URI`, `EQUILIBRIUM_METACOG_GENERATIVE_WINDOW_TICKS`, `EQUILIBRIUM_METACOG_GENERATIVE_MAX_AGE_SEC`, `EQUILIBRIUM_METACOG_GENERATIVE_EXPECTED_TICK_SEC`, `EQUILIBRIUM_METACOG_GENERATIVE_SPAN_TOLERANCE`, `EQUILIBRIUM_METACOG_INSIGHT_COOLDOWN_SEC`, `EQUILIBRIUM_METACOG_FLOW_COOLDOWN_SEC`, `EQUILIBRIUM_METACOG_INSIGHT_LOW_THRESHOLD`, `EQUILIBRIUM_METACOG_INSIGHT_HIGH_THRESHOLD`, `EQUILIBRIUM_METACOG_INSIGHT_MAX_TICKS_TO_CROSS`, `EQUILIBRIUM_METACOG_INSIGHT_CONFIRM_TICKS`, `EQUILIBRIUM_METACOG_FLOW_FLOOR`, `EQUILIBRIUM_METACOG_FLOW_MAX_STDEV`, `EQUILIBRIUM_METACOG_FLOW_MIN_TICKS`
- **Removed / renamed**: none.
- `.env_example` updated: yes, all 17, with calibration rationale inline.
- `docker-compose.yml` updated: yes, all 17 (required — this service lists env vars individually).
- Local `.env` synced: **yes.** `scripts/sync_local_env_from_example.py` is a no-op from a linked worktree (`.env` is gitignored so worktrees have none; the live file is in the primary checkout), so the 17 keys were written directly into `/mnt/scripts/Orion-Sapienform/services/orion-equilibrium-service/.env` with values identical to `.env_example`. Verified present: 17/17, both enable flags `false`.
- Skipped keys requiring operator action: none.
- **Both new enable flags are `false` in the shipped `.env_example`, `False` in `settings.py`, `:-false` in `docker-compose.yml`, and `false` in the live `.env`.** Confirmed by a `docker compose config` render and a deterministic test.

## Tests run

```text
pytest services/orion-equilibrium-service/tests tests/test_metacog_generative_trigger_signals.py -q
  -> 2 failed, 172 passed

  Both failures are PRE-EXISTING and unrelated, in test_bus_synaptic_poll_e2e.py:
    - test_poll_above_threshold_triggers: feeds error=0.87 against its own
      threshold=1.0 and expects a fire (stale after the threshold was set to 1.0)
    - test_trigger_carries_edge_count_and_context: asserts `"reason" in <pydantic
      model>`, which is never valid (BaseModel has no __contains__)
  Confirmed unrelated: this branch touches neither that test file nor
  transport_metacog_gate.py (`git diff origin/main --stat` on both is empty).

pytest services/orion-cortex-exec/tests/test_metacog_publish_lane.py \
       services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py \
       services/orion-cortex-exec/tests/test_metacog_trigger_lineage.py \
       services/orion-cortex-exec/tests/test_metacog_route_profile.py \
       services/orion-cortex-exec/tests/test_metacog_draft_trigger_kind_type_mapping.py -q
  -> 34 passed

python scripts/check_service_env_compose_parity.py orion-equilibrium-service
  -> OK: all 80 .env_example keys are exposed via environment:
```

New test coverage (69 tests): 23 pure-detector, 16 service-layer (de-dupe, cooldown-retry, staleness, fetch-limit invariant, shipped-disabled), 11 gate-builder, 6 reader-parsing, 9 cooldown-lane-independence (including a 5-lane non-starvation case and a structural guard that both kinds stay registered), 8 executor type-mapping.

Note on the CLAUDE.md §17 scripts: `check_env_template_parity.py`, `check_schema_registry.py` and `check_bus_channels.py` **do not exist in this repo** — the `Makefile` documents this explicitly (there is no `agent-check` target either). `check_service_env_compose_parity.py` is the real equivalent and was run. The two contract surfaces were instead verified by direct inspection, reported above.

## Evals run

```text
No eval harness exists for orion-equilibrium-service (no services/orion-equilibrium-service/evals/
directory). Not claiming eval coverage.
```

In place of an eval harness, a **read-only replay against real production history** was used as the quality check — 2330 real ticks / ~21h of `substrate_attention_self_model` pushed through the actual reader, detectors and gate builders in sliding 20-row windows, simulating the real poll loop and cooldown lanes:

```text
insight episodes (de-duped on low_at):  4
insight publishes after 300s cooldown:  4
flow condition-true windows:           71
flow publishes after 1800s cooldown:    7

first real insight fire:
  reason: confidence_recovery:0.668->0.944:ticks_to_cross=4
  upstream: evidence_source=attention_self_model_prediction_error_confidence,
            detector=sustained_low_to_high_transition, low_value=0.6685,
            high_value=0.944, ticks_to_cross=4, cross_span_sec=120.5,
            confirm_ticks=2, window_ticks=20, low/high_threshold=0.7/0.9
first real flow fire:
  reason: flow_regime:min=0.902:mean=0.919:stdev=0.0156:ticks=20
  upstream: ..., detector=sustained_high_low_variance_regime, tick_count=20,
            span_sec=573.0, min_value=0.902, mean_value=0.91881,
            stdev_value=0.01559, floor=0.9, max_stdev=0.02
```

This satisfies Acceptance Check 3 as far as is possible without enabling the flags: real, non-empty, distinguishable `upstream` evidence. `cross_span_sec=120.5` for `ticks_to_cross=4` and `span_sec=573.0` for `tick_count=20` both match a real 30s cadence, confirming the contiguity guards read genuine consecutive ticks.

## Docker/build/smoke checks

```text
docker compose --env-file <root .env> --env-file <service .env> \
  -f services/orion-equilibrium-service/docker-compose.yml config
  -> renders cleanly; all 17 new keys resolve from the live .env, with
     EQUILIBRIUM_METACOG_INSIGHT_TRIGGER_ENABLE: "false"
     EQUILIBRIUM_METACOG_FLOW_TRIGGER_ENABLE: "false"

Live read-only smoke against the real table (no writes, no publishes, no flag changes):
  AttentionSelfModelReader.fetch_recent_samples(limit=20)
  -> 20 real rows, correct ascending order, both detectors correctly no-fire on
     the current window (newest value 0.777, below the high band), reader closed
     cleanly.
```

`scripts/safe_docker_build.sh` could not be used for this (it resolves `.env` relative to the worktree, which has none — `.env` is gitignored so linked worktrees never receive it). The raw read-only `config` invocation above is what CLAUDE.md §8 permits for one-off read-only checks. No build/`up` was run: both flags are off, so there is nothing new to observe at runtime until a human enables them.

`scripts/safe_graphify_update.sh`: the known destructive-update bug recurred (`graphify update .` produced 2467 nodes against an existing 28307). The wrapper refused and restored the backup, leaving nothing to commit; graph verified intact at 28307 nodes afterward. Consistent with the ~15 prior occurrences; root cause still unknown.

## Review findings fixed

Code review ran in a subagent and returned 2 must-fix, 6 should, 6 nits. Both must-fix findings were real bugs that also falsified claims I had written in docstrings and the README.

- **Finding (M1, must)**: neither gate had a freshness or tick-contiguity guard. Row adjacency is not tick adjacency — the reader silently drops rows with a missing/non-finite confidence, so 20 "consecutive" rows can span hours; and the writing tick is itself flag-gated, so it can stop while the poll loop keeps reading a frozen window that satisfies both conditions forever. Reproduced against the real detectors: a 20-row window covering **6.08h** fired flow while reporting `tick_count=20` as though it were 10 minutes of calm; a low **5 hours** before the high run fired insight reporting `ticks_to_cross=1`; and 20 rows all **3 days old** fired flow. This made two of my own docstrings and one README line false claims.
  - **Fix**: both detectors now bound their window in wall-clock seconds as well as ticks (`max_cross_span_sec` / `max_span_sec`, derived from `EXPECTED_TICK_SEC × SPAN_TOLERANCE`); the poll loop rejects any window whose newest row exceeds `GENERATIVE_MAX_AGE_SEC`, logging a warning that names the writing flag; and both record the real span in `upstream` so a stored row is self-auditing. Reused the `max_age` convention from this service's existing `felt_state_reader`.
  - **Evidence**: 7 new regression tests reproducing all three inputs (`test_insight_rejects_a_low_hours_before_the_high_run`, `test_flow_rejects_a_window_that_only_looks_consecutive`, `test_stale_window_is_rejected`, plus span-recording assertions). Post-fix real-data replay confirms neither gate became degenerate (insight still 4, flow still 71/7).

- **Finding (M2, must)**: `insight` double-fired on a single real recovery. `high_at` is the start of the *trailing* high run, not episode identity: when a high run breaks on one sub-threshold tick and re-forms, it re-anchors. Reproduced publishing the same recovery **twice, 390s apart** — clearing the 300s cooldown, so two `orion_metacog` rows, two LLM drafts, two `epiphany` entries for one event.
  - **Fix**: de-dupe on `recovery.low_at` (the arming low), which was identical across both fires; a genuinely new recovery necessarily has a new low crossing, so nothing legitimate is suppressed.
  - **Evidence**: `test_low_at_is_stable_when_the_high_run_breaks_and_reforms` asserts `len(high_ats) > 1` **and** `len(low_ats) == 1` over a simulated sliding window — it fails if the old key is restored. Service-layer `test_insight_does_not_refire_when_the_high_run_breaks_and_reforms` asserts exactly 1 publish. Real-history replay shows the fix landing: **5 fires before, 4 after**.

- **Finding (S1, should)**: the de-dupe key was consumed *before* the cooldown check, so a cooldown-suppressed event was marked seen and — since insight's key is stable — never retried.
  - **Fix**: `_publish_metacog_trigger` returns whether it actually published; the key is recorded only on `True`. Every pre-existing caller ignores the return value, so their behavior is unchanged.
  - **Evidence**: `test_cooldown_suppressed_insight_stays_retryable` / `..._flow_...` assert the key stays `None` while suppressed, then publishes once the cooldown elapses.

- **Finding (S2, should)**: `WINDOW_TICKS == FLOW_MIN_TICKS` with zero headroom; raising `FLOW_MIN_TICKS` above `WINDOW_TICKS` would silently make flow a permanent no-op, despite `.env_example` documenting the opposite.
  - **Fix**: `_generative_fetch_limit()` returns `max(window_ticks, flow_min_ticks, max_ticks_to_cross + confirm_ticks)`, making the documented invariant true by construction instead of by operator discipline.
  - **Evidence**: `test_fetch_limit_covers_flow_min_ticks_even_if_window_is_set_lower`, `test_fetch_limit_covers_insight_cross_plus_confirm`.

- **Finding (S3/S4, should)**: the service-layer de-dupe logic — the load-bearing correctness claim and the source of M2 — had **zero** test coverage, and the one "stability" test used an append-only list rather than the real sliding window, so it could not have caught M2.
  - **Fix**: new `test_generative_metacog_gate_evaluation.py` (16 tests) covering both evaluate methods, de-dupe, cooldown-retry, staleness, and the fetch-limit invariant; the misleading test was narrowed and its docstring corrected to point at the real one.
  - **Evidence**: 16 passed; the new sliding-window test fails against the pre-fix key.

- **Finding (S5, should)**: the reader's docstring falsely claimed to be the first Postgres access in this service's `app/` — `substrate_metacog_gate.py` already reaches Postgres via `felt_state_reader.py` on a live, default-enabled path, and that reader already had the `max_age` guard this one lacked (§0A step-5 existing-mechanism check not landing).
  - **Fix**: claim corrected, and the `max_age` pattern adopted (which M1 needed anyway).
  - **Evidence**: docstring in `attention_self_model_reader.py`.

- **Finding (S6, should)**: `logger.exception` on every failure path meant ~2880 stack traces/day during a Postgres outage, burying the first real fire the README tells an operator to watch for.
  - **Fix**: first failure logs a full traceback, consecutive failures log one warning line with a counter, counter resets on a successful query.
  - **Evidence**: `_log_failure()` in `attention_self_model_reader.py`.

- **Findings (N1, N3, N4, N6, nits)**: no deterministic test that the flags default off; an unreachable `try/except` in `_get_attention_self_model_reader` (construction does no I/O); `-> Any` on a function returning a concrete type; `trigger_kind` compared unstripped/uncased.
  - **Fix**: added `test_both_generative_flags_default_to_disabled`; removed the dead `try/except` and documented where the real fail-open lives; tightened the annotation to `-> AttentionSelfModelReader`; normalized the comparison with `.strip().lower()`.

Review also verified clean, worth recording: the detector index math survived an adversarial boundary sweep (empty/single/all-low/all-high/`confirm_ticks > len`/low-at-index-0) with no off-by-one and no recurrence of the `armed_tick_idx or i` falsy-zero bug; `asyncio.to_thread` offload and `CancelledError` handling are correct; env/settings/compose parity is exact in both directions; the `executor.py` change preserves prior behavior for every pre-existing kind; and `max_stdev` is on the same `statistics.stdev` estimator as the calibration figures, so there is no borrowed-constant scale mismatch.

**N2 (nit) accepted, not fixed**: on shutdown, if cancellation lands while `asyncio.to_thread(fetch_recent_samples)` is in flight, `close()` can shut the socket under a live query in the worker thread. Shutdown-only, and the query is already wrapped in a try/except that returns `[]`, so the worst case is one suppressed warning line during shutdown (now warning-only after S6).

## Restart required

`orion-substrate-runtime` is **untouched** — do not restart it.

```bash
# orion-equilibrium-service (new poll loop, new settings, new env keys)
cd /mnt/scripts/Orion-Sapienform
docker compose \
  --env-file .env \
  --env-file services/orion-equilibrium-service/.env \
  -f services/orion-equilibrium-service/docker-compose.yml \
  up -d --build

# orion-cortex-exec (trigger_kind -> entry type mapping)
docker compose \
  --env-file .env \
  --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml \
  up -d --build

# verify both came up
curl -fsS http://localhost:8081/health   # equilibrium (confirm port against its compose)
docker compose -f services/orion-equilibrium-service/docker-compose.yml logs --tail=50
```

Restarting is safe with both flags off: the poll loop returns immediately unless one is enabled, so no new Postgres connection is opened and no new trigger can fire. The `orion-cortex-exec` restart is what makes the type mapping live.

To enable later (a human decision, after a live-data check — **not** part of this patch):

```bash
# in services/orion-equilibrium-service/.env
EQUILIBRIUM_METACOG_INSIGHT_TRIGGER_ENABLE=true   # and/or
EQUILIBRIUM_METACOG_FLOW_TRIGGER_ENABLE=true
# then restart orion-equilibrium-service and watch for the first real row:
#   trigger_kind=insight|flow,
#   upstream.evidence_source=attention_self_model_prediction_error_confidence
```

## Risks / concerns

- **Severity: medium — thresholds are provisional.**
  - Concern: 0.70/0.90 and `max_stdev=0.02` come from a ~21h window on a table that has only existed since 2026-07-29. The design doc schedules a longer-window re-run for **2026-08-02**.
  - Mitigation: both gates ship disabled, so nothing fires until after that re-run; every threshold is env-tunable without a code change; the thresholds in force are recorded in each trigger's `upstream`, so stored rows stay interpretable after retuning. Marked provisional in code, `.env_example` and README.

- **Severity: low-medium — flow's variance ceiling currently does no work.**
  - Concern: at `floor=0.90` the qualifying-window count is identical at `max_stdev` 0.02/0.03/0.05, so the "low-variance" half of flow's definition is not currently binding. A reader could over-trust the name.
  - Mitigation: disclosed in three places rather than shipped silently; kept because it becomes binding if the floor is lowered (at 0.85, 426 windows pass the floor alone) or the field's range shifts; `stdev_value` is recorded in `upstream` so its real behavior stays auditable.

- **Severity: low — flow re-announces a state rather than firing once per episode.**
  - Concern: `ended_at` is not stable episode identity, so flow's volume is governed by its cooldown rather than by event boundaries.
  - Mitigation: measured — 71 condition-true windows reduce to 7 publishes per 21h at the 1800s default. Stated plainly in code and README rather than implied to be once-per-episode; a true anchor would need a second, wider query, deliberately deferred.

- **Severity: low — `orion_metacog` still has no confirmed real consumer.**
  - Concern: this is a standing open question in this service's README. Adding trigger kinds improves evidence quality but is not progress on that question and should not be reported as if it were.
  - Mitigation: none needed here; flagged so it is not mistaken for resolved.

- **Severity: low — 2 pre-existing test failures left in place.**
  - Concern: `test_bus_synaptic_poll_e2e.py`'s two stale assertions still fail.
  - Mitigation: unrelated to this branch (different trigger kind, untouched files). Not fixed here because deciding whether the test or `transport`'s 1.0 threshold is wrong is a real calibration judgment about `transport`, not a typo — worth its own small patch rather than being buried in this one.

## PR link

<filled in on creation>
