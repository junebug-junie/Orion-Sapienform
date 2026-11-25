## Summary

Three organs that only had an exemption now write their experiences into Orion's shared record (`temporal_self_event`), so every organ that reads Orion's now can see them.

- **Intention:** each closed intention becomes one experience: carried (the reply was stored) or not carried (with the reason). Source kind `intention_close`.
- **Day letter:** each written letter becomes one experience. Its threads ride along as refs, the same refs Prime's thread concerns carry. Source kind `day_letter`.
- **Personality:** each Prime choice that the personality loop folded becomes one experience, with the leaning it joined. Source kind `personality_choice`.
- `config/integration/organs.yaml` drops the three `exempt_emit` entries. Every organ now emits.

Part of #3429 (epic #3434).

## Outcome moved

Before: 13 of 16 organs emitted experiences. After: all 16 do.

Live reads over all history, SELECT only (2026-11-25):

- `intention_close`: 5 events. 3 were produced. 2 were not produced, because Hub refused the order (no thread to answer in).
- `day_letter`: 57 events, 2026-10-01 to 2026-11-24. The latest carries 5 thread refs.
- `personality_choice`: 92 events. 68 declined every option. 24 chose one.

## Current architecture

The spec (2026-11-15 integration design, A1) names six missing sources. Patch 1 added reading_turn, prediction_surprise and prime_thought. These three were left `exempt_emit: "... integration patch 7"`.

## Architecture touched

- `orion/schemas/temporal_self.py`: three new `TemporalSourceKind` values. They are ADDITIVE on forbid models.
- `orion/temporal_self/sources.py`: three adapters, plus `EXPERIENCE_ONLY_KINDS` and `EXPERIENCE_ONLY_TABLES`.
- `services/orion-durable-runs/app/temporal_self_sources.py`: three `SourceQuery` reads.
  - `intention_close`: Prime's `make intention:<id>` acts, 4-day slack, with a lateral join to the carry run's act.
  - `day_letter`: by `created_at`.
  - `personality_choice`: lane `prime`, by fold time, 3-day slack on the (lane, occurred_at) index.
- `orion/temporal_self/now.py`: plain-word `now_label` fallbacks for a stored row with no label.
- `config/integration/organs.yaml`: registry.
- Spec A1 line: corrected the `intention_close` source.

## Files changed

- `orion/schemas/temporal_self.py`: new kinds; stale comment fixed.
- `orion/temporal_self/sources.py`: the adapters `intention_close`, `day_letter`, `personality_choice` and `letter_threads`.
- `orion/temporal_self/now.py`: fallback labels.
- `services/orion-durable-runs/app/temporal_self_sources.py`: the reads.
- `config/integration/organs.yaml`: exemptions removed; the overlap is noted.
- `orion/temporal_self/tests/test_now.py`: one test per adapter, plus fallback labels and the thread-ref contract with `orion/prime/day_letter.py`.
- `services/orion-durable-runs/tests/test_temporal_self_patch7a_sources_postgres.py`: new. Runs the three reads on real Postgres. Covers the window edges, an expired take and the lane filter.
- `tests/test_check_integration_seam.py`: the "emit or say why" test now plants its own non-emitting organ, because no real organ is exempt any more.
- `docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md`: A1 source line.

## Schema / bus / API changes

- Added: `TemporalSourceKind` values `intention_close`, `day_letter`, `personality_choice`.
- Removed, renamed: none.
- Behavior changed: orion-durable-runs' chronicle now writes these kinds into `temporal_self_event`.
- Compatibility: these values are additive on `extra=forbid` and `Literal` models (`TemporalNowItemV1`). The services that read the now must deploy before orion-durable-runs:
  - orion-thought
  - orion-hub
  - orion-mind
  - orion-prediction

  The FCC sandbox sync also runs after the deploy.
- No bus channel. No migration.

## Env/config changes

- No env keys changed. No `.env_example` changed.

## Tests run

```text
pytest orion/temporal_self/tests orion/dig/tests tests/test_check_integration_seam.py tests/test_report_integration_coverage.py  -> all passed
pytest services/orion-durable-runs/tests (throwaway postgres:16 container, ORION_ADMISSION_TEST_DSN)  -> 1269 passed, 1 skipped
re-run after review fixes: patch7a postgres + experience_now tests -> 19 passed
pytest services/orion-sql-writer/tests/test_act_downstream_use.py -> 6 passed
python scripts/check_integration_seam.py --require-base -> OK (16 organs; 35 source kinds covered)
python scripts/check_env_template_parity.py -> ok
```

## Evals run

```text
pytest orion/integration/evals -> 12 passed
python -m orion.integration.evals.run_now_ablation_eval --offline -> PASS
Live read (SourceReader on live conjourney, SELECT only, 2026-09-01..now): counts above
```

## Docker/build/smoke checks

```text
No image built. Runtime is UNVERIFIED until the deploy: no live temporal_self_event row of these kinds exists yet.
```

## Metric gate

No magnitude was added. All three kinds are unscored, so salience is NULL. `intention_close` passes the act close's own `surprise_nats` into the column, as prime_thought does. It is not used for salience.

## Review findings fixed

- Finding: the `intention_close` 2-day slack dropped expired takes. The take horizon is 72 h + 1 h, and live horizons run up to 3d01h.
  - Fix: slack is now 4 days. Added a Postgres test row: opened 3d2h before the window, closed inside it.
  - Evidence: `test_patch7a_reads_return_only_window_rows_through_their_adapters`.
- Finding: the `personality_choice` 1-day slack loses folds if the loop is down for more than a day.
  - Fix: slack is now 3 days, served by the index.
- Finding: the carry-run join picked the latest run, not the one inside the take.
  - Fix: ASC order, bounded by `a.closed_at`.
- Finding: fallback labels could print "None", and a string `declined_all` read as true.
  - Fix: both are guarded and tested.
- Finding: a stale schema comment, and the spec named the wrong `intention_close` source.
  - Fix: both updated.
- Finding: every `personality_choice` repeats a `prime_thought` about 10-20 min later. This is kept on purpose (spec: one per choice point).
  - Fix: documented in `organs.yaml`. Both experiences carry the same `act:` ref.

## Restart required

Order matters (additive Literal):

```bash
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
scripts/safe_docker_build.sh orion-prediction up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- Severity: low.
  - Concern: if orion-durable-runs deploys before the readers, a reader that parses a now item with a new kind fails on the forbid model.
  - Mitigation: follow the restart order above.
- Severity: low.
  - Concern: the Inner Sky does not yet know the verdict words `not_produced`, `chose` and `declined_all`. They render neutral, and its feed eval lists them as unclassified.
  - Mitigation: a separate Inner Sky PR (out of scope here).
- Issues opened: #3963 (main is red on the leased lane caller gate because of scripts/analysis/reflection_register_sources.py. It came from the self-question door PR, not this one. This PR inherits that one failing check). Issues closed: none (#3429 stays open for 7b and 7c).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3960

🤖 Generated with [Claude Code](https://claude.com/claude-code)
