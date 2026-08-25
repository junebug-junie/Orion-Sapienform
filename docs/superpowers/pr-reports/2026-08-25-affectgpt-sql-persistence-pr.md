# orion-sql-writer: durable persistence for orion:affectgpt:assessment

## Summary

- `orion-sql-writer` is now the first real durable-persistence consumer of `orion:affectgpt:assessment` (`JuniperMultimodalAffectV1`) -- new table `juniper_multimodal_affect_log` via a new `JuniperMultimodalAffectSQL` model.
- Closes the specific follow-up flagged when PR #1865 shipped: the only durable trace of a real AffectGPT capture was a 1h-TTL Redis SETEX mirror (`orion/situational/juniper_affect_state.py`) -- once that key expires, or a redeploy clears Redis, nothing recorded a capture ever happened.
- `event_id` has no field on the wire schema (unlike the sibling tiling-window text signal, `JuniperAffectiveStateSQL`) -- synthesized from the envelope `correlation_id` via a new pure, unit-tested helper (`_affectgpt_multimodal_event_id`), so a re-publish of the same tick merges instead of duplicating.
- Privacy boundary: `transcript` (Whisper's verbatim transcription of Juniper's spoken words) is never persisted -- no column declared, and (after a review-caught gap, see below) the write-failure fallback path explicitly redacts it too. `raw_response` (the model's own generated affect read, already live in the Hub UI/bus) is kept in full.
- `channels.yaml`, `.env_example`, hand-synced `.env`, both services' READMEs, and CI's test allowlist all updated in the same changeset.

## Outcome moved

A real capture (webcam+mic, GPU1 AffectGPT inference on circe) now has a durable Postgres row independent of Redis TTL/redeploy state, closing a genuine "the data exists nowhere after an hour" gap that was flagged but not yet fixed when the live cognition path shipped (PR #1865).

## Current architecture

Before this patch: `orion:affectgpt:assessment` had exactly one live-subscribable consumer, `services/orion-juniper-affective-state/scripts/tap_assessments.py` (a manual debug CLI), plus (since PR #1865) the Redis SETEX mirror feeding chat-turn situational context. `OrionBusAsync.publish()` is plain Redis pub/sub, not a stream -- with no durable-storage subscriber, every event that isn't caught by the mirror or the debug tap is gone once its TTL expires.

## Architecture touched

- `services/orion-sql-writer`: new model, `MODEL_MAP`/`DEFAULT_ROUTE_MAP`/subscribe-channel wiring, a new `event_id`-synthesis helper, a redaction fix in the shared write-failure fallback path.
- `orion/bus/channels.yaml`: `orion-sql-writer` added to `orion:affectgpt:assessment`'s `consumer_services`, comment corrected.
- `services/orion-juniper-affective-state`: README note (no code change -- this service already publishes everything needed).

## Files changed

- `services/orion-sql-writer/app/models/juniper_multimodal_affect.py`: new `JuniperMultimodalAffectSQL` model.
- `services/orion-sql-writer/app/models/__init__.py`: registered the new model.
- `services/orion-sql-writer/app/settings.py`: `DEFAULT_ROUTE_MAP` entry, `orion:affectgpt:assessment` added to the default subscribe list.
- `services/orion-sql-writer/app/worker.py`: import, `MODEL_MAP` entry, new `_affectgpt_multimodal_event_id()` helper + call site, and a scoped redaction in the shared write-failure `except` block (review fix).
- `services/orion-sql-writer/.env_example`: `SQL_WRITER_SUBSCRIBE_CHANNELS` and `SQL_WRITER_ROUTE_MAP_JSON` both updated (the subscribe list REPLACES rather than merges with the env value, so this one is load-bearing; the route map is forgiving -- code defaults still apply even if an env copy lags -- but kept in sync for consistency).
- `orion/bus/channels.yaml`: `orion-sql-writer` added to `orion:affectgpt:assessment`'s `consumer_services`; comment corrected (a cognition consumer exists via a separate Redis mirror, not a subscription to this channel).
- `services/orion-sql-writer/README.md`, `services/orion-juniper-affective-state/README.md`: persistence documented.
- `.github/workflows/orion-sql-writer-tests.yml`: new SQL-shape test file added to the unit job's explicit allowlist (this workflow runs a hardcoded file list, not the whole `tests/` directory -- without this the new tests would never run in CI, a gap that pre-existing sibling test files (`test_juniper_affective_state_sql_shape.py`, `test_dev_economics_ledger_sql_shape.py`) still have; not fixed here to keep this patch scoped to what I added).
- `services/orion-sql-writer/tests/test_juniper_multimodal_affect_sql_shape.py`: new, 16 tests.

## Schema / bus / API changes

- Added: `JuniperMultimodalAffectSQL` / `juniper_multimodal_affect_log` table (no migration needed -- `Base.metadata.create_all(bind=engine)` runs at boot).
- No changes to `JuniperMultimodalAffectV1` itself (already registered in `orion/schemas/registry.py`).
- `orion:affectgpt:assessment` gains `orion-sql-writer` as a second real consumer.
- Compatibility notes: additive only, no existing consumer or payload shape changed.

## Env/config changes

- Added: nothing new (no new env *keys* -- `orion:affectgpt:assessment` appended to the existing `SQL_WRITER_SUBSCRIBE_CHANNELS` list value, and the corresponding route added to the existing `SQL_WRITER_ROUTE_MAP_JSON` value).
- `.env_example` updated: yes (both keys' values).
- Local `.env` synced: hand-edited directly (subscribe-channels value only -- the route-map value is deliberately left alone per established precedent: `Settings.route_map` merges the env JSON *over* code defaults, so a route missing from a stale `.env` copy still resolves via `DEFAULT_ROUTE_MAP`; `scripts/sync_local_env_from_example.py` confirmed (again) to silently skip existing-key value changes, so this was done by hand rather than trusted to the script).
- Skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=.:services/orion-sql-writer .venv/bin/python -m pytest -q \
  orion/grammar/tests/test_ledger.py \
  services/orion-sql-writer/tests/test_consumer_resilience.py \
  services/orion-sql-writer/tests/test_route_map_completeness.py \
  services/orion-sql-writer/tests/test_world_pulse_routing.py \
  services/orion-sql-writer/tests/test_grammar_event_routing.py \
  services/orion-sql-writer/tests/test_grammar_ledger_sql_shape.py \
  services/orion-sql-writer/tests/test_juniper_affective_state_sql_shape.py \
  services/orion-sql-writer/tests/test_dev_economics_ledger_sql_shape.py \
  services/orion-sql-writer/tests/test_juniper_multimodal_affect_sql_shape.py \
  services/orion-sql-writer/tests/test_route_coverage.py
73 passed
```

## Evals run

None -- this service has no eval harness under `services/orion-sql-writer/evals/`; not adding one here since this patch is a straight persistence-wiring change with no quality/behavior dimension an eval would measure (mirrors the precedent set by the sibling `JuniperAffectiveStateSQL`/`DocSemanticDriftSQL` persistence patches).

## Docker/build/smoke checks

Not run -- no runtime behavior, port, health check, or dependency changed; `Base.metadata.create_all()` creates the new table idempotently at next boot with no migration step. Flagging for Juniper: this needs a real restart to pick up the code (see below), and the actual first live row is worth a quick spot-check after that.

## Review findings fixed

- Finding: the transcript-redaction privacy claim only held on the success path -- the shared write-failure fallback (`_write_fallback`) persisted the RAW, unfiltered `env.payload` (including `transcript`) into `bus_fallback_log` on any exception (schema drift, DB error).
  - Fix: explicit `sql_model is JuniperMultimodalAffectSQL` redaction scoped to that one `except` block in `worker.py`, not a change to the shared handler other models rely on.
  - Evidence: new test `test_transcript_is_redacted_from_the_fallback_log_on_write_failure` forces `_write` to raise, asserts `transcript` is stripped from what reaches `_write_fallback` while `raw_response` and the error message survive; passes.
- Finding: `source` column's SQLAlchemy-side `default="affectgpt"` could never fire (the wire schema's `Literal["affectgpt"]` always supplies it), misleadingly implying the value is optional.
  - Fix: removed the default, documented why in a code comment.
  - Evidence: existing tests (including the SQLite merge-idempotency test, which sets `source` explicitly) still pass.
- Finding: "a redelivered event upserts" was stated without the caveat that `OrionBusAsync` is plain Redis pub/sub with no redelivery -- the merge-based key only protects against an actual re-publish, and a capture published while `orion-sql-writer` itself is disconnected is lost before reaching this table, same as any bus consumer.
  - Fix: softened the model docstring and the `orion-juniper-affective-state` README to state this precisely, and documented that the one real producer already explicitly keeps envelope-level and payload-level `correlation_id` in sync (rather than asserting that in the abstract).
  - Evidence: `services/orion-juniper-affective-state/app/main.py:416-431` (`_publish_event`) read directly to confirm the sync behavior described.
- Finding (disclosed, not fixed): this is the 4th near-identical inline "synthesize a primary key from correlation_id/env.id/uuid4" special case in `handle_envelope()` (alongside `CollapseMirror`, `MetacogEntry`, `CollapseEnrichment`) -- a real generalization opportunity.
  - Not fixed: refactoring 3 unrelated, already-merged call sites is out of scope for a persistence-only patch and risks unrelated regressions; noted here as a legitimate follow-up.
- Finding (disclosed, not fixed): CLAUDE.md's metric quality gate isn't restated in this PR.
  - Not fixed: judged not to apply -- this patch durably persists an already-existing, already-vetted event (`JuniperMultimodalAffectV1` predates this PR by the AffectGPT worker's own rollout), it does not introduce a new metric/signal/detector.

## Restart required

```bash
# Pick up the new model/route/subscribe-channel wiring:
docker compose \
  --env-file .env --env-file services/orion-sql-writer/.env \
  -f services/orion-sql-writer/docker-compose.yml \
  up -d --build
```

## Risks / concerns

- Severity: low
- Concern: the write-failure fallback path (`bus_fallback_log`) now has a bespoke per-model redaction rule rather than a generic mechanism -- a future new model with its own sensitive-but-undeclared field would need the same treatment added by hand, or it inherits the original gap.
- Mitigation: documented explicitly in the model's docstring and this report so it's discoverable; the underlying generalization opportunity (one shared id-synthesis/redaction helper across all 4 special cases) is flagged as a real follow-up above, deliberately left out of this patch's scope.

## PR link

<!-- filled in after `gh pr create` -->
