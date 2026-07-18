# Prediction-error attribution wiring — durable `contributing_turn_ids` + two consumers

## Summary

- Closed the durability gap PR #1205 disclosed: `orion/substrate/falkor_codec.py`'s
  `DYNAMICS_METADATA_KEYS` allowlist silently dropped `metadata['contributing_turn_ids']`
  on every durable FalkorDB round trip. Promoted it to a native Cypher property
  (`contributing_turn_ids_json`, same `_json_list()`/`_parse_json_list()` JSON-string
  pattern as `taxonomy_path_json`/`evidence_refs_json`).
- Decode is fail-open on malformed/missing/wrong-shape values (unlike `evidence_refs_json`,
  which intentionally raises — that field is load-bearing identity data, this one is
  best-effort attribution).
- Wired the now-durable list into two real consumers, satisfying the codec's own
  "promote only with a second consumer" rule:
  - `orion/substrate/attention_broadcast.py::substrate_pressure_signals()` — appends
    `contributing_turn_ids` onto `AttentionSignalV1.evidence_refs`.
  - `orion/substrate/attention_self_model.py::reduce_attention_self_model()` — new optional
    `harness_closure_signal` kwarg enriches the `field_salience_only` branch's
    `reason_narrative` with a sustained-surprise clause; every other branch and every
    existing caller is untouched (regression test proves byte-identical narrative when
    omitted).
- Design doc, PR report, and README status notes (per Juniper's explicit ask for both, sent
  mid-task) recording the decision: why the promotion is justified now, why there is zero
  `goal.drive_origin`/`GoalProposalEngine` coupling (checked against the full charter), and
  why the enrichment is scoped to `field_salience_only` only.
- `DYNAMICS_ENGINE_OWNED_METADATA_KEYS`, the 20-entry cap, and
  `_write_prediction_error_node()`'s carry-forward/dedup logic are all unchanged — this
  patch only makes the already-produced list durable and consumable.

## Outcome moved

The harness-closure prediction-error node's per-turn attribution (`contributing_turn_ids`)
now survives a `SubstrateDynamicsEngine.tick()`-triggered cache rebuild instead of silently
vanishing, and two real, already-shipped surfaces (`AttentionSignalV1.evidence_refs`, the
AST/HOT reducer's narrative) can now actually read it — closing the gap between "produced"
and "consumable" that PR #1205 explicitly disclosed and deferred.

## Current architecture

Before this patch: `_write_prediction_error_node()` (`services/orion-substrate-runtime/app/
worker.py`) wrote `metadata['contributing_turn_ids']` in-process, but `falkor_codec.py`'s
closed 5-key `DYNAMICS_METADATA_KEYS` allowlist had no slot for it, so the value was dropped
on every durable Falkor round trip. Zero consumers existed downstream of the in-process
write, since nothing durable could have fed one anyway.

## Architecture touched

- `orion/substrate/falkor_codec.py` — allowlist + encode/decode (contract layer).
- `orion/substrate/attention_broadcast.py` — one consumer (`evidence_refs`).
- `orion/substrate/attention_self_model.py` — one consumer (reducer narrative).
- `orion/sentience_striving_program/README.md`, `services/orion-substrate-runtime/README.md`
  — status/documentation only, no behavior change.
- No service, Docker, bus, or env surface touched. No new schema fields.

## Files changed

- `orion/substrate/falkor_codec.py`: `DYNAMICS_METADATA_KEYS` gains `contributing_turn_ids`
  (6th entry, was 5); `_dynamics_properties_from_metadata()` encodes
  `contributing_turn_ids_json`; `_dynamics_metadata_from_row()` decodes it fail-open,
  omitting the key when absent/empty/malformed.
- `orion/substrate/tests/test_falkor_codec.py`: dict-equality fixtures updated for the new
  always-present `contributing_turn_ids_json` key; 6 new tests (encode, decode, round-trip,
  empty-omitted, missing-key, malformed-JSON, wrong-shape-JSON).
- `orion/substrate/attention_broadcast.py`: `substrate_pressure_signals()` appends
  `contributing_turn_ids` (stringified) onto `evidence_refs`.
- `orion/substrate/tests/test_attention_broadcast.py`: 2 new tests (present/absent).
- `orion/substrate/attention_self_model.py`: `reduce_attention_self_model()` gains
  `harness_closure_signal: dict | None = None`; `field_salience_only` branch appends a
  narrative clause when `prediction_error > 0.0` and `contributing_turn_ids` is non-empty.
- `orion/substrate/tests/test_attention_self_model.py`: new `TestHarnessClosureSignal` class,
  5 tests (enrichment, byte-identical-when-omitted regression, zero-error non-enrichment,
  empty-turns non-enrichment, branch-isolation).
- `docs/superpowers/specs/2026-07-18-prediction-error-attribution-wiring-design.md`: new
  design-mode doc, the decision record for this patch.
- `orion/sentience_striving_program/README.md`: status notes under §6 item 2 and §9b
  items 2/3, dated 2026-07-18.
- `services/orion-substrate-runtime/README.md`: documents `_write_prediction_error_node`'s
  fixed-node-id + `contributing_turn_ids` mechanism and this patch's durability/consumer
  wiring, under the existing "Unified turn bus listeners" section.
- `docs/superpowers/pr-reports/2026-07-18-prediction-error-attribution-wiring-pr.md`: this
  report.

`orion/substrate/README.md` does not exist in this repo — checked, not invented.

## Schema / bus / API changes

- Added: `contributing_turn_ids_json` native Cypher property (concept nodes,
  `falkor_codec.py`); `reduce_attention_self_model(..., harness_closure_signal: dict | None
  = None)` optional keyword-only parameter.
- Removed: none.
- Renamed: none.
- Behavior changed: `AttentionSignalV1.evidence_refs` for substrate-pressure signals now
  includes contributing-turn ids when present on the source node (previously node-id-only).
  `AttentionSelfModelV1.reason_narrative`'s `field_salience_only` text gains a clause only
  when a caller explicitly supplies `harness_closure_signal` with non-trivial data — no
  existing caller does yet, so no live output changes until a future patch wires the caller.
- Compatibility notes: `DYNAMICS_METADATA_KEYS` growing by one key is additive and backward
  compatible — existing rows without `contributing_turn_ids_json` decode with the key simply
  absent from `metadata` (verified by a dedicated missing-key test). No `AttentionSelfModelV1`
  schema field added; `reason_narrative: str` already covers the new clause.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: n/a (no service touched).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: n/a, not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/substrate/tests -q
→ 381 passed, 17 warnings (pre-existing warnings, unrelated to this patch)

/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest scripts/analysis/tests/test_measure_ast_hot_reducer.py -q
→ 8 passed

/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-substrate-runtime/tests/test_worker_prediction_error_node.py services/orion-substrate-runtime/tests/test_worker_falkor_routed_store.py -q
→ 14 passed, 1 failed (test_write_prediction_error_node_preserves_dynamics_state_on_rewrite)
   Confirmed pre-existing and unrelated to this patch: stashed all changes, re-ran against
   unmodified origin/main HEAD (commit c214033e), same test fails identically. Not touched
   by this patch (DYNAMICS_ENGINE_OWNED_METADATA_KEYS is unchanged). Matches PR #1205's own
   report, which found "16 failed (pre-existing, confirmed identical on unmodified main via
   stash-and-rerun)" in this same test directory.
```

## Evals run

No dedicated eval harness exists for `orion/substrate/` beyond
`scripts/analysis/measure_ast_hot_reducer.py` (a real-data replay script, run below in lieu
of a formal eval). Per CLAUDE.md §11, the touched-service eval gap for `orion/substrate/`
generally is a known, standing gap (not introduced by this patch) — the module's own unit
test suite (381 tests) is the primary coverage.

## Docker/build/smoke checks

Not applicable at code-review time — no service, Dockerfile, or compose file touched.
`scripts/safe_graphify_update.sh` run after edits (see below).

**Live-data check (manual, in lieu of wiring FalkorDB reads into the Postgres-only
`measure_ast_hot_reducer.py` replay script — judged disproportionate for this patch, per the
task's own explicit escape hatch):**

```bash
redis-cli -p 6380 GRAPH.LIST
# graphiti_temporal, orion_substrate, orion_recall

redis-cli -p 6380 GRAPH.QUERY orion_substrate \
  "MATCH (n {node_id: 'node:substrate.harness_closure'}) RETURN n.node_id, n.dynamic_pressure, ..."
# empty result -- node does not exist yet

redis-cli -p 6380 GRAPH.QUERY orion_substrate \
  "MATCH (n) WHERE n.node_id CONTAINS 'harness_closure' RETURN n.node_id, labels(n)"
# harness_closure:<uuid> x3 -- pre-#1205 legacy per-event nodes, not the new shared node

docker inspect orion-athena-substrate-runtime --format '{{.Created}}'
# 2026-07-18T22:06:37Z -- container created after PR #1205 merged (21:59:55Z), and
# `docker exec orion-athena-substrate-runtime grep 'node:substrate.harness_closure'
# /app/app/worker.py` returns real matches, confirming the running image already has
# PR #1205's fixed-shared-node code (but not yet this patch's codec/consumer changes,
# which land with the restart below).

docker exec orion-athena-substrate-runtime env | grep -E \
  "SUBSTRATE_WRITE_PREDICTION_ERROR_NODES|ENABLE_POST_TURN_CLOSURE_LISTENER"
# SUBSTRATE_WRITE_PREDICTION_ERROR_NODES=true
# ENABLE_POST_TURN_CLOSURE_LISTENER=true
```

**Finding, reported plainly per CLAUDE.md's "runtime truth beats config truth":** the live
`orion-substrate-runtime` container already runs PR #1205's fixed-shared-node code, with the
feature flag enabled, but `node:substrate.harness_closure` does not yet exist as a node in
`orion_substrate` — zero `surprise_unresolved=True` `HarnessPostTurnClosureV1` events have
landed since that container's last restart. This means the live end-to-end durability claim
("`contributing_turn_ids` survives a tick-triggered cache rebuild in production") is
**UNVERIFIED** pending a real unresolved harness-closure event post-deploy of *this* patch.
The codec/consumer wiring itself is fully covered by unit tests against the real production
schemas (`ConceptNodeV1`, `AttentionSignalV1`, `AttentionSelfModelV1`) — this is an honest
gap in live proof, not in test coverage.

```bash
scripts/safe_graphify_update.sh
```

## Review findings fixed

- Finding: Design doc claimed `DYNAMICS_METADATA_KEYS` gains a "7th entry, was 5" and that
  the codec's docstring comment was "updated" to a new hardcoded count — both wrong; the
  tuple has 6 entries (5 original + 1 new), and the actual code comment was rewritten to
  drop the hardcoded count entirely rather than being bumped to a new number.
  - Fix: corrected the design doc's "Proposed schema / API changes" section to say "6th
    entry, was 5" and accurately describe the comment rewrite.
  - Evidence: `docs/superpowers/specs/2026-07-18-prediction-error-attribution-wiring-design.md`,
    commit after the review pass; `grep -n "5-key\|5 key" orion/substrate/falkor_codec.py`
    returns zero matches, confirming the comment no longer states a stale count.
- Finding: Design doc's "Files likely to touch" section referenced this PR report file
  before it existed on disk.
  - Fix: wrote this PR report in the same patch, resolving the reference.
  - Evidence: this file.
- Findings (a)/(b)/(c) from the task's explicit review checklist — fail-open JSON round-trip,
  regression-proof optional param, zero accidental goal_drive_origin/GoalProposalV1
  coupling — all independently verified correct by the reviewing subagent with no fixes
  needed. Full detail: reviewer confirmed `_dynamics_metadata_from_row()` catches
  `ValueError` from `_parse_json_list()` and degrades to omission (not raise); confirmed
  `harness_closure_signal` is structurally unreachable from the `top_down_override`/
  `bottom_up_salience` branches (the variable is never referenced there); confirmed zero new
  `goal_drive_origin`/`GoalProposalEngine`/`GoalProposalV1`/`capability_policy` references
  anywhere in the diff outside the design doc's own explanatory prose.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml \
  up -d --build
```

Required for the `contributing_turn_ids_json` durability fix and the two consumer changes to
take effect live. No other service touched.

## Risks / concerns

- Severity: **informational, not blocking**
  Concern: The live end-to-end durability claim (`contributing_turn_ids` surviving a real
  tick-triggered cache rebuild in production) is `UNVERIFIED` — no `node:substrate.
  harness_closure` write has occurred against the currently-running container since its last
  restart, so there is nothing live to observe yet.
  Mitigation: Post-deploy check named explicitly below. Codec/consumer logic is fully unit-
  tested against real production schemas in the interim.
- Severity: **pre-existing, not introduced by this patch**
  Concern: `test_write_prediction_error_node_preserves_dynamics_state_on_rewrite`
  (`services/orion-substrate-runtime/tests/test_worker_falkor_routed_store.py`) fails on
  unmodified `origin/main` too (confirmed via stash-and-rerun), consistent with PR #1205's
  own disclosed "16 failed (pre-existing)" finding in this test directory.
  Mitigation: None needed from this patch; flagged for whoever picks up that pre-existing gap
  next, not silently absorbed into this PR's scope.

## What to check live post-deploy

After the restart above, once a real unresolved harness-closure event occurs (a harness turn
with `surprise_unresolved=True`):

```bash
redis-cli -p 6380 GRAPH.QUERY orion_substrate \
  "MATCH (n {node_id: 'node:substrate.harness_closure'}) RETURN n.dynamic_pressure, n.prediction_error, n.contributing_turn_ids_json"
```

Confirm `contributing_turn_ids_json` is populated, and re-run the same query again after the
next `SubstrateDynamicsEngine.tick()` cadence (~30s, `SUBSTRATE_DYNAMICS_TICK_INTERVAL_SEC`)
to confirm the value is still present — i.e. it survived the cache rebuild that used to
silently drop it before this patch.

## PR link

<populated by `gh pr create` below>
