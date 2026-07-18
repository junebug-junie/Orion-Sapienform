# Prediction-error attribution wiring — durable `contributing_turn_ids` + two consumers

Status: design mode + implemented in the same PR (root CLAUDE.md's implementation-mode
follow-through requirement — this doc captures the decision record, the code lands alongside
it, not as a separate follow-up).

## Arsonist summary

PR #1205 (`feat/harness-closure-pressure-accumulation`) gave the harness-closure
prediction-error lane a shared, fixed `node_id` (`node:substrate.harness_closure`) so
repeated unresolved-surprise events accumulate onto one node instead of spawning a new node
per turn, and added `metadata['contributing_turn_ids']` (capped at 20, deduped, oldest
dropped) to attribute which harness turns contributed to that node's current pressure. That
PR shipped with a disclosed, deliberate gap: `orion/substrate/falkor_codec.py`'s
`DYNAMICS_METADATA_KEYS` is a closed 5-key allowlist for native Cypher scalar properties on
concept nodes, and `contributing_turn_ids` was not in it — so every value written into that
key was silently dropped the next time `SubstrateDynamicsEngine.tick()`'s ~30s cache rebuild
re-hydrated the node from durable Falkor rows (`_hydrate_from_durable()`). The in-process
write briefly held the full value; nothing durable did. PR #1205's own brief explicitly
scoped `falkor_codec.py` out ("do not touch `DYNAMICS_ENGINE_OWNED_METADATA_KEYS` or
`falkor_codec.py`"), so the fix was deferred, not skipped.

This patch closes that gap and, per the codec's own documented rule ("promote to
first-class Cypher properties only with a second consumer"), wires the now-durable list into
two real consumers in the same changeset — satisfying the rule rather than promoting the key
speculatively ahead of any consumer.

## Current architecture

- **Producer (unchanged by this patch):** `services/orion-substrate-runtime/app/worker.py`'s
  `_write_prediction_error_node()` accumulates `metadata['contributing_turn_ids']` on
  repeat writes to a shared `node_id` (PR #1205). Carry-forward logic, the 20-entry cap, and
  dedup are all already shipped and untouched here.
- **Codec gap (fixed by this patch):** `orion/substrate/falkor_codec.py`'s
  `DYNAMICS_METADATA_KEYS` allowlist controlled which metadata keys survive an
  encode→Falkor-write→decode round trip as native Cypher scalar properties.
  `contributing_turn_ids` (a `list[str]`) could not be added directly — Cypher properties on
  a node are scalars, not arrays — so it needed the same JSON-string-encoding pattern already
  used for `taxonomy_path_json`/`evidence_refs_json` (`_json_list()`/`_parse_json_list()`
  helpers, `encode_node_properties()`/`_provenance_from_row()`).
- **Consumers before this patch:** zero real consumers of `contributing_turn_ids` existed
  downstream of the in-process write — the durability gap meant nothing could have consumed
  a durable value even if it wanted to.

## Why `contributing_turn_ids` qualifies for promotion now

`falkor_codec.py`'s own comment on `DYNAMICS_METADATA_KEYS` states the rule: promote a
metadata key to a native Cypher property "only with a second consumer" — i.e. the codec
allowlist is not a dumping ground for anything a producer happens to write; it exists because
two already-shipped consumers (`SubstrateDynamicsEngine.tick()`, `attention_broadcast.
_node_salience()`) genuinely read those keys back out. `contributing_turn_ids` had a
producer (PR #1205) but zero consumers, which is exactly the state the rule is designed to
block from silently accumulating unused allowlist entries.

This patch adds two real consumers in the same changeset, which is what makes the promotion
legitimate rather than speculative:

1. `orion/substrate/attention_broadcast.py::substrate_pressure_signals()` — already builds
   `AttentionSignalV1.evidence_refs` from the node id. This patch appends
   `contributing_turn_ids` (stringified) onto that list, so a workspace-competition signal
   sourced from a sustained-surprise node carries the specific harness turns that produced
   it, not just an opaque node id.
2. `orion/substrate/attention_self_model.py::reduce_attention_self_model()` — the AST/HOT
   reducer (PR #1196/#1196-follow-ups). Gains an optional `harness_closure_signal` kwarg
   that, when the `field_salience_only` branch fires, enriches `reason_narrative` with a
   clause naming the contributing-turn count and current `prediction_error` magnitude.

Both consumers read the same underlying data (`contributing_turn_ids` + `prediction_error`
off the one shared node), but they are not redundant with each other in the sense the metric
quality gate's independence check cares about: `evidence_refs` is a flat inspectability list
consumed by whatever reads `AttentionSignalV1` (debug surfaces, downstream salience scoring),
while the reducer's narrative is a distinct textual "why" artifact consumed by a different
read-only measurement path (`scripts/analysis/measure_ast_hot_reducer.py`). Per the metric
quality gate (root CLAUDE.md §0A): this is not a *new* metric being minted, it is the same
already-gated `prediction_error`/`contributing_turn_ids` signal (traced to
`_write_prediction_error_node()`, theory-anchored to the charter's §9b item 3 Predictive
Processing thread, already confirmed live 2026-07-18 per the README) becoming durable and
inspectable through two seams that already existed. No new independence/theory-anchor
analysis is owed here beyond what PR #1205 and the README's item 3 entry already did.

## Why this has zero coupling to `goal.drive_origin` / the halted `GoalProposalEngine`

Checked against the full `orion/sentience_striving_program/README.md` charter, not just a
grep for the string:

- `goal.drive_origin` is produced by `GoalProposalEngine`
  (`orion/spark/concept_induction/`), which the charter's §8 explicitly halted. The only
  charter item that currently depends on `goal.drive_origin` is §6 item 6
  ("Revisit `capability_policy.py`'s coupling to live salience"), gated on §6 item 3 closing
  that dependency first — neither item 6 nor `capability_policy.v1.yaml` is touched by this
  patch.
- The prediction-error lane this patch extends is §9b item 3 (Predictive Processing/Active
  Inference), confirmed in the README as field-native substrate — `execution_prediction_
  error()`/`transport_prediction_error()` writing directly onto `FieldStateV1` nodes, no
  `SelfStateV1` or `goal.drive_origin` anchor. `node:substrate.harness_closure` (this
  patch's actual subject) is the same `_write_prediction_error_node()` mechanism, same
  no-goal-provenance shape.
- The AST/HOT reducer's `voluntary_override`/`goal_drive_origin` field (used only in the
  *separate* `top_down_override` branch, `VoluntaryOverrideV1.goal_drive_origin`) is
  explicitly untouched by this patch — the new `harness_closure_signal` kwarg only affects
  the `field_salience_only` branch, which has no `goal_drive_origin` field at all in its
  narrative. Grepped the full diff for `goal_drive_origin`/`GoalProposalV1`/
  `capability_policy` post-patch: zero matches outside the pre-existing, untouched
  `top_down_override` branch.

## Why this is scoped to the `field_salience_only` branch only

`reduce_attention_self_model()` has three narrative-bearing branches:

- `top_down_override` — already has a real, specific "why" (`VoluntaryOverrideV1`'s
  chosen/beaten loop ids, applied bias, goal artifact/drive-origin). Not touched.
- `bottom_up_salience` — already has a real "why" (pure salience dispatch, no override
  active). Not touched.
- `field_salience_only` — the fallback branch when the GWT-dispatch lane is stale/absent.
  Before this patch, its narrative only reported an aggregate `field_overall_salience`
  number with no per-node attribution — the one branch with no specific "why" beyond a bare
  salience reading. This is exactly the gap `harness_closure_signal` fills: when a real,
  sustained prediction-error node exists, the narrative can now say *which* recent turns
  drove that surprise and how large it currently is, instead of staying silent on it.

Widening the two other branches to also read `harness_closure_signal` was considered and
rejected: both already have a real, specific, non-generic "why" from their own primary
signal (`VoluntaryOverrideV1` / fresh broadcast salience) — adding a second, unrelated signal
into an already-answered "why" would blur rather than sharpen the narrative, and neither
branch's own docstring/tests anticipate it.

## Proposed schema / API changes

- `orion/substrate/falkor_codec.py`:
  - `DYNAMICS_METADATA_KEYS` gains `"contributing_turn_ids"` (6th entry, was 5) — the
    comment above the tuple was rewritten to drop the hardcoded "closed 5-key allowlist"
    count entirely (now refers to the tuple itself, so it can't go stale again on the next
    addition) rather than being bumped to a new fixed number.
  - `DYNAMICS_ENGINE_OWNED_METADATA_KEYS` unchanged (4 entries) — `contributing_turn_ids` is
    owned by `_write_prediction_error_node()`'s own carry-forward logic, a distinct
    mechanism from `SubstrateDynamicsEngine.tick()`'s pressure/dormancy computation.
  - New native Cypher property: `contributing_turn_ids_json` (JSON-encoded string list,
    same pattern as `taxonomy_path_json`/`evidence_refs_json`). Decode is fail-open
    (malformed/missing → key omitted from `metadata`, no raise) — deliberately different
    from `evidence_refs_json`'s raise-on-corruption behavior, because `evidence_refs` is
    load-bearing identity/provenance data and `contributing_turn_ids` is best-effort
    attribution metadata.
- `orion/substrate/attention_broadcast.py::substrate_pressure_signals()`:
  `AttentionSignalV1.evidence_refs` now includes `contributing_turn_ids` (stringified)
  appended after the node id, when present. No schema change to `AttentionSignalV1` itself
  (already a `list[str]` field).
- `orion/substrate/attention_self_model.py::reduce_attention_self_model()`: new
  keyword-only, optional parameter `harness_closure_signal: dict | None = None` — a plain
  dict (`{"prediction_error": float, "contributing_turn_ids": list[str]}`), not a Pydantic
  model, not a raw graph node import. No `AttentionSelfModelV1` schema field added —
  `reason_narrative: str` is sufficient (per the task's own scope check, confirmed by
  reading `orion/schemas/attention_self_model.py`).

## Files likely to touch (this patch's actual diff)

- `orion/substrate/falkor_codec.py` — allowlist + encode/decode.
- `orion/substrate/tests/test_falkor_codec.py` — round-trip, omitted-when-empty, fail-open
  malformed/missing coverage.
- `orion/substrate/attention_broadcast.py` — `evidence_refs` wiring.
- `orion/substrate/tests/test_attention_broadcast.py` — present/absent coverage.
- `orion/substrate/attention_self_model.py` — `harness_closure_signal` kwarg +
  `field_salience_only` narrative clause.
- `orion/substrate/tests/test_attention_self_model.py` — enrichment, byte-identical
  regression, zero-error/empty-turns non-enrichment, branch-isolation coverage.
- `orion/sentience_striving_program/README.md` — §6 item 2 and §9b items 2/3 status notes.
- `docs/superpowers/pr-reports/2026-07-18-prediction-error-attribution-wiring-pr.md` — PR
  report (this patch's follow-through artifact).
- This doc.

## Non-goals

- Not touching `DYNAMICS_ENGINE_OWNED_METADATA_KEYS`, the 20-entry cap, or the dedup/carry-
  forward logic in `_write_prediction_error_node()` — all already shipped by PR #1205,
  unchanged here.
- Not touching `top_down_override`, `TopDownBiasCombiner`, `GoalContext`, `goal_context.py`,
  `orion/substrate/attention/top_down.py`, `capability_policy.v1.yaml`, or anything
  Objective-6-related.
- Not adding a bus consumer/producer for `AttentionSelfModelV1` — it stays read-only per
  Phase 1's explicit scope (module docstring, `orion/substrate/attention_self_model.py`).
- Not wiring a live FalkorDB read into `scripts/analysis/measure_ast_hot_reducer.py` (a
  Postgres-only replay script) — disproportionate for this patch's scope; the live check for
  this PR was done manually via `redis-cli GRAPH.QUERY` against the running
  `orion-athena-falkordb` container and reported in the PR report instead.
- Not adding a new `AttentionSelfModelV1` schema field — `reason_narrative: str` already
  covers the new clause.

## Acceptance checks

- `pytest orion/substrate/tests -q` passes with the new/updated tests (round-trip,
  fail-open decode, `evidence_refs` present/absent, narrative enrichment/regression/branch-
  isolation).
- A concept node's `metadata['contributing_turn_ids']` survives
  `encode_node_properties()` → `decode_concept_node()` without requiring the in-process
  cache — i.e. it now survives a `SubstrateDynamicsEngine.tick()`-triggered
  `_hydrate_from_durable()` rebuild, closing PR #1205's disclosed gap.
- `substrate_pressure_signals()` on a node with `contributing_turn_ids` set produces an
  `AttentionSignalV1.evidence_refs` list containing both the node id and every contributing
  turn id; a node without the key produces the old (node-id-only) behavior unchanged.
- `reduce_attention_self_model(..., harness_closure_signal=None)` (every existing caller)
  reproduces today's narrative byte-for-byte; passing a real `harness_closure_signal` only
  changes `reason_narrative` in the `field_salience_only` branch, never `attention_reason`,
  never the `top_down_override`/`bottom_up_salience` branches.
- Live check (manual, `redis-cli -p 6380 GRAPH.QUERY orion_substrate ...` against
  `orion-athena-falkordb`): confirmed the running `orion-substrate-runtime` container
  (created 2026-07-18T22:06:37Z) already carries PR #1205's fixed-node-id code and
  `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES=true` live, but `node:substrate.harness_closure`
  does not yet exist as a node in `orion_substrate` (only pre-#1205 legacy
  `harness_closure:<uuid>` nodes are present) — no `surprise_unresolved=True`
  `HarnessPostTurnClosureV1` event has landed since that container's last restart. This is
  an honest `UNVERIFIED` for the live end-to-end durability claim until a real unresolved
  harness closure occurs post-deploy of *this* patch; the codec/reducer/broadcast wiring
  itself is fully covered by unit tests against the real production schemas.

## Recommended next patch

- After this patch deploys and a live `node:substrate.harness_closure` write with
  `contributing_turn_ids` occurs, re-run the manual `redis-cli GRAPH.QUERY` check (or a
  short one-off script) to confirm the value survives a subsequent dynamics tick in
  production, not just in unit tests.
- If `scripts/analysis/measure_ast_hot_reducer.py`'s Postgres replay is ever extended to
  cover FalkorDB-sourced inputs generally (not just for this one node), wiring
  `harness_closure_signal` into that replay becomes proportionate — tracked here as a real
  follow-up, not silently dropped.
