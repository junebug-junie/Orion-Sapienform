## Summary

- Phase 3 of the inner-state unification plan: `spark_embodiment_narrative` — the real hardware node most salient this tick, rendered into both metacog prompt templates alongside `spark_phi_narrative`.
- This is the one phase whose own stated gate requires deployment first ("Phase 2 merged AND deployed for a real traffic window"). Satisfied it directly: deployed Phases 0–2 in this same session, confirmed real `dominant_node` values live on the bus (`node:atlas`/`node:circe` alternating, matching the Phase 4 cross-check's 81.7%/18.3% split) *before* writing any Phase 3 code.
- Found and fixed a real scoping bug before it shipped — caught by writing the regression test first, not by review.
- Found and fixed a live production incident during deployment, unrelated to Phase 3's own code: a second, independent schema consumer (`orion-sql-writer`) needed two separate rebuilds to catch up with two separate schema changes.
- Deployed end-to-end and verified via direct Postgres query: real `dominant_node`/`dominant_node_reason` values now persisting in `spark_telemetry`.

## Outcome moved

Orion's metacognition prompts now receive a real, node-attributed embodiment cue — "Right now, athena is the most salient part of my body (node contract_pressure is elevated)" — sourced from a genuinely trained/computed signal chain (field state → attention salience → self-state composition → phi), not a placeholder.

## Current architecture

`SparkStateSnapshotV1` (not `PhiIntrinsicRewardV1`) is the schema `orion-cortex-exec`'s `executor.py` actually reads for prompt-building (via `spark_snap`, fetched over the state-service RPC). Phase 2 added `dominant_node`/`dominant_node_reason` to `PhiIntrinsicRewardV1`; this phase required threading the same values through the *other* schema.

## Architecture touched

`orion/schemas/telemetry/spark.py`, `services/orion-spark-introspector/app/worker.py`, `services/orion-cortex-exec/app/{executor,spark_narrative}.py`, `orion/cognition/prompts/log_orion_metacognition_{draft,enrich}.j2`. Live deploy also required rebuilding `orion-sql-writer` (twice — see incident below).

## Files changed

- `orion/schemas/telemetry/spark.py`: additive `SparkStateSnapshotV1.dominant_node`/`.dominant_node_reason`.
- `services/orion-spark-introspector/app/worker.py`: wires the same `_dominant_hardware_node()` result already computed for `PhiIntrinsicRewardV1` into the `SparkStateSnapshotV1` construction. Adds `dominant_node`/`dominant_node_reason` `Optional[str] = None` defaults near `encoder_tick_ok = False`, fixing a scoping bug (see Review findings).
- `services/orion-cortex-exec/app/spark_narrative.py`: `spark_embodiment_hint`/`spark_embodiment_narrative`, mirroring `spark_phi_hint`/`spark_phi_narrative` exactly. Honest on absence — `"none"`/an explicit no-node sentence, never a fabricated node name.
- `services/orion-cortex-exec/app/executor.py`: wired into both metacog prompt passes — `ctx["embodiment_hint"]`/`ctx["spark_embodiment_narrative"]`, the `_render_prompt` Jinja-undefined-safety defaults dict, and the `_METACOG_DRAFT_CTX_LEN_KEYS` prompt-size-logging tuple — matching every existing `phi_hint`/`spark_phi_narrative` touch point.
- `orion/cognition/prompts/log_orion_metacognition_{draft,enrich}.j2`: one new `EMBODIMENT` block each.
- `orion/self_state/inner_state_registry.py`, `orion/self_state/README.md`, `services/orion-spark-introspector/README.md`, `docs/superpowers/plans/2026-07-12-inner-state-unification-plan.md`: registry, docs, plan status updated.
- `services/orion-cortex-exec/tests/test_spark_narrative_embodiment.py` (new), `services/orion-spark-introspector/tests/test_phi_reward_emit.py`: 7 new tests total.

## Schema / bus / API changes

- Added: `SparkStateSnapshotV1.dominant_node`, `.dominant_node_reason` (additive, `Optional[str] = None`).
- Compatibility notes: `orion.normalizers.spark.normalize_spark_state_snapshot` is schema-driven (`set(SparkStateSnapshotV1.model_fields.keys())`), so it picked up the new fields automatically once `orion-sql-writer` was rebuilt against current code — no normalizer change needed.

## Env/config changes

None.

## Tests run

```text
PYTHONPATH=. :services/orion-spark-introspector pytest services/orion-spark-introspector/tests -q
140 passed, 1 skipped

PYTHONPATH=. :services/orion-cortex-exec pytest services/orion-cortex-exec/tests/test_spark_narrative_embodiment.py \
  services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py services/orion-cortex-exec/tests/test_metacog_publish_lane.py -q
32 passed

PYTHONPATH=. pytest tests/test_inner_state_registry_gate.py -q
8 passed

python scripts/check_inner_state_registry.py
inner_state_registry gate OK (9 entries checked)
```

Regression-confirmed: the `UnboundLocalError` fix was verified by temporarily reverting it and re-running `test_spark_snapshot_dominant_node_none_when_encoder_skipped` — reproduced the exact crash (`cannot access local variable 'dominant_node' where it is not associated with a value`), then re-applied the fix and confirmed green.

## Evals run

None applicable.

## Docker/build/smoke checks

**Deployed live on Athena.** Full sequence, in order:

```bash
# Phases 0/1/2 (prerequisite for this phase's own gate):
docker compose --env-file .env --env-file services/orion-self-state-runtime/.env \
  -f services/orion-self-state-runtime/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-spark-introspector/.env \
  -f services/orion-spark-introspector/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-sql-writer/.env \
  -f services/orion-sql-writer/docker-compose.yml up -d --build   # incident fix, see below

# Phase 3 itself:
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build   # all 4 sub-services
docker compose --env-file .env --env-file services/orion-spark-introspector/.env \
  -f services/orion-spark-introspector/docker-compose.yml up -d --build   # re-rebuild, schema changed again
docker compose --env-file .env --env-file services/orion-sql-writer/.env \
  -f services/orion-sql-writer/docker-compose.yml up -d --build   # re-rebuild, second schema change
```

Verified live, end to end:
- `redis-cli SUBSCRIBE orion:self:phi_reward` → real `dominant_node`/`dominant_node_reason` on the bus.
- `redis-cli SUBSCRIBE orion:spark:state:snapshot` → same fields present on `SparkStateSnapshotV1` messages.
- `SELECT metadata->'spark_state_snapshot'->>'dominant_node' FROM spark_telemetry ...` → `node:circe`, `reason: node pressure present`, confirming the full bus→Postgres path.

## Review findings fixed

- Finding: `dominant_node`/`dominant_node_reason` are assigned only inside the encoder-success `try` block, but the `SparkStateSnapshotV1` construction (added this phase) reads them unconditionally, later, on every tick — `UnboundLocalError` on any disabled/degraded/frozen/failed tick.
  - Fix: `None` defaults declared alongside the existing `encoder_tick_ok = False` pattern, before the conditional block.
  - Evidence: reverted the fix, re-ran the regression test, confirmed it reproduces the exact crash; re-applied, confirmed green. Independent code-review pass separately traced the full function control flow and confirmed no other unguarded read site exists.
- Finding (mid-deployment, not code review — a live incident): `orion-sql-writer` needed rebuilding **twice** — once for Phase 2's `PhiIntrinsicRewardV1.dominant_node` (fixing a live `extra_forbidden` `ValidationError` on every phi-reward write), once more for Phase 3's `SparkStateSnapshotV1.dominant_node` (silently dropping the field from `spark_telemetry`'s stored JSON with no error, since `extra="forbid"` doesn't fire on missing-vs-extra in that direction — a normalizer schema-driven filter, `orion.normalizers.spark.normalize_spark_state_snapshot`, just quietly excluded fields its running code didn't know about yet).
  - Lesson recorded in the plan doc: `extra="forbid"` schemas need every consumer identified and rebuilt together, not just the primary producer/consumer pair — and re-checked after *each* subsequent schema change to the same object, not just once.

## Restart required

Already done live — see Docker/build/smoke checks above. No further restart needed.

## Risks / concerns

- Severity: low
- Concern: the embodiment narrative is now live in real chat/metacog prompts. Its actual effect on Orion's self-report tone/content hasn't been observed over a real chat session yet — only verified for correctness at the data-plumbing level (right node named, no crashes, no fabrication).
- Mitigation: the narrative text is explicitly framed as internal guidance ("do NOT copy verbatim into output" — matching the existing `spark_phi_narrative` convention in both templates), same safety margin already relied on for the phi narrative.
- Severity: low
- Concern: two live incidents this session (both `orion-sql-writer` schema-drift issues) point at a repeatable gap — no automated check currently catches "a schema this service consumes changed and this service wasn't rebuilt."
- Mitigation: named explicitly, not silently patched around. A real fix (e.g., extending `scripts/check_inner_state_registry.py` or a separate consumer-staleness check) is a natural future addition to this initiative's registry/gate work, not built here — out of scope for this phase.

## PR link

Branch pushed: `docs/attention-provenance-crosscheck-phase4` (continued in-session from Phase 4; contains Phase 4's docs commit plus this Phase 3 implementation commit)
