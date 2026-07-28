# Landing Pad retirement

**Status**: Implemented. Branch `fix/kill-landing-pad`.

**Trigger**: direct follow-on from the same session's cognition-trace audit
(`docs/superpowers/specs/2026-07-28-cognition-trace-signal-gateway-consumer-audit.md`) and
`orion-spark-introspector`'s retirement (PR #1413). That earlier audit found `orion-landing-pad`
was already mostly write-only (3 of its 4 advertised capabilities had no real consumer); this
session confirmed its one remaining live data source (`orion-spark-introspector`'s
`spark.state.snapshot.v1` publish) went away with that service's own retirement, with nothing
replacing it.

## Why kill it, not fix it

- **Ingest**: `PAD_INPUT_ALLOWLIST_PATTERNS` (`orion:telemetry:*,orion:cortex:*,orion:spark:*`)
  only ever had one live producer feeding real data through it — spark-introspector's
  `spark.state.snapshot.v1`. That's gone. The other two reducers (`metric`, `biometrics`) were
  already confirmed dead in the earlier audit: `metric` had zero real producers anywhere in the
  repo, and `biometrics` was unreachable due to an allowlist/channel mismatch that predates this
  change.
- **Pulse trigger** (`orion:pad:signal` → `orion-equilibrium-service`'s `metacog_pad_pulse_threshold`):
  mathematically unreachable given the live reducer set (max observed salience 0.55 against an
  0.8 threshold), confirmed in the earlier audit. Dead regardless of the ingest question.
- **RPC query skill** (`LandingPadMetricsSnapshotVerb`/`LandingPadLastEventsVerb` in
  `orion-cortex-exec`, `MetacogContextService`'s frame/stats fetch in `executor.py`): the one part
  of the service with a genuinely live caller — but a skill that queries a permanently-empty
  buffer isn't worth keeping a whole service running for.

## What changed

- Deleted `services/orion-landing-pad/` wholesale.
- Deleted `orion/schemas/pad/` (`PadEventV1`, `StateFrameV1`, `PadRpcRequestV1`,
  `PadRpcResponseV1`) and their `orion/schemas/registry.py` registrations.
- Deleted the six `orion:pad:*` channels plus `orion:exec:result:PadRpc:*` from
  `orion/bus/channels.yaml`; removed `orion-landing-pad` from `orion:telemetry:biometrics` /
  `orion:biometrics:sample` / `orion:biometrics:summary`'s `consumer_services` (it was declared
  there but never actually subscribed — same catalog-drift shape found for `orion:cognition:trace`
  in the earlier audit).
- `orion-cortex-exec`: removed `LandingPadMetricsSnapshotVerb`, `LandingPadLastEventsVerb`, the
  `_pad_rpc_request` helper, and the `channel_pad_rpc_request`/`channel_pad_rpc_reply_prefix`
  settings. Removed the skill-id routing entries in `actions_skill_registry.py`,
  `orion/cognition/skills_manifest.py`, `capability_bridge.py`, and the phrase triggers in
  `supervisor.py`.
- `orion-cortex-exec/app/executor.py`: removed `MetacogContextService`'s two pad RPC calls
  (`get_latest_frame`, `get_stats`), the `pad_summary`/`pad_short` computation, the
  `"Landing Pad: ..."` line in the metacog context summary, and the now-dead
  `_format_pad_frame_summary`/`_format_pad_stats_summary` helpers. Removed `pad_frame` /
  `pad_frame_json` / `pad_stats` / `pad_stats_json` from `orion/schemas/context_provenance.py`'s
  `live_runtime_projection` registry — keeping them registered would have been exactly the
  false-liveness claim that registry exists to prevent (per its own docstring), since these ctx
  keys can no longer ever be populated by a live computation.
- `orion-equilibrium-service`: removed the `orion:pad:signal` subscription and the whole
  `trigger_kind="pulse"` block in `service.py`, plus `channel_pad_signal` and
  `metacog_pad_pulse_threshold` from settings/`.env_example`/`docker-compose.yml`.
- `orion-hub`: removed `LANDING_PAD_URL`/`LANDING_PAD_TIMEOUT_SEC` settings, the dead
  `_fetch_landing_pad` helper (never called even before this change) and its `/api/debug/build`
  reference, the three landing-pad phrase mappings in `skill_runner_catalogue.py`, and the
  matching `<optgroup>` in `templates/index.html` (with the rest of the dropdown renumbered).
- `orion-notify-digest`: removed the already-deprecated `LANDING_PAD_URL` field and its
  `warn_deprecated_landing_pad_url` validator (pre-existing dead weight independent of this
  change — it warned operators to stop using this exact key back when `TOPIC_FOUNDRY_URL` was
  introduced).
- `orion-cortex-orch`: removed the two landing-pad skill IDs from `_DIRECT_VERB_TRIGGERS`.
- `config/mesh_remediation_roster.yaml`: removed landing-pad's remediation entry.
- Deleted `tests/test_landing_pad_spark_snapshot_reducer.py`,
  `scripts/smoke_bus_publish_pad.py`, `docs/landing_pad.md`,
  `docs/preflight-landing-pad-metrics-inventory.md`,
  `services/orion-cortex-exec/tests/test_skill_verbs.py`'s landing-pad test (plus its
  now-orphaned `_FakeBus`/`_Codec` fixtures), `orion/signals/adapters/tests`-equivalent verb
  test, and `tests/test_metacog_context_summary.py` (existed solely to test the deleted
  `_format_pad_*` helpers).
- Updated `docs/contracts.md`, `docs/operator_skill_prompt_catalogue.md`, root `README.md`,
  `services/orion-hub/README.md` to describe current reality instead of the retired design.

## Non-goals / left alone

- `services/orion-mesh-guardian/tests/*.py` and `services/orion-bus-mirror/README.md` /
  `services/orion-bus-tap/README.md` still use `landing-pad`/`orion:pad:*` as an arbitrary
  example service/pattern name in otherwise-generic test fixtures and doc examples — cosmetic,
  not functionally dependent on the real service, not touched.
- `services/orion-bus/app/bus_observer.py`'s `_ENVELOPE_FIELD_CANDIDATES` comment still cites
  landing-pad's old `redis_store.py` as the historical reason the `"data"` field candidate
  exists — left as provenance documentation for a still-active fallback list, not touched.

## Acceptance checks

- `orion/bus/channels.yaml` parses (253 channels) and `orion.schemas.registry` imports cleanly.
- `orion-cortex-exec`, `orion-equilibrium-service`, `orion-hub`, `orion-mesh-guardian` test
  suites pass (module-by-module for cortex-exec, whose test files use a shared verb-registry
  singleton and must run standalone rather than as one directory — pre-existing collection-order
  quirk, confirmed unrelated to this change).
- `orion-mesh-guardian`'s roster loader confirms `landing-pad` is absent from the live roster and
  the remaining 7 services still load correctly.
- Two test files needed real updates beyond deletion: `orion/schemas/tests/test_context_provenance.py`
  (dropped 4 keys from the expected key list) and `services/orion-hub/tests/test_skill_runner_catalogue.py`
  (catalogue count 23 → 20).
- Four pre-existing, unrelated test failures confirmed present with this diff removed (via
  `git stash`) before touching anything: `test_skill_verbs.py::test_github_recent_prs_includes_truncated_body`
  (date-relative GitHub PR fixture), `test_context_provenance.py::test_static_ctx_assignments_covered`
  (`llm_serving_node`/`trigger_upstream_json` already unclassified), and
  `orion-equilibrium-service/tests/test_bus_synaptic_poll_e2e.py`'s
  `test_poll_above_threshold_triggers`/`test_trigger_carries_edge_count_and_context` (unrelated
  to `transport_metacog_gate.py`, which this diff never touches) — none caused by, or fixed by,
  this change.
