# PR report: vision root cause + recall transport render gate

## Summary

- Root-caused and fixed why `node:substrate.vision.prediction_error` read exactly 1.0 for 40,839 ticks: `orion-vision-frame-router` had no restart policy and never came back after a clean shutdown 30h earlier. Fixed, redeployed, live-verified.
- Corrected the recall render-gate spec: the threshold is 0.15 (matching `orion-equilibrium-service`'s already-live trigger), not the 0.25 the spec originally proposed.
- Found the spec's named file (`orion-cortex-orch`'s `conversation_front.py`) is dead code, never called. Rebuilt the resolver in the real live path, `orion-mind`'s `evidence.py`.
- Adversarial pass caught and fixed a real URN-collision bug in `orion/metrics/lineage.py` before it shipped.
- Code review (5 finder agents) caught 8 more real bugs across the implementation; all fixed in the same branch.

## Outcome moved

- Orion's own camera-staleness signal reads real values again instead of a permanently pinned 1.0.
- When the bus-transport signal crosses the same threshold that already triggers Orion's metacognitive reflection on transport, his own recall now actually says something about it — instead of the equilibrium trigger firing while recall stays silent (the original bug this handoff was opened to fix).

## Current architecture

Recall's `falkor_bus_synaptic_adapter.py` queried FalkorDB for per-edge bus anomalies and wrote its own English sentence per edge, filtered by a recency window. Nothing gated whether Orion heard about transport state on the mesh-wide aggregate — recall just handed over whatever prose the adapter wrote, unconditionally, for every anomalous edge found. `node:substrate.vision`'s staleness channel was a dead read for 30 hours because its upstream frame router silently never restarted.

## Architecture touched

- `services/orion-vision-frame-router` — restart policy, unbuffered logging.
- `services/orion-recall/app/storage/falkor_bus_synaptic_adapter.py` — stopped writing prose for publish-gap fragments, dropped that query's recency filter (age now informational, not filtering).
- `services/orion-recall/app/fusion.py` — exempted structured-signal fragments from the low-info-social content filter.
- `services/orion-mind` — new Postgres dependency, new `config/` mount, new resolver module, wired into the live evidence-pack builder.
- `orion/field/channel_glossary.py`, `config/field/field_channel_glossary.v1.yaml` — optional node-qualified glossary entries.
- `orion/metrics/lineage.py` — URN collision fix for node-qualified entries.
- `docs/superpowers/specs/2026-09-03-recall-signal-rendering-design.md` — corrected in place, both mistakes recorded with why.

## Files changed

- `services/orion-vision-frame-router/docker-compose.yml` — `restart: unless-stopped`
- `services/orion-vision-frame-router/Dockerfile` — `PYTHONUNBUFFERED=1` (its own logs were silently buffered, which is how the redeploy itself was hard to verify)
- `docs/superpowers/specs/2026-09-03-recall-signal-rendering-design.md` — both corrections recorded before the sections they change
- `config/field/field_channel_glossary.v1.yaml` — one node-qualified `prediction_error` entry (`node:substrate.bus_synaptic` only — the vision one has no consumer in this patch, and the repo's own orphan-metric CI gate correctly refused it when I tried adding it anyway)
- `orion/field/channel_glossary.py` — `resolve_channel_entry()`, prefers a node-qualified entry, falls back to the bare one
- `orion/metrics/lineage.py` — qualifies `name`/URN for node-scoped glossary entries so they can't collide with the bare entry
- `config/metrics/metric_definitions.lock.json` — re-locked, 1 new metric (`node:substrate.bus_synaptic.prediction_error`)
- `services/orion-recall/app/storage/falkor_bus_synaptic_adapter.py` — publish-gap fragments carry `text=""`, no recency filter on that query, `meta` unchanged; causal-latency fragments untouched (non-goal)
- `services/orion-recall/app/fusion.py` — `meta.signal_kind`-carrying candidates skip the low-info-social filter (review finding — see below)
- `services/orion-mind/app/recall_signal_resolver.py` — new module: gate, Postgres series read, sentence construction
- `services/orion-mind/app/evidence.py` — resolver branch in `build_evidence_pack()`
- `services/orion-mind/app/engine.py` — threads the DSN/threshold from settings into the call
- `services/orion-mind/app/settings.py`, `.env_example`, `docker-compose.yml`, `requirements.txt` — new DSN + threshold config, `psycopg2-binary`, `/repo` read-only mount
- Tests alongside every file above

## Schema / bus / API changes

- Added: `metric://field_channel/orion-field-digester/node:substrate.bus_synaptic.prediction_error` (glossary entry + lineage URN)
- Removed: none
- Renamed: none
- Behavior changed: recall's publish-gap bus-synaptic fragments no longer carry English prose (empty `text`, structured `meta` only); the rendered sentence for the whole bus now comes from `orion-mind`'s resolver instead
- Compatibility notes: a fragment with no `signal_kind` (every fragment shape before this patch) renders byte-identically to before, both in the old dead code path and the real `evidence.py` path (test-covered in both)

## Env/config changes

- Added keys: `RECALL_TRANSPORT_PG_DSN`, `RECALL_TRANSPORT_RENDER_GATE_THRESHOLD` (both on `services/orion-mind`)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes (`services/orion-mind/.env_example`)
- Local `.env` synced: yes — with a real gap found and fixed along the way: `scripts/sync_local_env_from_example.py` writes to the **primary checkout's** `.env`, not the worktree's own (its own documented design — worktrees don't have a canonical `.env` of their own). A worktree's local `.env` still needs a manual copy from the primary checkout after running the sync script. Missed this once during implementation (code review caught it: the container was live-verified with an empty DSN); fixed and redeployed.
- Skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=<worktree> .venv/bin/python -m pytest tests/test_metric_lineage_gate.py tests/test_metric_definition_drift.py tests/test_metric_lineage.py tests/test_field_channel_glossary.py tests/test_metric_liveness.py tests/test_metric_generic_consumers.py -q
218 passed

cd services/orion-recall && PYTHONPATH=<worktree>/services/orion-recall .venv/bin/python -m pytest tests/ -q
274 passed, 4 failed — same 4 failures confirmed pre-existing on unmodified main (test_brain_recall_profile.py, test_process_recall_active_turn_exclusion.py, test_recall_policy_harness.py, test_recall_vector_amputation.py), unrelated to this patch

cd services/orion-mind && PYTHONPATH=<worktree>:<worktree>/services/orion-mind .venv/bin/python -m pytest tests/ -q
80 passed, 5 failed — same 5 failures confirmed pre-existing on unmodified main (test_http_contract.py x3, test_projection_starvation.py x2), unrelated to this patch
```

No eval harness exists for `orion-recall`/`orion-mind`/`orion-vision-frame-router` beyond their pytest suites; not adding one in this patch.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-vision-frame-router up -d --build
scripts/safe_docker_build.sh orion-recall up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
```

Live verification, in order:
1. `node:substrate.vision` staleness: `1.000` → `0.000` within one tick after the frame-router redeploy (orion-substrate-runtime logs).
2. `orion-vision-host` resumed processing `retina_fast` tasks immediately after the frame-router redeploy (its own logs).
3. Inside the deployed `orion-mind` container (`docker exec orion-mind python -c ...`, real settings, real Postgres, real mounted config): `fetch_bus_synaptic_prediction_error_series()` returned a real 60-row series; `render_bus_synaptic_digest_line()` produced `"Transport: 5% of live bus channels running anomalous, against a 0.25 watch threshold and 0.50 summarize. Loudest right now: orion:vision:edge:health from vision-edge. Trend in substrate_field_state.field_json -> node_vectors -> 'node:substrate.bus_synaptic' -> prediction_error."` — confirms the Postgres read, the `db_readonly` helper, the `/repo` config mount, and the glossary resolution all work together end to end.
4. Confirmed and fixed a real deploy gap: the first live-verification pass ran with `RECALL_TRANSPORT_PG_DSN` empty in the deployed container (local `.env` sync gap above) — re-verified after the fix with the real DSN present.

Not done: an actual live kill of `orion-bus-mirror` to observe the "not writing" state under a real outage (spec's own suggested live check). Covered instead by unit tests simulating an empty series and a partial-window degenerate-zero series (the two failure shapes that state exists to catch), plus the direct trace confirming `_bus_synaptic_tick` really does keep writing degenerate `0.0` forever once its own edge query ages out — not deliberately disrupting a live production service without asking first.

## Review findings fixed

- Finding: `fusion.py`'s `low_info_social` filter dropped every `publish_gap_zscore` fragment unconditionally on any substantive query (empty `text` → always "low info"), silently making the whole render-gate feature inert for the majority of real chat turns.
  - Fix: exempt candidates carrying `meta.signal_kind` from that filter.
  - Evidence: `test_publish_gap_fragment_survives_fuse_candidates_on_a_substantive_query` (new; the existing adjacent test never set `substantive_query=True`, which is why this was missed originally).
- Finding: `SET LOCAL statement_timeout` issued after `conn.set_session(autocommit=True)` is silently a no-op.
  - Fix: switched to `orion.db_readonly.open_readonly_connection()`, the repo's canonical helper (session-level `SET statement_timeout`, plus read-only-session enforcement).
  - Evidence: `test_series_fetch_uses_the_canonical_readonly_helper_with_bounded_timeouts`.
- Finding: a new blocking Postgres call runs synchronously inside the async request handler with no thread-pool offload.
  - Considered, not changed: `orion-mind`'s entire mind-run pipeline (`run_mind`/`run_mind_llm_synthesis`, including LLM synthesis with wall time up to `MIND_WALL_MS_DEFAULT`=180s) is already fully synchronous end to end, called un-awaited from the async handler. This call's bounded ~1.3s worst case (1s connect + 300ms statement timeout) is a small addition to an already much larger existing blocking pattern, not a new architectural class of problem. A real fix would mean converting the whole pipeline to async — out of scope for this patch.
- Finding: `classify_channel_series()` only returns `"dead"` when every value in the window is subnormal; an outage starting partway through the window reads as `quiet`/`live` with `latest=0.0`, silently failing the render gate instead of triggering the degenerate state.
  - Fix: also check `latest` directly for subnormal against `SUBNORMAL_CUTOFF`.
  - Evidence: `test_render_degenerate_when_only_the_tail_is_zero`.
- Finding: removing the publish-gap query's recency filter let a permanently frozen edge win "loudest right now" forever.
  - Fix: added a local 300s freshness check for that display detail only, independent of the render gate/liveness logic.
  - Evidence: `test_render_does_not_attribute_loudest_to_a_frozen_edge`, `test_render_skips_a_frozen_edge_and_names_the_next_fresh_one`.
- Finding: the liveness check was skipped entirely whenever recall's Falkor fetch returned no fragments — indistinguishable from a genuine Falkor/bus-mirror outage.
  - Fix: gate is now on the Postgres DSN being configured, not on the fragment list's emptiness — `substrate_field_state` is written by a different service's own Falkor query, so its freshness is independent of whether recall's own Falkor connection is healthy right now.
  - Evidence: `test_render_checks_postgres_even_with_no_handled_fragments_when_dsn_is_set`.
- Finding: stale docstring pointers to `conversation_front.py` (the dead-code location the spec originally named).
  - Fix: corrected in `falkor_bus_synaptic_adapter.py`'s module docstring and inline comment.
- Finding: local `.env` for `orion-mind` was missing the two new keys, so the first live-verification pass silently ran with an empty DSN.
  - Fix: re-synced from the primary checkout (see Env/config changes above for the real gap this exposed in the sync script's design).
  - Evidence: redeployed, `docker exec orion-mind env` confirms both keys now present; re-ran the live smoke check with real data.
- Finding: an earlier draft added a `fetch_bus_synaptic_graph_liveness()` Falkor query, unused by the final design.
  - Fix: found and reverted this myself before code review, once I confirmed `classify_channel_series()`'s existing `dead` verdict (plus the `latest`-check fix above) already covers the failure mode it was meant for — avoided shipping an orphaned producer.

## Restart required

Already redeployed and live-verified during this session:

```bash
scripts/safe_docker_build.sh orion-vision-frame-router up -d --build
scripts/safe_docker_build.sh orion-recall up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
```

These three services are currently running this branch's build on the shared host. If you'd rather not run unmerged code live until this PR is reviewed, redeploy from `main` after merge (no special `ORION_HOST_REPO_ROOT` override needed post-merge — the primary checkout will have the glossary/config changes once merged):

```bash
scripts/safe_docker_build.sh orion-vision-frame-router up -d --build
scripts/safe_docker_build.sh orion-recall up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
```

## Risks / concerns

- Severity: low. Concern: `orion-mind` now has its first Postgres dependency and its first blocking network call inside the request path. Mitigation: bounded (~1.3s worst case), fail-open everywhere, matches the existing pipeline's much larger synchronous LLM-call pattern.
- Severity: low. Concern: `services/orion-hub/scripts/substrate_lattice_routes.py` already has a pre-existing, unrelated drift bug (its hand-maintained mirror still uses the pre-rename key `stream_backlog_pressure` instead of `bus_synaptic_pressure`) — found during the blast-radius check for the threshold decision, not fixed here (would widen scope past the handoff's three items). Flagging for a follow-up.
- Severity: low. Concern: `node:substrate.vision`'s node-qualified glossary entry was deliberately NOT added in this patch (no consumer yet) — the two-domains-share-a-channel-name collision item 1 surfaced still applies to vision's own rendering, whenever something needs to disambiguate it. Follow-up, not blocking.
- Severity: none, informational: the vision-frame-router container's own name (`orion-orion-athena-vision-frame-router`) has a doubled `orion-` prefix from `container_name: orion-${PROJECT:-dev}-vision-frame-router` where `PROJECT` is itself `orion-athena`. Cosmetic, unrelated to the outage, not touched.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/recall-render-gate-and-vision-router
