# PR Report: CoLA-Derived Novelty Signal (`feat/cola-novelty-signal`)

## Summary

- Adds `POST /v1/understand` to `orion-llama-cola-host`: a single deterministic forward pass through CoLA's `bc_mode` (Inverse Dynamics) branch, returning the pooled pre-argmax action-codebook distribution for finished text.
- Patches `intention.py`'s `IntentionModel_v1.forward()` `bc_mode` return from a 2-tuple to a 3-tuple to expose that distribution (previously only the collapsed argmax index was returned).
- `orion-spark-introspector` calls the new endpoint directly per chat turn (bypassing `orion-llm-gateway`'s shared routing table on purpose), scores novelty as cosine distance from a bounded, LRU-evicted rolling per-session reference, and publishes it as `SparkSignalV1.novelty_delta` over the existing `orion:spark:signal` bus — the same wire `orion-cortex-exec` already reads for metacognition.
- Adds `signal_type="language"` to `SparkSignalV1` (additive) instead of overloading an existing value that would have mislabeled the signal downstream.
- Ships disabled via `COLA_UNDERSTAND_ENABLE=false` and fails open on any error.

## Outcome moved

`orion-cortex-exec`'s metacognitive context can now receive a novelty signal derived from CoLA's latent-action understanding of what was actually said in a turn, instead of only the self-state-substrate-derived novelty baseline (which reflects biometric/attention pressure, not language content).

## Current architecture

`orion-llama-cola-host` existed but was unreachable from the live chat path (routing table maps `chat`/`agent`/`metacog`/`quick` to `llamacpp`; the one code path that would default to `llama-cola` is never called) and only exposed a stochastic, generation-time policy-sampling signal (`action_indices`, sampled at `tau=2.0`) via `/v1/chat/completions` — meant for output diversity, not for scoring the meaning of a finished reply. `orion-spark-introspector` had no producer of language-derived novelty at all; its only novelty source was `_phi_from_self_state`, derived from biometric/attention pressure dimensions.

## Architecture touched

- `services/orion-llama-cola-host`: new endpoint, model return-signature change (only `IntentionModel_v1`, the currently-configured variant — other unused variants in `intention.py` untouched).
- `services/orion-spark-introspector`: new HTTP client call, novelty scoring, signal publish, wired into the existing `handle_semantic_upsert` chat-turn handler.
- `orion/schemas/telemetry/spark_signal.py`: additive schema change (new `signal_type` literal).
- No changes to `orion-llm-gateway`, `SelfStateV1`, or the `SignalMapper`/`OrionTissue` grid pipeline (left as-is, out of scope).

## Files changed

- `services/orion-llama-cola-host/intention.py`: `bc_mode` branch in `IntentionModel_v1.forward()` now also returns the pre-argmax action-probability distribution.
- `services/orion-llama-cola-host/app/main.py`: new `POST /v1/understand` endpoint + request/response schemas.
- `services/orion-llama-cola-host/README.md`: documents the new endpoint.
- `services/orion-llama-cola-host/tests/`: new — model-level tests with a tiny randomly-initialized CoLA model (real torch execution, not mocked) proving the 3-tuple return, valid softmax distribution, determinism, and the endpoint's pooling logic.
- `services/orion-spark-introspector/app/worker.py`: new CoLA-understand HTTP client, LRU-bounded rolling per-session novelty history, `SparkSignalV1` publish, wiring into `handle_semantic_upsert`; extracted `_publish_spark_signal()` shared with the existing turn-effect-alert path.
- `services/orion-spark-introspector/app/settings.py`: new `COLA_*` settings (all opt-in, disabled by default).
- `services/orion-spark-introspector/.env_example`: documents the new keys.
- `services/orion-spark-introspector/README.md`: new §6.8 documenting the feature, data flow, settings, and rollback.
- `services/orion-spark-introspector/tests/test_cola_novelty_signal.py`: new — unit, fail-open, LRU-eviction, dimension-mismatch, and `handle_semantic_upsert` wiring tests, plus an in-process trace proving a published signal reaches the same `phi_stats` merge `orion-cortex-exec` reads.
- `orion/schemas/telemetry/spark_signal.py`: added `"language"` to `signal_type`'s literal set (additive; verified no exhaustive-match consumer exists elsewhere).

## Schema / bus / API changes

- Added: `POST /v1/understand` on `orion-llama-cola-host`.
- Added: `SparkSignalV1.signal_type` literal `"language"`.
- Behavior changed: `IntentionModel_v1.forward(bc_mode=True)` now returns a 3-tuple instead of 2-tuple. Verified this is the only production call site in the repo (no other caller assumes the old arity); other model variants (`v1p`, `v1a`, base) in the same file were left untouched since they're not the currently-configured model (`config.v == 1`).
- Compatibility notes: both additions are backward compatible; no existing consumer breaks.

## Env/config changes

- Added keys (all in `services/orion-spark-introspector/.env_example`): `COLA_UNDERSTAND_ENABLE` (default `false`), `COLA_UNDERSTAND_URL`, `COLA_UNDERSTAND_TIMEOUT_SEC`, `COLA_NOVELTY_WINDOW`, `COLA_NOVELTY_MAX_SESSIONS`, `COLA_NOVELTY_GAIN`, `COLA_NOVELTY_SIGNAL_TTL_MS`.
- `orion-llama-cola-host` needed no new env keys.
- Local `.env` synced manually (main checkout's real `services/orion-spark-introspector/.env`, since the default `sync_local_env_from_example.py` prefix allowlist doesn't cover new `COLA_*` keys) — confirmed `git check-ignore` still holds, nothing tracked.
- No skipped keys requiring operator action beyond flipping `COLA_UNDERSTAND_ENABLE=true` once `orion-llama-cola-host` is confirmed running and reachable.

## Tests run

```text
services/orion-spark-introspector: 61 passed, 1 skipped (pytest tests/ -q)
services/orion-llama-cola-host:     4 passed (pytest tests/ -q)
repo-root tests/test_spark_metrics_v2.py, test_spark_contract_gate.py:
  4 passed, 3 pre-existing failures (confirmed via git stash to reproduce
  identically on this branch's parent commit -- unrelated to this change,
  a stale-shape issue in orion/spark/orion_tissue.py's snapshot loading)
```

Run in an isolated venv since this sandbox had no dependencies preinstalled; installed each service's own `requirements.txt` plus CPU-only `torch`/`transformers==4.44.2` to genuinely execute `intention.py`'s patched model code with a tiny randomly-initialized CoLA-shaped model, rather than only AST-parsing it.

## Evals run

No eval harness exists for either touched service. Flagging as a gap: there's no automated check that the CoLA IDM branch's distribution is semantically meaningful (i.e. that paraphrases cluster and unrelated turns diverge) against the real 10B checkpoint — this environment has no GPU/model weights to run that against. Follow-up recommended before fully trusting this signal in metacognition (see Risks below).

## Docker/build/smoke checks

```text
docker compose --env-file .env --env-file services/orion-spark-introspector/.env \
  -f services/orion-spark-introspector/docker-compose.yml config   -> validates, new COLA_* keys pass through via env_file
docker compose --env-file .env --env-file services/orion-llama-cola-host/.env \
  -f services/orion-llama-cola-host/docker-compose.yml config      -> validates, confirms host binds :8005
```

No live container smoke: `orion-llama-cola-host` is not part of default deploy tooling (excluded from `mesh-utilities` auto-rebuild lists) and wasn't brought up in this environment. `COLA_UNDERSTAND_ENABLE=false` by default specifically so this ships without requiring that host to be live.

## Review findings fixed

- Finding: fail-open contract gap — dimension mismatch (cola-host redeployed with different `num_code`) would raise uncaught inside the fire-and-forget scoring task instead of the documented "logged and skipped" behavior.
  - Fix: explicit shape check resets the stale per-session reference before scoring; broader try/except as backstop.
  - Evidence: new test `test_score_cola_novelty_fails_open_on_dimension_mismatch`.
- Finding: fire-and-forget `asyncio.create_task` result wasn't retained, exposed to asyncio's weak-reference GC risk.
  - Fix: task stored in a module-level set with a done-callback to discard.
  - Evidence: new test `test_handle_semantic_upsert_schedules_cola_novelty_scoring` awaits from that set directly.
- Finding: session eviction was insertion-order FIFO, not LRU — an idle new session could starve out an actively-used older one.
  - Fix: switched to `OrderedDict` + `move_to_end` on every touch.
  - Evidence: new test `test_novelty_remember_evicts_least_recently_used_session`.
- Finding: `signal_type="recall"` gets embedded verbatim into `OrionSignalV1.summary` by `orion/signals/adapters/spark.py` for downstream consumers — a real semantic mislabel once enabled, not cosmetic.
  - Fix: added `signal_type="language"` to the schema instead.
  - Evidence: `test_publish_cola_novelty_signal_reaches_phi_stats_via_signal_bus` asserts `env.payload["signal_type"] == "language"`.
- Finding: new `httpx.AsyncClient()` opened per chat turn instead of pooled, on a hot path.
  - Fix: single lazily-initialized module-level client.
- Finding: signal-envelope publish logic duplicated between the new novelty path and the existing turn-effect-alert path.
  - Fix: extracted `_publish_spark_signal()`, both call sites now share it.
- Finding: no test exercised the actual `handle_semantic_upsert` wiring (only the helper functions in isolation).
  - Fix: added `test_handle_semantic_upsert_schedules_cola_novelty_scoring` / `..._skips_..._when_disabled`.
- Not fixed (documented limitation): no live smoke against the real loaded 10B CoLA model proving the IDM branch's distribution is semantically non-degenerate — no GPU/model weights available in this environment. See Risks.

## Restart required

```bash
# Only needed once COLA_UNDERSTAND_ENABLE is flipped to true:
docker compose -f services/orion-spark-introspector/docker-compose.yml restart
# orion-llama-cola-host would also need to actually be brought up and reachable
# at COLA_UNDERSTAND_URL (currently not part of default deploy tooling).
```

If left at the shipped default (`COLA_UNDERSTAND_ENABLE=false`), no restart is required — behavior is unchanged until explicitly enabled.

## Risks / concerns

- Severity: Medium
  - Concern: The CoLA model card's claim that its 64-code action distribution represents "distinct semantic meanings of language" is the paper authors' own framing, checked against their own benchmark — not independently validated against Orion's data. Enabling this feeds an unvalidated signal into metacognitive context.
  - Mitigation: ships disabled by default. Recommend a cheap validation pass (paraphrase pairs vs. unrelated pairs through `/v1/understand`, checking the pooled distributions actually cluster paraphrases together) before flipping `COLA_UNDERSTAND_ENABLE=true` in production.
- Severity: Low
  - Concern: `orion-llama-cola-host` is excluded from `mesh-utilities` auto-rebuild tooling and not part of default deploy compose sets; bringing it up for the first time in months may surface the model-loading issues already documented in its own README (shard/index mismatch).
  - Mitigation: documented in that README; the feature fails open regardless if the host can't load.
- Severity: Low (operational note, not a code risk)
  - Concern: running `docker compose ... config` against the real `.env` during this work printed a live-looking `HF_TOKEN` value in plaintext into the agent session transcript.
  - Mitigation: none applied in this PR (out of scope) — flagging so the operator can decide whether to rotate it.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/cola-novelty-signal
