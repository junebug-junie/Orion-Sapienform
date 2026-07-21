# orion-mind

FastAPI service exposing `run_mind()` (`orion.mind.v1.MindRunRequestV1` -> `MindRunResultV1`). This README does not attempt to cover the whole service -- it documents one specific, verified piece: the `llm_surface_instability` metacog trigger, added here because `services/orion-equilibrium-service/README.md`'s trigger-taxonomy documentation flagged this as the one real `MetacogTriggerV1` producer living outside that service with no README coverage anywhere (2026-07-21).

## `llm_surface_instability` metacog trigger

`app/uncertainty_metacog.py`'s `maybe_publish_llm_surface_instability_trigger()` fires an *advisory* `trigger_kind=llm_surface_instability` metacog trigger when an LLM response's own logprob telemetry looks unstable -- not a factual-confidence signal, a language-surface-instability one. Gated by `MIND_LLM_UNCERTAINTY_METACOG_ENABLED` (default `false`).

Fires (`should_emit_llm_surface_instability()`) when, off the response's `llm_uncertainty` telemetry dict:
- `unstable_span_count >= 1` (reason: `unstable_span`), or
- `mean_top1_margin < 0.75` (reason: `low_mean_margin`), or
- `low_logprob_token_count / token_count_observed > 0.15` (reason: `high_low_logprob_ratio`)

Never fires if `llm_uncertainty.available` is falsy or `token_count_observed <= 0` (reason: `unavailable`/`no_tokens`). `pressure` on the trigger is `min(1.0, low_logprob_token_count / token_count_observed)`; `upstream` carries the full `llm_uncertainty` dict, the phase name, and the specific `instability_detail` reason string.

**Unlike every trigger documented in `orion-equilibrium-service`'s README, this one bypasses equilibrium's gate logic entirely** -- `_publish_metacog_trigger_async()` publishes `MetacogTriggerV1` directly onto `MIND_METACOG_TRIGGER_CHANNEL`, which defaults to `orion:equilibrium:metacog:trigger` (the same channel equilibrium-service's own `_publish_metacog_trigger()` writes to for `orion-cortex-orch` to consume). There is no confidence/level floor here the way `relational`/`telemetry_anomaly` have -- the three boolean/threshold conditions above are the only gate.

| Env | Default | Purpose |
|-----|---------|---------|
| `MIND_LLM_UNCERTAINTY_METACOG_ENABLED` | `false` | Master gate |
| `MIND_METACOG_TRIGGER_CHANNEL` | `orion:equilibrium:metacog:trigger` | Publish channel -- shared with equilibrium-service's own triggers, not a dedicated inbound channel |
| `ORION_BUS_URL` | `redis://redis:6379/0` | Bus connection (fresh `OrionBusAsync` per publish, one-shot connect/publish/close) |
