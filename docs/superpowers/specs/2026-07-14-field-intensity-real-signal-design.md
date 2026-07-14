# Design spec: replace `field_intensity`'s theater aggregation with the already-live, already-honest execution-load signal

## Arsonist summary

`SelfStateV1.dimensions.field_intensity` is theater end to end, confirmed live: permanently pinned at exactly `1.0` across 42,478 samples over a 7-day window, zero variance. Root cause traced through every hop:

1. `field_intensity_score(overall_salience, recent_perturbation_saturation)` (`orion/self_state/scoring.py:150-155`) — `clamp01(0.6*overall_salience + 0.4*perturbation)`.
2. `overall_salience` = `max(t.salience_score for t in capped)` (`orion/attention/field_attention/builder.py:52`) — a raw max over per-target scores that genuinely vary, but since some target is nearly always near its own ceiling, the aggregate saturates almost every tick.
3. The per-target scores themselves are downstream of `node_vectors["cpu_pressure"]`/`["gpu_pressure"]`, which are **not independent hardware readings** — both are set to the exact same value, `float(atom.salience)` (`orion/substrate/biometrics_loop/grammar_extract.py:114,119`), an LLM's salience judgment about a telemetry summary, duplicated onto two differently-named channels. Confirmed live: identical to 16 decimal places on real nodes right now.

This is not a new discovery in the sense of "nobody knew" — `services/orion-spark-introspector/app/inner_state.py`'s phi/seed-v4 encoder **already excludes `field_intensity` from its trainable feature set** (`ENCODER_EXCLUDED_FELT`) and **already excludes `uncertainty`** (`DROPPED_DIMENSIONS`, also confirmed dead at exactly `0.0` live). Whoever built that pipeline reached the same verdict and routed around the problem instead of fixing it at the source. This spec fixes it at the source, reusing the same real signal that pipeline already trusts.

## Current architecture

- `field_intensity` feeds `weighted_overall_intensity()` (`orion/self_state/scoring.py`) — the system-wide condition classification (`quiet`/`steady`/`loaded`/`strained`/`unstable`).
- `field_intensity` feeds every proposal template's `match_score` via `orion/proposals/scoring.py`'s `template_match_score()` — confirmed live today (same investigation session) that this is the *dominant* term in `priority_score = base_priority + 0.4*match_score + 0.2*urgency + 0.1*confidence`, and directly explains why the newly-shipped `inspect_attended_target` proposal template (which also depends on `field_intensity`) never won a real dispatch tick despite correct binding resolution and 100% policy approval.
- A real, live, already-productionized alternative already exists: `services/orion-spark-introspector/app/inner_state.py`'s `execution_load`/`reasoning_load` computation (lines 277-313):
  - `execution_load_raw = round(math.log1p(float(completion_tokens_sum)), 4)`, source-labeled `"reasoning_activity.completion_tokens_sum"`, with an honest fallback to `log1p(step_count)` (`"execution_trajectory.step_count_fallback"`) and finally an explicit `0.0` / `"none"` — never a silent fake floor.
  - `reasoning_load_raw` mirrors this for `thinking_tokens_sum`, currently honestly `0.0` (no calls today had `thinking_enabled`), which is itself evidence this system tells the truth rather than fabricating a plausible-looking number.
  - Confirmed live in `phi_rewards` (Postgres): `phi_health: "ok"`, `encoder_version: "v20260712-seedv4-postfix"`, `features_version: "seed-v4"` — this is running right now, not aspirational.
- The raw input, `reasoning_activity_projection`, comes from `orion-thought`'s own HTTP endpoint: `GET {ORION_THOUGHT_BASE_URL}/projections/reasoning_activity` (fetched via `_fetch_orion_thought_reasoning_activity()`, `services/orion-spark-introspector/app/worker.py:916-930`, with a small read-through cache via `_substrate_read_cache()`).
- **Confirmed no circular dependency**: `build_inner_state_features(ss, ..., reasoning_activity_projection=...)` (`services/orion-spark-introspector/app/inner_state.py:354`) takes `reasoning_activity_projection` as an *independent* parameter, not derived from `ss` (the `SelfStateV1` object). spark-introspector's phi pipeline runs *downstream* of self_state (`phi_rewards.self_state_id` references a real self_state row) — so self_state cannot import from spark-introspector, but both can independently call `orion-thought`'s same HTTP projection with no ordering conflict.
- **New wiring required**: `orion-self-state-runtime` (the service running `orion/self_state/builder.py`, confirmed via `grep` — it's the sole importer) currently has zero connection to `orion-thought` — no HTTP client, no `ORION_THOUGHT_BASE_URL` in its `.env_example`. This is genuinely new plumbing for that service, not a rewire of something already there.

## Missing questions

- Does `orion-self-state-runtime` already make any outbound HTTP calls to sibling services (an existing client pattern to mirror), or is this its first external HTTP dependency? Changes how much scaffolding this needs — check before assuming a bare `httpx.AsyncClient()` call is the right shape.
- Does spark-introspector's `_substrate_read_cache()` read-through caching pattern need to be mirrored in `orion-self-state-runtime`, or is self_state's own tick cadence different enough that a direct uncached call is fine? Worth checking both services' poll intervals before deciding.
- Should `field_intensity` become a *pure* replacement (execution_load/reasoning_load only), or a blend that still incorporates *something* from the old field_attention pipeline once that's separately fixed? This spec proposes pure replacement — the old signal has no salvageable component; both the aggregation (`max()`) and the base data (`atom.salience` duplication) are broken, so there's nothing worth preserving a weighted share of.
- `reasoning_activity` as served by `orion-thought` — is it already subject-scoped to "orion" the way self_state is, or could there be an ambiguity here? Likely fine given self_state's own general orion-scoping, but worth a sanity check during implementation, not blocking the design.

## Proposed schema / API changes

No new schemas, no changes to `orion-thought`'s existing `/projections/reasoning_activity` endpoint — pure reuse.

`orion/self_state/scoring.py`: add a function mirroring `inner_state.py`'s computation exactly (same formula, same fallback chain, same honesty discipline — not a reinvention):

```python
def field_intensity_from_reasoning_activity(
    reasoning_activity_projection: dict | None,
    trajectory_active_runs: list[dict] | None = None,
) -> tuple[float, str]:
    """Real, log-scaled LLM completion-token throughput as a stand-in for
    field_intensity. Mirrors services/orion-spark-introspector/app/
    inner_state.py's execution_load computation exactly -- same formula,
    same fallback chain, same honesty discipline (explicit 0.0/"none"
    rather than a silent fake floor). Reused here, not reimplemented
    differently, so the two systems can't quietly drift apart again.
    """
    ...
```

`field_intensity_score()` (the old `max()`/perturbation-based function) stays in the file — check for other call sites first; delete only if confirmed unused elsewhere, otherwise mark deprecated with a comment pointing at this spec.

`self_state.dimensions["field_intensity"].reasons` should carry the real source label (e.g. `["reasoning_activity.completion_tokens_sum"]`), replacing today's degenerate `["no contributing channel evidence this tick"]` — this is a direct, visible improvement in `SelfStateV1`'s own self-honesty, not just a numeric fix.

## Files likely to touch

- `orion/self_state/scoring.py` — new `field_intensity_from_reasoning_activity()`, reusing the exact formula/fallback chain from `services/orion-spark-introspector/app/inner_state.py:277-295`.
- `orion/self_state/builder.py` — call the new function instead of `field_intensity_score(overall_salience, ...)` for the `field_intensity` dimension; thread through a `reasoning_activity_projection` parameter.
- `services/orion-self-state-runtime/app/worker.py` (or wherever inputs get assembled before calling the builder) — add an HTTP fetch to `orion-thought`'s `/projections/reasoning_activity`, mirroring `services/orion-spark-introspector/app/worker.py:916-930`'s `_fetch_orion_thought_reasoning_activity` pattern. Degrade gracefully (never raise) if `orion-thought` is unreachable — same discipline as every other adapter in this codebase.
- `services/orion-self-state-runtime/.env_example` (+ live `.env`, per this repo's mandatory env-parity rule) — new `ORION_THOUGHT_BASE_URL` key. Check whether a shared default for this already exists elsewhere in the repo before inventing a new one.
- Tests: find or create `orion/self_state/tests/test_scoring.py` (or wherever scoring is tested) — unit tests for the new function covering the real-value case, the step-count fallback, and the honest-zero case. Plus a builder-level test confirming `field_intensity` no longer routes through the old `max()`-based path.
- `orion/self_state/inner_state_registry.py` — update the `field_intensity` entry's notes to reflect the fix. This registry exists specifically to prevent re-discovery of exactly this class of bug; leaving it stale after fixing the bug it should have caught would defeat its purpose.

## Non-goals

- **Not fixing the `cpu_pressure`/`gpu_pressure` biometrics duplication** at its source (`biometrics_loop/grammar_extract.py`). Real hardware polling infrastructure exists (`orion/sensors/gpu_host_stats.sh`, a real `nvidia-smi`-based script) but isn't wired into the biometrics pipeline at all — that's a separate, larger piece of work (wiring a real poller into `node_vectors`), named as a follow-up, not blocking this patch.
- **Not fixing `uncertainty`** (also confirmed dead at exactly `0.0`, live, 7-day window). Separate root cause, not yet traced.
- **Not touching `field_attention/builder.py`'s `max()` aggregation itself.** `overall_salience` may have other real consumers beyond `field_intensity` (attention frame readers elsewhere) — this spec only changes what `field_intensity` is computed *from*, not the field_attention pipeline. If `overall_salience` turns out to have no other real consumers after this patch, that's a separate cleanup, not in scope here.
- **Not building the metric-liveness gate script** (a standing check for zero-variance dimensions) — a valuable, smaller, separate follow-up named in the brainstorm this spec grew out of.
- **Not unifying the two independent salience/attention systems** (conversational `compute_salience()` vs infrastructure `field_attention`) — investigated earlier in this session and rejected: they model genuinely different things (conversational topics vs mesh infrastructure), and forcing them together would fabricate a translation layer between incompatible evidence types.

## Acceptance checks

```bash
pytest orion/self_state/tests/ -k "field_intensity or scoring" -q
```

- `field_intensity_from_reasoning_activity()` returns a real, honestly-labeled, non-fabricated value when `completion_tokens_sum > 0`.
- Falls back correctly through the exact same chain as `inner_state.py`: step-count fallback, then explicit `0.0`/`"none"` — no silent fake floor at any stage.
- `self_state.dimensions["field_intensity"].reasons` reflects the real source label, not the old `"no contributing channel evidence this tick"`.
- The HTTP fetch to `orion-thought` degrades gracefully (self_state generation must not fail or hang if `orion-thought` is unreachable).
- **Live/manual check, documented in the PR report**: after deploying, query `substrate_self_state` over a real window (mirroring today's exact diagnostic: `select stddev(...) from substrate_self_state where created_at >= now() - interval '24 hours'`) and confirm `field_intensity` now has non-zero variance — the same query that found the bug, now used to confirm the fix.
- **Downstream ripple check, documented, not necessarily fixed in this patch**: `field_intensity` will likely sit much lower than the old permanent `1.0` most of the time, which will shift `weighted_overall_intensity()`'s condition classification and every proposal template's priority ranking (any template with `field_intensity` in its `dimensions` weights). Live-check `substrate_proposal_frames` priority distributions before/after, similar to today's P5 investigation — this is an expected, deliberate consequence of fixing a lie, not a regression, but it should be observed and named, not assumed silently.

## Recommended next patch

Implement via `/superpowers:subagent-driven-development`, worktree per repo convention (this spec's own worktree, `feat/field-intensity-real-signal` — reuse it). Single agent, not parallel tracks — the pieces are sequentially dependent (the HTTP wiring must exist before the builder can call it, the scoring function must exist before the builder call-site swap). Code review, README update for `orion-self-state-runtime` (document the new `orion-thought` dependency and why), live verification via the acceptance checks above (the same live Postgres diagnostic used to find the bug), commit, push, PR report.

Flag explicitly in the PR report: this changes `SelfStateV1.dimensions.field_intensity`'s real-world value distribution meaningfully (from a constant lie to real, honest, typically-lower values), which ripples into system condition classification and proposal prioritization. That's the point of the fix, not a side effect to hide — but it needs to be named plainly given how many downstream consumers read this field.
