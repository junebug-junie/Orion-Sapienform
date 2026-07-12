# PR: Fix field-digester diffusion permanently saturating capability channels

## Summary

- Live bug: `apply_diffusion()` accumulated across ticks, and with real edges routinely contributing up to ~0.9/tick against an 8%/tick decay, this drove capability channels like `capability:orchestration`'s `"pressure"` (which `resource_pressure` reads) to a permanent ceiling of 1.0 — confirmed dead-flat for the entire observed `substrate_field_state` history, predating every change made in this session's redesign effort.
- Two compounding root causes fixed: (1) multiple legitimate source channels feeding the same target (e.g. `cpu_pressure` + `transport_pressure` both → `"pressure"`) were summed instead of combined via `max()`, even within a single tick; (2) the result then carried forward and accumulated further every subsequent tick.
- `apply_diffusion()` is now fully memoryless per diffusion-target channel, matching the same principle self-state's own `build_self_state()` already uses.
- Code review found no bugs in the fix itself, but surfaced a significant fact: the endogenous-origination NO-GO verdict was measured against data this exact bug was contaminating, and this fix carries the same (arguably sharper) phi-corpus deployment caution already open from Phase 1.

## Outcome moved

`resource_pressure` (and every other capability-fed dimension) can, for the first time in this substrate's history, actually move. This is the concrete fix for the "garbage in" complaint that started the whole redesign — Phase 1 fixed a real but different bug (raw channel double-counting); this fixes the one that actually explains why the metric never moved.

## Current architecture

Before this fix: `apply_diffusion()` ran `tgt[ch] = min(1.0, tgt.get(ch, 0.0) + contribution)` — additive, cross-tick-persistent. `apply_decay()` ran immediately before it in the same tick and multiplied the same channels by 0.92, but couldn't keep pace with multiple channels each contributing up to `weight × diffusion_rate` (~0.9) every 2-second tick.

## Files changed

- `services/orion-field-digester/app/digestion/diffusion.py`: `apply_diffusion()` rewritten to collect the best (max) real contribution per `(target_id, channel)` across all edges this tick, then set every diffusion-target channel fresh (zeroing channels with no current contributor, clearing their provenance too) instead of accumulating. Full rationale in the function's docstring.
- `services/orion-field-digester/app/digestion/decay.py`: added a comment (no logic change) documenting that its `capability_vectors` decay loop is now fully superseded by the memoryless diffusion for every channel currently in the topology — found by code review, left in place as a safety net rather than removed, since removing it wasn't necessary to fix the bug and touching it would have widened scope.
- `services/orion-field-digester/tests/test_diffusion_provenance.py`: rewrote one existing test that asserted the old (buggy) accumulation-preserving behavior; added new tests for the actual regression (`test_sustained_real_contribution_does_not_ratchet_up_across_ticks`), max-not-sum convergence, no-current-contributor reset, and confidence/available_capacity baseline preservation.

## Schema / bus / API changes

None — `FieldStateV1`'s shape is unchanged. Behavior changed: `capability_vectors` values and `capability_provenance` now reflect only the current tick, not a historical blend.

## Env/config changes

None.

## Tests run

```text
pytest services/orion-field-digester/tests/ -q
→ 20 passed

pytest tests/test_field_state_schemas.py tests/test_field_digestion_rules.py \
  tests/test_field_topology_reconciliation.py tests/test_self_state_builder.py \
  tests/test_self_state_builder_hardening.py tests/test_self_state_schemas.py \
  tests/test_self_state_scoring.py tests/test_self_state_policy_loader.py -q
→ 51 passed
```

`git diff --check`: clean.

## Evals run

None applicable.

## Docker/build/smoke checks

**Not run — deliberately not deployed this session.** See Risks below; this needs a decision before it goes live, not just a rebuild.

## Review findings fixed

Two-angle review (line-scan, cross-file consumer check).

- Finding: `capability:transport`'s direct diffusion of `delivery_confidence→confidence` and `bus_health→available_capacity` is silently overwritten every tick by the pressure-derived formula.
  - Fix: not fixed — confirmed pre-existing (same precedence existed before this patch, just computed multiple times per tick with intermediate values instead of once with the final one). Documented in the new docstring as a known quirk needing its own follow-up decision.
- Finding: `apply_decay()`'s `capability_vectors` loop is now fully dead weight for every channel diffusion touches — a genuinely new consequence of this fix, previously load-bearing (the only counterweight to unbounded accumulation).
  - Fix: documented with a comment explaining why it's currently inert and why it's kept (a safety net for any future non-diffusion capability-channel writer), rather than removed.
  - Evidence: full test suite still passes; the comment is directly at the code in question.
- Finding: `src = state.node_vectors.get(edge.source_id) or state.capability_vectors.get(edge.source_id, {})` uses dict truthiness, which could silently misroute if a node vector is ever genuinely empty or a node/capability id string ever collides.
  - Fix: not fixed — confirmed byte-for-byte unchanged from before this patch (pre-existing), and currently safe given `reconcile_field_state_with_lattice` always fully populates node vectors and node/capability id prefixes never collide in the current topology. Flagged as a candidate hardening follow-up, not blocking.
- Finding (significant, not a bug in this diff): the 2026-07-08 endogenous-origination NO-GO verdict (`median_abs_trajectory=0.0000` vs required `≥0.03`) was measured against 120 days of data this exact saturation bug was contaminating — the dimensions that metric reads are fed by the same capability channels that were pinned at 1.0.
  - Not fixed here — not this PR's decision to make. Flagged clearly: the verdict should be re-measured once this fix is live, before being trusted further. This is not grounds to reverse the NO-GO decision on its own; it's grounds to re-run the measurement.
- Finding (expected consequence, not a bug): metacog reflection triggers (`orion/substrate/metacog_trigger_signals.py`) and attention-novelty scoring (`orion/attention/field_attention/scoring.py`) both read dimensions that will now show real tick-to-tick variance instead of staying artificially flat — expect these to fire noticeably more often post-deploy. This is the metric finally being honest, not a regression, but worth knowing about before deploying so a sudden increase in trigger volume isn't mistaken for something broken.

## Restart required

**Deliberately not run this session — printed here for when the phi-corpus question below is resolved:**

```bash
docker compose --env-file .env --env-file services/orion-field-digester/.env \
  -f services/orion-field-digester/docker-compose.yml up -d --build
```

`orion-self-state-runtime` does not itself need a rebuild for this fix (it only reads whatever `field_digester` produces), but its behavior will change the moment `field-digester` is restarted with this code.

## Risks / concerns

- Severity: **high — same class of concern as the still-open Phase 1 phi-corpus decision, arguably sharper here.**
  Concern: `agency_readiness_score()` directly consumes `resource_pressure` (`base -= resource_pressure * 0.15`), and `agency_readiness` is one of the four `SelfStateV1`-sourced features the live phi encoder (seedv4) trains on. Phase 1's fix shifted an already-varying distribution; this fix takes `resource_pressure` from a **constant** (zero variance, pinned at 1.0 for its entire history) to something that will actually vary for the first time ever once deployed — likely a larger distributional jump than Phase 1's.
  Mitigation: **not resolved.** The three options from Phase 1's phi-corpus flag (accept-and-document / version-pin / retrain) apply here too, and this fix should not be deployed to the live field-digester until that question — still open since Phase 1 — is actually decided. This PR is code-complete and fully tested; it is deliberately not restarted live.
- Severity: medium
  Concern: the endogenous-origination NO-GO verdict may have been measured on contaminated data.
  Mitigation: re-measure `scripts/analysis/measure_autonomy_gate.py` after this fix is live, before treating the existing NO-GO verdict as settled for any future decision.
- Severity: low
  Concern: metacog/attention-novelty trigger volume will likely increase post-deploy.
  Mitigation: none needed beyond awareness; monitor after deploy.

## PR link

<!-- filled in after push -->
