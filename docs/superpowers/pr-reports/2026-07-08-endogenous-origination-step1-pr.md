# PR: Endogenous drive origination — Step 1 (DEFAULT-OFF)

**Status:** IMPLEMENTED + reviewed, default-OFF. A want with no external cause.
Ships behind `ORION_ENDOGENOUS_ORIGINATION_ENABLED=false`; **do not enable until
measurement 0(a) passes + sign-off** (spec gate). This branch also carries the
leaky-math spec refresh (earlier commits).

## Summary

- New `orion/autonomy/endogenous_origination.py` — a deterministic `OriginationEngine`
  that reads the continuous `SelfStateV1` stream (bounded ring of 8) and, **only in
  exogenous silence**, mints a `TensionEventV1(origin="endogenous")` when internal
  dynamics cross the band: `P = w_D·drift + w_W·dwell + w_A·agency ≥ threshold`.
- Pure substrate math — no LLM, no keyword/text routing. Degrades to `None`, never
  raises; ring is `deque(maxlen=window)`; magnitude capped; cooldown-gated.
- `TensionEventV1` gains `origin` + `origination_signal` (back-compat defaults).
- `goals._drive_origin`: an endogenous lead tension → `drive_origin="endogenous"`.
- Wired into concept-induction's `_tensions_from_self_state`: observe every
  self-state, `maybe_originate` merged into the tick — flows through the unchanged
  publish → `DriveEngine` → goals path.

## Architectural deviation from the spec (deliberate)

The spec placed the ticker in `orion-substrate-runtime`. I put it in
**concept-induction** instead, because that service *already* consumes
`substrate.self_state.v1`, holds `_previous_self_state`, and runs `DriveEngine` +
`GoalProposalEngine`. Placing it there rides the existing seam; the substrate-runtime
placement would have required inventing a new cross-service tension-publish +
consumer contract (violates "ride existing seams / no invented abstractions").

## Verified behavior (leaky substrate)

- Single endogenous firing @ cap 0.5 → drive pressure 0.500 (< 0.62 activate):
  a nudge, never activates alone.
- Eval `run_origination_eval.py`: **QUIET fired=2 over 20 min, BUSY fired=0
  (world-wins), 20/20 busy ticks suppressed by exogenous input. PASS.**

## Files changed

- `orion/autonomy/endogenous_origination.py` (new), `orion/autonomy/tests/test_endogenous_origination.py` (new, 13).
- `orion/core/schemas/drives.py` — `TensionEventV1.origin` + `origination_signal`.
- `orion/spark/concept_induction/goals.py` — `_drive_origin` lead_origin branch.
- `orion/spark/concept_induction/bus_worker.py` — engine construct + self-state hook.
- `orion/spark/concept_induction/settings.py`, `services/orion-spark-concept-induction/.env_example` — flag (default false) + params.
- `orion/spark/concept_induction/tests/test_endogenous_origination_wiring.py` (new, 5).
- `orion/autonomy/evals/run_origination_eval.py` (new).

## Schema / bus / API changes

- `TensionEventV1` gains optional `origin` (default `exogenous`) + `origination_signal`
  (default `{}`). Additive on the already-registered model → **no registry churn**.
- No new channels/kinds. Endogenous tensions publish on the existing tension channel.

## Env/config changes

- Added `ORION_ENDOGENOUS_ORIGINATION_ENABLED=false` + `ORIGINATION_*` params to
  `.env_example`. Run `python scripts/sync_local_env_from_example.py` on host.

## Tests run

```text
pytest orion/spark/concept_induction/tests orion/autonomy/tests -q  → 279 passed
  (13 origination unit + 5 wiring new)
python orion/autonomy/evals/run_origination_eval.py  → RESULT: PASS
```

## Review findings (Task 9, subagent) — no blocker/major

- **MINOR** — schema fields serialize on every tension even flag-off. Real impact
  low (monorepo single-deploy; no strict external consumer found). Mitigation:
  deploy schema-before-producer only if an external revalidating consumer is added.
- **MINOR** — exogenous-quiet gate is scoped to the self-state channel; concurrent
  input on other channels isn't visible to the gate. Documented design scope;
  900s cooldown bounds frequency.
- **NIT (fixed)** — per-snapshot `unresolved` copy now locally capped at 16.
- **NIT** — `predictive` drive is intentionally unmapped (predictive pressure is
  the exogenous world-pulse drive; endogenous origination targets the internal five).

## Enable / rollback

- Enable ONLY after 0(a) GO + review: `ORION_ENDOGENOUS_ORIGINATION_ENABLED=true`
  and restart concept-induction. Rollback: set `false` → producer emits nothing,
  byte-identical prior behavior.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-spark-concept-induction/.env \
  -f services/orion-spark-concept-induction/docker-compose.yml up -d --build
# (only if/when the flag is enabled after 0(a))
```

## Risks / concerns

- Severity low while flag is off (default). The change is inert until enabled.
- Enabling is a cognition-loop change gated on measurement 0(a) — not to be flipped
  without the drift verdict + review.

## PR link

<to be filled by `gh pr create` / GitHub UI>
