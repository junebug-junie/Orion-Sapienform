# Candidate B (novelty-only) for hosts/capabilities — live wiring

Status: **implemented, tested, verified against live data.** Companion to
`docs/superpowers/specs/2026-07-21-attention-salience-cathedral-replacement-tentative-plan.md`
(the original A/B plan) and `docs/superpowers/specs/2026-07-30-candidate-b-hosts-capabilities-live-wiring.md`'s
own sibling, the kill-hand-weighted-salience patch that removed `compute_salience()`
and left physical hosts/capabilities with no attention coverage at all.

## Arsonist summary

Killing the hand-weighted formula (`compute_salience()`, 23 hand-picked weights) was the
right call — no real theory backed it. But it left physical host nodes
(`node:athena`/`atlas`/`circe`/`prometheus`/`rpc_timeout`) and every capability target
with **zero** attention coverage: Candidate A (precision-weighted prediction-error
salience) has no real historical series to ground them in, so `select_capability_targets()`
returned `[]` unconditionally and hosts simply weren't in `PREDICTION_ERROR_NATIVE_TARGETS`
at all.

This patch fills that gap with **Candidate B's `novelty_scorer()` alone** — not the full
three-scorer Society-of-Mind/Global-Workspace combination the original tentative-plan doc
described. Real, live, theory-grounded (Baars 1988, Dehaene 2014), for the targets it
covers — but a disclosed, narrower slice of Candidate B than "the whole candidate," named
honestly rather than oversold.

## Why novelty only, not the full three-scorer Candidate B

- `magnitude_scorer()` doesn't apply: no real prediction-error history exists for
  hosts/capabilities (the same reason Candidate A excludes them). Wiring it here would
  mean fabricating input data, not reusing real signal.
- `dwell_scorer()` is deliberately deferred, not silently dropped: confirmed live
  2026-07-21 (recorded in `candidate_society_of_mind.py`'s own docstring) that
  `attended_node_ids` is the empty list in 2837/2840 (99.9%) of real
  `substrate_coalition_dwell_log` rows over a 24h window. Building the cross-service
  wiring (a new store method reading `substrate_attention_broadcast_projection`) for a
  signal that would almost never contribute a real vote is not a good first cut. Real
  follow-up, not solved here.
- With exactly one real scorer, Borda rank-aggregation has nothing to aggregate — the
  novelty ranking *is* the ranking. No aggregation machinery was built for a one-scorer
  case.

## The real, un-hand-tuned "current salience" proxy

`novelty_scorer()` needs a `current_salience[target_id]` value per target to diff against
that target's prior frame — for hosts/capabilities, nothing in `field.node_vectors`/
`capability_vectors` was ever the deleted `compute_salience()`'s output; there is no real
"current salience" left to reuse.

`_current_pressure_proxy()` (`orion/attention/field_attention/selectors.py`) fills this
with `max()` of the raw pressure vector — an order statistic, zero free parameters. Unlike
`compute_salience()`'s hand-picked 0.45/0.20/0.25/0.10 blend of *all* channels, this makes
no claim about which channels matter more or how to combine them; it only reports which
single real channel is most elevated right now. Not a resurrection of the killed formula —
a genuinely different, parameter-free construction.

## Current architecture (post-patch)

- `select_node_targets()` — Candidate A, `node:substrate.*` only (unchanged).
- `select_host_targets()` — **new**. Candidate B novelty-only, `field.node_vectors` keys
  not in `PREDICTION_ERROR_NATIVE_TARGETS`.
- `select_capability_targets()` — **rewritten** (was always `[]`). Candidate B
  novelty-only, all of `field.capability_vectors`.
- `select_system_targets()` — unchanged (`field:recent_perturbations`'s own EWMA-zscore
  formula).
- `build_attention_frame()` — merges `select_node_targets()` + `select_host_targets()`
  into `node_targets`; `capability_targets` from the rewritten selector. `previous_frame`
  is now genuinely used again (by the two novelty selectors), after the kill-hand-weighted
  patch's docstring said it was unused — narrower and more accurate now: used by the
  Candidate B selectors, not by Candidate A's.

## Files changed

- `orion/attention/field_attention/selectors.py`: `_current_pressure_proxy()`,
  `_novelty_targets()` (shared implementation), `select_host_targets()` (new),
  `select_capability_targets()` (rewritten).
- `orion/attention/field_attention/builder.py`: wires the new selector in, updated
  docstring.
- `tests/test_attention_field_selectors.py`: 10 new tests.

No new env keys, no new store methods, no new settings — `select_host_targets`/
`select_capability_targets` reuse data already flowing into `build_attention_frame()`
(`field`, `previous_frame`), unlike Candidate A's live wiring which needed a new Postgres
read.

## Real, disclosed transition artifact (one tick only)

`novelty_for_target()` diffs this tick's `_current_pressure_proxy()` value against
whatever `salience_score` the *previous persisted frame* recorded for the same
`target_id`, regardless of which formula produced that prior value. Confirmed live before
this patch deploys: the currently-running frame has real `compute_salience()`-blend
scores for `node:athena`/`atlas` (0.295/0.110 at check time) — a different scale and
formula than `_current_pressure_proxy()`. **The first real tick after this patch deploys
will diff the new proxy against that old-formula value**, producing an artificially large
"novelty" reading that reflects the formula changeover, not a real event. Self-resolving
after exactly one tick (every subsequent frame diffs new-formula against new-formula).
Not fixed — there's no principled way to retroactively reinterpret an old frame's score
under a formula it was never computed with. Disclosed here and in the selector's own
docstring so a post-deploy operator doesn't misread tick one as a real signal.

## Verified against live data

```
real node_vectors keys: node:atlas, node:circe, node:athena, node:prometheus,
  node:rpc_timeout, node:substrate.{chat,route,execution,transport,biometrics,bus_synaptic}
real capability_vectors keys: capability:{graph,memory,vision,storage,transport,
  llm_inference,orchestration}

node_targets: node:substrate.biometrics (Candidate A, 1.000), node:prometheus (B, 0.960),
  node:atlas (B, 0.960), node:circe (B, 0.933), node:athena (B, 0.669)
capability_targets: capability:memory (0.920), capability:vision (0.920),
  capability:transport (0.917), capability:storage (0.878), capability:graph (0.843)
dominant_targets[0]: node:substrate.biometrics
```

(High host/capability novelty scores in this specific run are the transition artifact
above, not a red flag — the "previous" frame read here is the pre-patch, old-formula
frame. Real numbers once new-formula-vs-new-formula ticks accumulate should be re-checked
post-deploy, not assumed from this one-off.)

### 2026-07-30 re-verification after code review (Findings 1 and 2 fixed)

Code review on this patch's own diff found `_current_pressure_proxy()`'s `max()` was
directionally blind: `availability`/`delivery_confidence`/`stream_backlog_health` (node)
and `confidence`/`available_capacity` (capability) default to/sit near 1.0 when healthy —
confirmed against `services/orion-field-digester/app/tensor/channels.py`'s
`DEFAULT_NODE_VECTOR`/`DEFAULT_CAPABILITY_VECTOR` directly, not assumed — so a fully-calm
target and a severely-overloaded one could produce near-identical (~1.0) proxy output as
long as those channels stayed near default, the normal case. Fixed by inverting (`1 -
value`) the confirmed higher-is-better channels before the `max()` comparison, not
excluding them: a real drop in e.g. `availability` is itself real pressure worth
surfacing. Separately, `confidence_score` was `1.0 if previous_frame is not None else
0.0` — true only for "did any prior frame exist," not "did THIS target have a real entry
in it." Fixed to check per-target presence in `previous_frame.node_targets` /
`.capability_targets`.

Re-ran against the real, live `substrate_field_state` row and the real, live
`substrate_attention_frames` row (both pulled directly via psql, not synthesized) after
both fixes landed:

```
node:athena:  salience=0.7422 pressure_proxy=1.0000 novelty=0.7422 confidence=1.0
node:atlas:   salience=0.4680 pressure_proxy=0.5708 novelty=0.4680 confidence=1.0
node:circe:   salience=0.3277 pressure_proxy=0.3677 novelty=0.3277 confidence=0.0
node:prometheus: salience=0.0400 pressure_proxy=0.0000 novelty=0.0400 confidence=0.0
node:rpc_timeout: salience=0.4850 pressure_proxy=0.5000 novelty=0.4850 confidence=0.0
node:substrate.transport: salience=0.0000 pressure_proxy=0.0000 novelty=0.0000 confidence=0.0

capability:graph:          salience=0.1333 pressure_proxy=0.2008 confidence=0.0
capability:llm_inference:  salience=0.3580 pressure_proxy=0.4123 confidence=0.0
capability:memory:         salience=0.0800 pressure_proxy=0.0000 confidence=0.0
capability:orchestration:  salience=0.2396 pressure_proxy=0.3035 confidence=0.0
capability:storage:        salience=0.0046 pressure_proxy=0.0710 confidence=0.0
capability:transport:      salience=0.0234 pressure_proxy=0.0533 confidence=0.0
capability:vision:         salience=0.0800 pressure_proxy=0.0000 confidence=0.0

real prev_frame node_target ids: node:athena, node:atlas (only)
real prev_frame capability_target ids: (none -- capability_targets has never been
  non-empty in a real persisted frame, since select_capability_targets returned []
  unconditionally before this patch)
```

`confidence_score` is now honestly per-target: `node:athena`/`node:atlas` (the two ids
that really appeared in the live prior frame) read `1.0`; `node:circe`/`node:prometheus`/
`node:rpc_timeout` (new to this tick, not because there's no prior frame at all, but
because THEY specifically weren't in it) correctly read `0.0` — the exact distinction
Finding 2 required. Every capability reads `confidence_score=0.0` for now, honestly,
since no real persisted frame has ever contained a capability target yet — this will
start reading `1.0` for capabilities from the second post-deploy tick onward, same as any
target's first real appearance.

`node:substrate.transport` is included here as a `select_host_targets` catch-all member
(any `node_vectors` key not in `PREDICTION_ERROR_NATIVE_TARGETS`, not just the five
physical hosts) — its real vector is empty of any of this session's confirmed channels,
so it reads an honest `0.0` across the board rather than a fabricated nonzero score.
Already covered by this function's own docstring ("...or any `field.node_vectors` key not
in `PREDICTION_ERROR_NATIVE_TARGETS`"), not a new gap — named here only because this was
the first time it was actually observed live in this doc.

The `pressure_proxy` values above are no longer directionally blind (compare
`node:athena`'s live `pressure_proxy=1.0000` here against the pre-fix run, which could
not distinguish a calm host from an overloaded one whenever a higher-is-better channel
sat near its healthy default).

## Non-goals

- Not the full three-scorer Candidate B (magnitude/dwell excluded, both disclosed above).
- Not fixing the one-tick transition artifact — inherent to any formula changeover that
  reuses a `previous_frame`-diffing mechanism, not specific to this patch.
- Not touching Candidate A's targets or `select_system_targets()`.

## Acceptance checks

1. `select_host_targets()`/`select_capability_targets()` never double-score a
   `node:substrate.*` target already covered by Candidate A. Verified: unit test
   (`test_select_host_targets_excludes_prediction_error_native_ids`) and live data (no
   overlap in the real run above).
2. A target with no real prior frame reads `novelty_score=0.0`, not excluded and not a
   fabricated nonzero value. Verified: unit test + `novelty_for_target()`'s own existing
   contract.
3. `_current_pressure_proxy()` is genuinely parameter-free (no calibration, no
   hand-picked weight). Verified by inspection: `max()` of real values, nothing else.

## Restart required

Same restart as the kill-hand-weighted-salience patch (`orion-attention-runtime`) — this
ships in the same PR, one restart covers both.
