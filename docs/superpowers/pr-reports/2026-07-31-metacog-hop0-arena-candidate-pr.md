# Hop 0: sustained metacog trend as a first-class arena candidate

Branch: `feat/metacog-hop0-arena-candidate`
Date: 2026-07-31
Status: **DONE_WITH_CONCERNS** — ships flag-off, and has never fired on real data (disclosed below)

## Summary

- Builds the piece the stream-of-consciousness arc has repeatedly stopped short of. The hop-chain
  design scopes hop 0 as a reducer *"registered as a candidate producer the same way reverie is —
  **not built as a bespoke standalone script**"*, and that registration is what never got built:
  `orion/metacog/trend_reducer.py`'s computation already existed, was already wired into
  `orion-equilibrium-service`'s metacog **trigger** path, and sat flag-off. A trigger is not arena
  citizenship.
- Unblocked this session: the design doc's step 1 was to wait for the confidence-gated
  `repair_pressure` series to pass ~20 rows and re-run the baseline measurement. It did — **n=20,
  `GENUINE_VARIATION`**, up from `INSUFFICIENT_DATA` at n=12.
- `orion/proposals/builder.py`'s `reverie_candidates` → `external_candidates`: reverie is no longer
  the only non-deterministic producer, and the arena stays the shared arbitration mechanism rather
  than any producer getting a private scheduler.
- Ships **OFF** (`ORION_METACOG_HOP_PROPOSE_ENABLED=false`), `operator_review` unconditionally.

## Outcome moved

There is now a real path from "a metacog signal has kept escalating" to "a governed candidate
competing for an attention budget slot against every other cognitive act." Before this, the trend
computation could only produce an LLM-drafted reflection into a write-only table.

Verified the gate is genuinely unconditional, end to end: `evaluator.py` maps
`required_policy_gate == "operator_review"` → `requires_operator_review`, and
`config/execution_dispatch/execution_dispatch_policy.v1.yaml` lists that in
`blocked_policy_decisions`. Risk 0.3 also sits below `defer_above_risk: 0.60`, so it is not diverted
to "deferred" before the gate is consulted.

## Architecture touched

`orion/metacog/` (new converter), `orion/proposals/builder.py` (param rename),
`services/orion-proposal-runtime` (store read + flag-gated worker branch). No schema change, no new
bus channel, no new trigger kind. `source` is `str | None`, so `"cognitive_hop"` needs no registry
change.

## Files changed

- `orion/metacog/proposal.py` (new) — `trend_result_to_candidate()`, mirroring
  `orion/reverie/proposal.py` field-for-field.
- `orion/proposals/builder.py` — `external_candidates`.
- `services/orion-proposal-runtime/app/{store,worker,settings}.py` — read, wiring, flag.
- `.env_example`, `docker-compose.yml`, live `.env` — env parity.
- `orion/metacog/tests/test_proposal.py` (new) — 13 tests.

## Two measured findings that shaped the design

**1. `is_sustained_trend` detects sustained ESCALATION, not sustained elevation.** Held at 0.55 for
12 ticks, z decays `463 → 1.79 → 1.19 → … → 0.27` and only **one** tick qualifies, because the EWMA
baseline adopts the new level as normal. A rising ramp sustains. Defensible for a prediction-error
framing — a persistently high value genuinely *is* the new normal — but it means hop 0 will not open
a chain about a condition that has been quietly bad for an hour, which is plausibly something a train
of thought should notice. Not changed here: the reducer is shared with the equilibrium trigger path,
so re-shaping it is its own patch with its own measurement. **Now pinned by a test.**

**2. Priority is run-length based, not z-proportional.** A flat input run drives the EWMA variance
toward its `1e-6` floor, so the first excursion after one scores `z=463`. A z-proportional salience
would have put hop 0 in the arena at priority 1.0 every time it fired — the design doc's own named
danger case (*"a chain that always self-scores just above the preemption threshold never actually
gets interrupted; 'interruptible' becomes a claim, not a verified property"*).

**Correction on that second one, from review:** my justification over-claimed. The z=463 artifact
requires a *perfectly* flat run; the real series has variance `4.15e-3`, three orders above the
floor, with max real z of 7.56. The run-length choice is still the better one, but "the live series
is exactly that shape" was too strong. Recorded rather than quietly amended.

## Why the store does not filter `confidence > 0`

That gate was the right discriminator until 2026-07-30, when `repair_pressure_v2`'s confidence fix
landed. Previously a confidently-calm text-fallback reading persisted with `confidence=0.0`,
indistinguishable from the appraiser's true "no evidence" signal. Post-fix every row carries the real
`_TEXT_FALLBACK_CONFIDENCE` (0.65). Verified live, per day:

```
07-24  5/26 confidence>0        07-30  3/4
07-29  7/23                     07-31  6/6  @ avg 0.650
```

So the gate now filters nothing, and re-applying it would discard exactly the readings that establish
the rest state — a confidently-calm 0.087 *is* the baseline the z-score needs to be anomalous
against. Independently re-verified in review.

## Tests run

```text
$ pytest orion/metacog/tests tests/test_proposal_frame_builder.py \
         tests/test_proposal_transport_readonly_candidates.py -q
59 passed in 0.39s

$ services/orion-proposal-runtime$ pytest tests -q
3 passed
```

No regressions from the `external_candidates` rename: the two builder test files give **15 passed on
both this branch and `origin/main`**.

`orion/reverie/tests/test_proposal.py` is **8 failed / 31 passed on `origin/main` too** —
`spontaneous_thought_to_candidate() got an unexpected keyword argument 'self_state_id'`, stale since
the 2026-07-22 SelfStateV1 burn. Pre-existing, untouched by this PR, but worth a follow-up given this
PR's premise is "mirror reverie."

## Live verification

```text
readings in 7d window:                43
sustained ticks over real history:     0
candidate produced:                 NONE
latest: z=-0.403  count=43  cold_start=False
```

**Hop 0 has never fired on real data.** The reducer loads real rows, folds them correctly, and is not
cold-start or degenerate — it simply hasn't seen a sustained escalation. That is honest behavior for a
quiet series, but it means the design doc's acceptance check 2 (*"does it ever win a budget slot, does
it ever get preempted"*) is **unanswerable today**. Ships off accordingly.

## Review findings fixed

1. **must-fix — `chain_id` not stable across a chain's life.** Hashed the latest reading, so one
   continuous escalating trend minted a new identity every row (reproduced: consec 3/4/5/6 → 4
   distinct ids). Breaks feedback/consolidation attribution and leaves hop 1 nothing to point
   `parent_hop_id` at. Now keyed on the run's onset. My own test had asserted the instability as
   correctness.
2. **must-fix — `confidence_score` permanently saturated.** `state.count / 100.0` only grows, pinning
   at 1.0 after ~100 rows (~4.5 days) and never returning — exactly what metric quality gate step 4
   forbids, and the `dimension_confidence()` precedent I cited uses count only as a cold-start gate.
   Now inverted from |z| against the arena's own 3.0 convention, tested to reach both 0.0 and 1.0.
3. **should-fix — empty `evidence_refs`** while asserting "elevated for N readings". Now carries the
   run's rows.
4. **should-fix — unbounded replay.** A bare `LIMIT 500` slides the replay's start point once history
   exceeds the cap, so the EWMA restarts mid-series. Now an explicit 7-day window, with ordering cast
   to `timestamptz` rather than relying on lexicographic varchar ordering.
5. **should-fix — characterization was comment-only.** The escalation-vs-elevation property now has a
   test that folds a step-hold through the real reducer and asserts exactly 1 sustained tick.

Confirmed clean by review: the `external_candidates` rename (no caller missed), env parity across all
four surfaces, and the unconditional `operator_review` gate.

## Restart required

None — ships flag-off, no behavior change until enabled.

```bash
# when ready to enable, after a real candidate is observed:
ORION_METACOG_HOP_PROPOSE_ENABLED=true
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-proposal-runtime up -d --build
```

## Risks / concerns

- **Severity: should-know. Never fired on real data.** See Live verification. Enabling it proves
  nothing until a real sustained escalation occurs; the acceptance checks stay open.
- **Severity: should-know. Reducer baseline state is not checkpointed.** The worker replays from a
  fresh `MetacogTrendStateV1()` each tick. `trend_reducer.py`'s own docstring has a
  "## Checkpointed, resumable state" section claiming it answers the design doc's MQ4. I deferred it
  on the grounds that hop 0 opens a chain and has nothing to resume — but review correctly notes
  that is *chain* state, whereas this is *reducer baseline* state, which the reducer was explicitly
  built to persist. Either persist it or withdraw that docstring claim; not resolved here.
- **Severity: note. Frame truncation silently drops candidates.** `builder.py` does
  `active = active[: policy.limits.max_candidates]` and never appends the overflow to `suppressed`.
  Live: 12 templates, `max_candidates: 10`, frame shows 10 active / **0 suppressed** — 2 vanished
  with no record. Pre-existing, but this PR is the first to push a *new* producer through that cap.
- **Severity: note. Replay cost.** ~1,252 proposal frames/hour, each re-querying and re-folding, on a
  table gaining ~8.7 rows/day and with no index on `created_at`. Trivial today, not at 100k rows.

## PR link

<to be filled after push>
