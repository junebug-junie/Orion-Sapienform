# PR: End dispatch-arena starvation — visible truncation, a reserved lane, and aging

Branch: `feat/dispatch-starvation-fix`

## Summary

- Fixed a **measured** starvation defect: over three live hours the same five read-only templates
  held all five dispatch slots on essentially every tick while `prune_build_cache` — the only
  action in the arena that changes anything about the host — was dispatched **0 times**.
- Made the loss **visible**. `make_blocked` hardcoded `dispatch_kind="noop"` on every blocked
  record, so a starved mutating action and a starved inspect were byte-identical in the one column
  that distinguishes them. Confirmed against 300 real stored frames: every blocked row in live
  history says `noop`.
- Added a **reserved lane** (1 of 5 slots for `maintenance_bounded`), free on ticks nothing claims
  it.
- Added **aging**: a bounded bonus per consecutive loss, so starvation is finite rather than
  permanent — the general fix, applying to every kind, not just the one that prompted it.
- Fixed **two of my own bugs found after the first deploy**, both of the "reports success while
  changing nothing" class: a counter-key collision that made aging a permanent no-op for exactly
  the templates it existed to serve, and a save path that silently wiped every counter.

## Outcome moved

`prune_build_cache` went from structurally-unwinnable to winning a reserved slot whenever it is
proposed. Instrumentation that did not exist before now answers "what is starving, and for how
long" directly from the stored frames.

**Not yet demonstrated:** a real `builder_prune` execution with a `bytes_reclaimed` number. See
Risks.

## Current architecture (before)

`orion/execution_dispatch/builder.py` iterated `policy_frame.decisions` in order and appended to
`candidates` until `len(candidates) >= max_dispatch_candidates` (5), then blocked the rest. Because
the upstream proposal frame is priority-sorted, this was effectively "top 5 by a single scalar,
first-come-first-served." Blocked records recorded `dispatch_kind="noop"` regardless of kind, and
nothing was ever aged.

## Architecture touched

`orion/execution_dispatch/` (builder, policy model), `orion/schemas/execution_dispatch_frame.py`,
`config/execution_dispatch/`, and `services/orion-execution-dispatch-runtime/` (worker `_tick`
carry-forward, store loader). No bus channel carries this frame; no contract change beyond two
additive schema fields.

## The defect, measured

```
inspect_bus_channel_catalog            DISPATCHED  267
inspect_attended_target                DISPATCHED  267
summarize_transport_contract_drift     DISPATCHED  267
inspect_field_topology_catalog         DISPATCHED  243
watch_transport_backpressure           DISPATCHED  209
...
prune_build_cache    BLOCKED:max_dispatch_candidates_exceeded          4
prune_build_cache    BLOCKED:policy_decision:requires_operator_review  2
prune_build_cache    DISPATCHED                                        0
```

A single scalar ranking conflates **urgency** (a spike) with **importance** (a persistent level).
A steady-state-high signal like disk fullness is maximally penalised by that: high *and* boring, so
it loses forever, and loses invisibly.

Also measured, and it bounds what this patch can achieve: `prune_build_cache` reaches the arena
only **29 times in 3 hours** (vs 1876 for the leaders), avg priority 0.4163, avg rank 9.5 of a
10-slot slate. The reserved lane **converts opportunities, it does not create them**.

## Files changed

- `orion/execution_dispatch/builder.py`: `_admit_candidates` (two-pass reservation + general
  fill), `starvation_key` / `proposal_template_key` / `effective_priority`, blocked records carry
  real kind + streak, frame-level starvation warnings. `_decision_template_key` collapsed onto the
  shared parser.
- `orion/execution_dispatch/policy.py`: `DispatchLimitsV1.reserved_slots_by_scope`,
  `starvation_aging_bonus_per_tick`, `starvation_aging_bonus_cap`.
- `orion/schemas/execution_dispatch_frame.py`: `ExecutionDispatchFrameV1.starvation_counts`,
  `ExecutionDispatchCandidateV1.starvation_ticks`. Both additive with defaults.
- `config/execution_dispatch/execution_dispatch_policy.v1.yaml`: the three new limits.
- `services/orion-execution-dispatch-runtime/app/store.py`: counters loaded off the row the
  staleness baseline already reads; `_coerce_starvation_counts`.
- `services/orion-execution-dispatch-runtime/app/worker.py`: `carry_forward_baseline` for the two
  save paths that compute no counters of their own; stale ordering comment corrected.
- `tests/test_dispatch_starvation.py`: 21 tests, new file.
- `tests/test_execution_dispatch_runtime_worker.py`: +1 round-trip test.
- `services/orion-execution-dispatch-runtime/README.md`: new "Slot arbitration and starvation"
  section; stale "Known gap" replaced.

## Schema / bus / API changes

- Added: `ExecutionDispatchFrameV1.starvation_counts` (`dict[str,int]`, default `{}`),
  `ExecutionDispatchCandidateV1.starvation_ticks` (`int`, default `0`).
- Behavior changed: blocked candidates now carry their real `dispatch_kind` instead of `"noop"`.
  `reasons[0]` is deliberately unchanged (`max_dispatch_candidates_exceeded`) so existing queries
  keep working; new detail rides in `reasons[1:]`.
- Compatibility: verified by loading **300 pre-existing stored frames** through the new models —
  `ok=300 bad=0`. No `schema_version` bump; no `channels.yaml` change owed (no bus channel carries
  this frame). `ExecutionDispatchFrameV1` is registered by class in `orion/schemas/registry.py`.

## Env/config changes

- Added keys: none.
- `.env_example` updated: not applicable — no env key added, removed, renamed, or changed meaning.
- Config-only change, in `config/execution_dispatch/execution_dispatch_policy.v1.yaml`, which is
  checked in and shipped in the image (verified: see Docker checks).

## Tests run

```text
pytest tests/test_dispatch_starvation.py -q                       21 passed
pytest tests/test_execution_dispatch_runtime_worker.py -q         48 passed
pytest tests/test_maintenance_dispatch_gating.py -q               19 passed
pytest tests/test_execution_dispatch_builder.py -q                 8 passed
pytest tests/test_execution_dispatch_runtime_store.py -q          29 passed
pytest tests/test_execution_dispatch_frame_schemas.py -q          10 passed
pytest tests/test_execution_dispatch_policy_loader.py -q           6 passed
                                                          --------------------
                                                                 141 passed
```

Red-before-green confirmed for the three tests that matter, not assumed:

- Reserved lane and aging: re-run under pre-patch config (`reserved_slots_by_scope={}`, bonus 0.0)
  → prune admitted `False`, aged loser admitted `False`.
- Counter-key collision: re-run with the old `{kind}:{target}` key → `AssertionError`.
- Unevaluable counter wipe: fix reverted, test alone → `AssertionError: an unevaluable frame must
  carry starvation counters forward`.

Pre-existing failures on `main`, unrelated and untouched: `test_consolidation_policy_loader.py`
(3), `test_feedback_policy_loader.py` (1).

## Evals run

```text
none — services/orion-execution-dispatch-runtime has no evals/ directory.
```

This service has no eval harness. Not created here: the behaviour this patch changes is
arbitration, whose quality signal is exactly the `starvation_counts` history the patch begins
persisting, and there is not yet enough of it to write a meaningful eval against. Follow-up below.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-execution-dispatch-runtime build      -> Built
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d      -> Started

docker run --rm --entrypoint python <image> -c "load_execution_dispatch_policy(...)"
  config in image: ['/app/config/execution_dispatch/execution_dispatch_policy.v1.yaml']
  reserved: {'maintenance_bounded': 1}
  aging per tick/cap: 0.002 0.25
  max_candidates: 5
  maintain route timeout: 720.0

docker logs --since 6m  -> 0 error/traceback/exception lines
bus url: redis://100.92.216.81:6379/0 (Tailscale, per mandate)
```

Live path moved — real frames, post-deploy:

```
kind      starv  count      starvation_counts (a real frame)
inspect     4     15        {"inspect_transport_status:capability:transport": 4,
summarize   3      4         "inspect_execution_pressure:capability:orchestration": 4,
observe     2      4         "watch_transport_backpressure:capability:transport": 1,
                             "inspect_field_topology_catalog:...yaml": 1}
```

Before this patch every one of those rows read `noop` with no streak.

## Review findings fixed

Code review run in a subagent at `high` effort. Every finding below was independently reproduced
before being fixed.

- **Finding:** `starvation_key` was `{proposal_kind}:{target_id}`, and two live templates collide
  on `inspect:capability:orchestration` — `inspect_attended_target` (avg rank 3.0, admitted almost
  every tick) and `inspect_execution_pressure` (avg rank 7.5, starves). The winner's reset popped
  the loser's counter every tick, so aging was a **permanent no-op for exactly the population it
  existed to serve**.
  - **Fix:** key on `{template_key}:{target_id}`, template recovered from the id shape
    `stable_proposal_id` actually builds, split from the left (attention_frame_id contains colons),
    with a coarser fallback so an id-format change degrades aging rather than crashing dispatch.
  - **Evidence:** found by me in live frames ~10 min after the first deploy — only one key
    (`inspect:capability:transport`, the one with a single owner) ever appeared while ~16
    candidates per window were starving. Reviewer independently reproduced it with an 11-tick
    closed loop. Post-fix live frames show
    `"inspect_execution_pressure:capability:orchestration": 4` accumulating.

- **Finding:** `build_unevaluable_execution_dispatch_frame` was stamped with `updated_baseline`,
  which omits `starvation_counts`, so it saved `{}`, became the newest row, and the next tick's
  load erased every accumulated counter. Same hole already patched on the sibling stale-discard
  path one screen above it.
  - **Fix:** both non-computing save paths now share an explicit `carry_forward_baseline`.
  - **Evidence:** new `test_unevaluable_frame_carries_starvation_counters_forward`, confirmed red
    against the unfixed worker. Latent, not currently firing: 0 unevaluable frames in 24h of
    34,161.

- **Finding:** `_admit_candidates` enforced capacity on a `set` of `decision_id` but returned a
  list filtered by membership — duplicate ids would exceed capacity.
  - **Fix:** count the emitted list. **Evidence:** reviewer demonstrated `len(admitted)=3` at
    `capacity=2` with duplicate ids; unreachable with today's producer.

- **Finding:** counters clear on **admission**, not on a confirmed send, which is only safe while
  `max_dispatches_per_tick >= max_dispatch_candidates`.
  - **Fix:** decision recorded explicitly, and the coupling is now
    `test_the_send_budget_cannot_silently_drop_an_admitted_candidate` rather than a coincidence of
    two equal config values. **Evidence:** reviewer measured 0 of 2,738 frames leaving a prepared
    candidate unsent.

- **Finding:** every aging test hand-injected `prev_counts`, so none could catch a break in the
  counter **lifecycle** — only in the bonus arithmetic. This is why the collision bug reached
  production.
  - **Fix:** closed-loop multi-tick test feeding each frame's own output into the next tick, plus
    its inverse (a gap wider than the cap must starve across 500 real iterations); first worker
    round-trip test; the near-tautological determinism test replaced with input-order permutation.
  - **Evidence:** the closed-loop test immediately caught an arithmetic error in its own first
    draft (a 0.43 gap cannot be closed by a 0.25 cap).

- **Finding:** two functions parsed the same `proposal_id` format 400 lines apart. **Fix:**
  `_decision_template_key` delegates to `proposal_template_key`.

- **Finding:** `worker.py`'s send loop claimed `frame.candidates` is ordered by
  `build_proposal_frame`'s priority — after this patch the ordering authority is
  `_admit_candidates`' effective priority, which aging reorders, and the sequential take depends on
  it. **Fix:** comment corrected.

- **Finding:** README's copy-pasteable query lacked the `jsonb_typeof` guard `store.py` already
  carries; the "same failure as the drives program" line was unsourced. **Fix:** guard added;
  the line now attributed as Juniper's observation about shape, explicitly not a claim of shared
  root cause.

## Restart required

Already applied on this host during verification:

```bash
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
```

## Risks / concerns

- **Severity: medium.** **The prune has not yet been observed to actually fire.** This patch
  removes the capacity race, which is the gate that was demonstrably killing it (4 of the 6
  observed). It does not touch the other two constraints: `requires_operator_review` at the policy
  layer (2 of 6), and the fact that it only reaches the arena 29 times in 3 hours at all. Verdict
  is `DONE_WITH_CONCERNS` for exactly this reason — the outcome claim is "it can now win", not
  "it won".
  - **Mitigation:** `starvation_counts` and `starvation_ticks` are persisted precisely so the next
    check is a query, not a guess. The unmerged `fix/disk-capacity-pressure-trigger` branch is what
    would raise the 29.
- **Severity: low.** Both aging constants (0.002/tick, cap 0.25) are **uncalibrated starting
  values, disclosed as such** in the yaml, `policy.py`, and the README. There was no real starvation
  history to calibrate against because nothing recorded it until this patch.
- **Severity: low.** Aging changes admission ordering for **every** dispatch kind, not just the
  maintenance one. That is intended (the general fix), but it is a system-wide ranking change and
  the first patch in this arc to touch one. Blast radius is bounded by the 0.25 cap: it cannot
  invert a large priority gap, asserted by
  `test_aging_alone_cannot_close_a_gap_larger_than_its_cap` across 500 real iterations.
- **Severity: low.** `reserved_slots_by_scope` is a general mechanism with exactly one entry today.
  Reserving every slot would starve the read-only lane instead — the same defect pointed the other
  way — and is guarded by `test_the_shipped_config_actually_reserves_a_maintenance_slot`.

## Follow-ups not done here

1. No eval harness for `orion-execution-dispatch-runtime`. The natural first eval is a starvation
   eval over real `starvation_counts` history: "no dispatch kind exceeds N consecutive losses over
   a real window." Needs history this patch only just started producing.
2. `scripts/analysis/gate_channel.py` (the metric-gate script) — still unbuilt.
3. `prune_build_cache` has no per-template cooldown; `stable_dispatch_id` embeds `field_tick_id`,
   so there is no cross-tick dedup if it starts winning every tick.
4. Whether the drives program's starvation shares a root cause with this one is unexamined.

## PR link

<https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/dispatch-starvation-fix>
