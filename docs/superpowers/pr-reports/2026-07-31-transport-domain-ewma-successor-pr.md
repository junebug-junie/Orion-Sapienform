# Finish retiring the transport domain; bus_synaptic is the instrument

Branch: `fix/transport-domain-ewma-successor`
Date: 2026-07-31
Status: **DONE_WITH_CONCERNS** (see the failed live-node deletion)

## Summary

- The 2026-07-26 pass killed `node:substrate.transport`'s **write** and deliberately kept everything
  else. Five days later the reducer was still **reading** it.
- `_PREDICTION_ERROR_DOMAIN_NODE_IDS` still mapped `node:substrate.transport -> "transport"`, so the
  brain-frame reducer handed `transport` to `reduce_attention_self_model()` on every tick, off a node
  frozen since 2026-07-24.
- `transport_prediction_error()` **deleted**, not left importable.
- `predicted_shift`'s argmax was **unfiltered** — any key a caller supplied could become Orion's
  stated "what is about to shift". Now gated on `ACTIVE_INFERENCE_DOMAINS`.
- Ships a **failing gate**, not another comment: set equality between the node map and
  `ACTIVE_INFERENCE_DOMAINS`.
- Stale node **could not** be deleted from live FalkorDB — it resurrects in 6s from a frozen 7-day-old
  snapshot, and survives stopping `substrate-runtime`. Reported as an open defect, not claimed as done.
- Also flips `ORION_METACOG_HOP_PROPOSE_ENABLED` to `true` (Juniper's go-ahead).

## Outcome moved

The Active-Inference domain set is now the same set everywhere: producer, reducer map, confidence
aggregate, and `predicted_shift`. Before this patch those four disagreed, and the disagreement was
invisible because the disagreeing member was frozen.

Live state of the node (it is still there — see the deletion section):

```text
node:substrate.transport
  prediction_error   0.556078
  observed_at        2026-07-24T21:55:26Z    (7 days stale)
  recency_score      0.0
  salience           0.556078                <- visible to generic concept consumers
  relationships      0
  provenance         substrate_runtime_worker / substrate_prediction_error   (producer dead)
```

## Why a comment was not enough

Nothing failed for those five days. The 2026-07-26 patch's own comment said killing the write "is
the entire change" — it was not, because a reader still listed the dead identity, and no test
expressed the invariant that a domain's producer and its readers must retire together.

This is CLAUDE.md's "deterministic gates over repeated yelling" case verbatim: the right fix for a
partial retirement is a failing gate, not a louder comment.

## The `predicted_shift` finding, stated honestly

`reduce_attention_self_model()`'s `predicted_shift` took an argmax over
`prediction_error_trend_by_domain` **with no filter**. `transport` was in that dict every live tick.

It **never actually won**. All persisted rows carrying a `predicted_shift`:

```text
execution      4597
biometrics     1490
bus_synaptic   1031
route             1
transport         0
```

That is not the filter working — there was no filter. It is that a frozen node has a flat `0.0`
trend, so `abs(trend) > abs(top_trend_val)` could never fire for it. A dead domain was invisible
here exactly as long as it stayed dead; a single backfill, replay, or one-off write would have made
the loudest key in the dict a retired instrument.

Removing the map entry alone fixes this instance and leaves the next retired domain free to repeat
it. The filter is the seam; the map entry is the instance.

## Current architecture

```text
_*_tick()  ->  _write_prediction_error_node()  ->  node:substrate.<domain>  (FalkorDB orion_substrate)
                                                          |
        _brain_frame_prediction_error_by_domain() reads them back via _PREDICTION_ERROR_DOMAIN_NODE_IDS
                                                          |
                              +---------------------------+---------------------------+
                              |                                                       |
              compute_prediction_error_trend()                        reduce_attention_self_model()
                              |                                          |                    |
                    predicted_shift (was UNFILTERED)      prediction_error_confidence   confidence
                                                          (ACTIVE_INFERENCE_DOMAINS)    (branch-gated)
```

`config/field/orion_field_topology.v1.yaml`'s `capability:transport` edge had **already** migrated to
`node:substrate.bus_synaptic` and needed no change — the EWMA successor was in place there. Verified
directly: zero references to `node:substrate.transport` remain in that file.

## Files changed

- `services/orion-substrate-runtime/app/worker.py`: map entry removed; the stale "kept, not deleted"
  comment block replaced with what actually happened.
- `orion/substrate/prediction_error.py`: `transport_prediction_error()` deleted, replaced by a
  comment recording why the "keep it, replay might want it" rationale did not hold.
- `orion/substrate/attention_self_model.py`: `predicted_shift` argmax filtered; two docstring claims
  that were true-and-load-bearing ("remains present in `prediction_error_by_domain` when a caller
  supplies it") corrected.
- `services/orion-substrate-runtime/tests/test_prediction_error_domain_map.py` (new): the gate.
- `orion/substrate/tests/test_attention_self_model.py`: two tests for the filter.
- `services/orion-hub/scripts/attention_organ_routes.py`: `transport` removed from
  `KNOWN_PREDICTION_ERROR_DOMAINS` (review finding 4).
- `scripts/analysis/measure_ast_hot_reducer.py`: the SECOND copy of the domain map, plus a stale
  4-domain claim in its generated report (review finding 2).
- `services/orion-substrate-runtime/README.md`: retirement section completed and corrected.
- `services/orion-proposal-runtime/.env_example`: hop 0 flag flipped, with rationale.

## Schema / bus / API changes

- Added: none.
- Removed: `transport_prediction_error()` (public symbol in `orion.substrate.prediction_error`,
  zero callers).
- Renamed: none.
- Behavior changed: `prediction_error_by_domain` / `prediction_error_trend_by_domain` no longer carry
  a `transport` key. `predicted_shift` can no longer name a non-`ACTIVE_INFERENCE_DOMAINS` domain.
- Compatibility: consumers iterate these dicts rather than indexing them, so a missing key is a
  smaller set, not a `KeyError`. `services/orion-hub/scripts/attention_organ_routes.py` drops
  `transport` from `KNOWN_PREDICTION_ERROR_DOMAINS` (review finding — see below).

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- Behavior changed: `ORION_METACOG_HOP_PROPOSE_ENABLED` `false -> true` in `.env_example`.
- `.env_example` updated: yes.
- local `.env` synced: **yes, done** — `services/orion-proposal-runtime/.env` line 21 edited directly
  to `true`. `scripts/sync_local_env_from_example.py` does not overwrite an existing value, so this
  was a hand edit, verified after. `.env` confirmed still gitignored (`git check-ignore`).
- `docker-compose.yml` inline default stays `:-false`. Intentional: the compose default is the
  ships-off posture for a fresh operator; the live `.env` carries the enabled value.
- skipped keys requiring operator action: none.

### Why the hop 0 flag is safe to flip now

It shipped off behind "flip only after a real, non-degenerate candidate is observed post-deploy" —
which was **unsatisfiable as written**: the converter only runs when the flag is on, so nothing could
ever be observed while it stayed off. Zero `cognitive_hop` candidates existed in
`substrate_proposal_frames` across its whole history, confirmed by query.

What actually makes it safe:

- `required_policy_gate="operator_review"` is unconditional in `orion/metacog/proposal.py` — worst
  case is a noisy proposal queue, never autonomous action.
- The arena already runs a live external producer under the identical contract: `reverie_thought`,
  518 candidates in a 3h window.
- The converter fails open — `worker.py` wraps it so a metacog read cannot break proposal generation.

**Measured post-deploy, and the answer is: it does not fire, and cannot on this input.**

The flag is confirmed live in the container (`ORION_METACOG_HOP_PROPOSE_ENABLED=true`), 90 proposal
frames were built in the first 5 minutes, `0` carried a `cognitive_hop` candidate, and `0`
`metacog_hop_candidate_failed` exceptions were logged — so the converter runs cleanly and returns
`None`, rather than erroring.

Replaying the **real** reducer over the **real** 7-day series (not a fixture):

```text
readings                45          (repair_pressure_appraisal_log, 7d, ~6.4/day)
sustained trend ticks    0          <- across the entire window
latest                  z=-0.526, consecutive_elevated=0, baseline n=45
distinct values          6 of 45
modal value             34 of 45    (0.08706577244027125, the text-fallback constant)
```

**76% of the series is one identical constant.** This is the metric quality gate's step-4 failure in
its degenerate form: the input is not a signal with a rest state, it is a constant with occasional
excursions. `is_sustained_trend` requires 3 consecutive *escalating* elevated ticks; a series that
sits on one value and steps off it briefly cannot produce that, and the flat run also drives the EWMA
variance toward its floor so the excursions that do occur score absurd z-scores rather than
meaningful ones.

This is recorded as a **result, not a caveat**. Hop 0 is correctly wired and provably exercised
against live data; what it lacks is an input series capable of expressing a trend. The next patch on
this arc should be choosing/constructing that series, not more hop plumbing. Leaving the flag on is
still right: it is now a live, zero-cost observation point, and the alternative (off) is what made
this measurable only after five days of it being unobservable.

## Live data change attempted (CLAUDE.md section 14) — and it FAILED

Snapshot taken first (`/tmp/retire-substrate-transport-node/before_snapshot.txt` — full properties
plus relationship count). Single node, zero relationships, dead producer, well under the threshold
that requires asking.

**The delete does not stick.** Three attempts, each reporting `Nodes deleted: 1`, each undone:

| # | method | result |
|---|--------|--------|
| 1 | plain `DELETE` while everything running | back within a minute, `activation` still ticking `0.166125 -> 0.166074 -> 0.166064` |
| 2 | `stop substrate-runtime` -> delete (`count(n)=0` verified while down) -> `up -d` | back anyway |
| 3 | delete on running system, polled every 2s | **resurrected after 6 seconds** |

What that rules out:

- **Not the dynamics loop.** `SUBSTRATE_DYNAMICS_TICK_INTERVAL_SEC` defaults to 30s; this took 6s.
- **Not `FalkorSubstrateStore`'s in-process cache alone.** That cache dies with the process, and
  attempt 2 deleted the node while the process was down.
- **Not a fresh write.** The restored node carries `activation: 0.166125` — the *exact* pre-delete
  value, not a decayed one — with `recency_score: 0.0` and `observed_at: 2026-07-24T21:55:26Z`.
  Something replays a frozen 7-day-old snapshot verbatim.
- **No log line.** `substrate-runtime`, `recall`, `attention-runtime`, `field-digester`,
  `graph-compression` all silent across the resurrection window.

`prediction_error` is the one property that does not come back — it is in
`EXTERNALLY_OWNED_METADATA_KEYS`, excluded from the `MERGE` write, which is itself a clue about the
writer's shape. `salience: 0.556078` does come back.

**Status: `UNVERIFIED`, and materially bigger than this PR.** If a substrate concept node cannot be
deleted, no substrate node can be retired from the graph at all — only from its readers. Chasing it
further inside a retirement PR would be the wrong place; it needs its own investigation. No denylist
was added either: that would paper over a writer nobody has identified yet.

The four code changes in this PR are unaffected — they remove every *reader* of the node, which is
the part that was actually feeding cognition.

## Tests run

```text
$ pytest orion/substrate/tests/test_attention_self_model.py -q
48 passed

services/orion-substrate-runtime$ pytest tests/test_prediction_error_domain_map.py -q
3 passed

services/orion-substrate-runtime$ pytest tests -q --ignore=tests/test_grammar_consumer_integration.py
23 FAILED/ERROR lines
```

Those 23 are **pre-existing**, established against a detached scratch worktree at `origin/main`:

```text
origin/main : 23 FAILED/ERROR lines
this branch : 23 FAILED/ERROR lines
diff        : one captured log line's line number (worker.py:540 -> :555), from added comments
```

Red-before-green, both new test files run against `origin/main`:

```text
test_prediction_error_domain_map.py
  FAILED  ... only in the node map: ['transport']
  FAILED  transport_prediction_error() is back in orion/substrate/prediction_error.py

test_attention_self_model.py
  FAILED  TestPredictedShift::test_retired_domain_cannot_win_the_argmax
  FAILED  TestPredictedShift::test_only_retired_domains_yields_no_prediction
```

## Evals run

```text
No eval harness exists for services/orion-substrate-runtime or orion/substrate.
```

Flagged, not claimed.

## Docker/build/smoke checks

See "Restart required". Deploy evidence appended post-merge.

## Review findings fixed

Code review ran in a subagent per CLAUDE.md section 12. Six material findings, all addressed.

- **Finding 1 (HIGH): the live node deletion did not stick; the claim was false.** Review checked
  the live rail and found `node:substrate.transport` present with `activation` still ticking.
  - Fix: the claim is **retracted**. Two further deletion attempts (including one with
    `substrate-runtime` stopped) also failed; the third measured a 6-second resurrection. Documented
    as an open defect with the evidence, in both the README and this report, rather than restated as
    done. Status `UNVERIFIED`.
  - Evidence: `Nodes deleted: 1` three times; `count(n)=1` after each; restored `activation`
    `0.166125` byte-identical to the pre-delete snapshot.
- **Finding 2 (MEDIUM): a second, ungated copy of the domain map.**
  `scripts/analysis/measure_ast_hot_reducer.py` — the harness `attention_self_model.py`'s own
  docstring names as this reducer's validation path — still listed `transport`, so it would have
  averaged six values where production averages five for the same tick.
  - Fix: entry removed, its now-inverted rationale rewritten, and the gate **extended to parse that
    file too**, so the class of drift is closed rather than this one instance. Its stale generated
    report line (a 4-domain claim printed next to a 5-domain number, wrong since 2026-07-25) fixed
    in the same pass.
- **Finding 3 (MEDIUM): the new gate had a one-line bypass.** Review demonstrated that
  `_PREDICTION_ERROR_DOMAIN_NODE_IDS["node:substrate.transport"] = "transport"` on a later line was
  invisible to the AST parser — both tests passed with the retired domain live again. Separately, a
  type annotation turned the node into `ast.AnnAssign` and made the gate blame a missing constant.
  - Fix: `ast.walk` over both assignment forms, plus explicit rejection of subscript assignment,
    `.update()`, `.setdefault()`, `.pop()`. Helpers now raise `AssertionError` rather than
    `pytest.fail()` **specifically so the bypasses are themselves testable** — `pytest.fail` raises a
    `BaseException` subclass that `pytest.raises(Exception)` does not catch, so a teeth-test written
    against it would have silently proven nothing.
  - Evidence: new `TestTheGateHasTeeth` reproduces all three shapes; 7 passed.
- **Finding 4: Hub `KNOWN_PREDICTION_ERROR_DOMAINS` still listed `transport`.** Either state is
  wrong — "still ticking, still readable elsewhere" while the node exists, or a permanent red
  "no node" badge once it does not.
  - Fix: removed. `harness_closure` deliberately stays: it is excluded from the aggregate but has a
    live producer, which is what that heading is for.
- **Finding 5: a docstring added by this commit overclaimed** — "`predicted_shift` filters to this
  constant like every other consumer". Not true: the branch-gated `confidence` still iterates the
  raw dict.
  - Fix: rewritten to name that consumer as the deliberate exception.
- **Finding 6: undisclosed level-shift in branch-gated `confidence`.** Dropping the frozen `0.556`
  term shrinks the denominator 6 -> 5; future rows in that branch read ~0.09 higher for identical
  substrate state. 25 persisted rows used the 6-domain basis, all before 2026-07-29T05:01:06Z.
  - Fix: disclosed at the constant's definition and here. Not suppressed — the basis string
    self-documents which domain set produced each value.
- **Finding 7: the PR report was untracked.** Fixed; committed with the code.

Review also verified clean, independently: the `transport_prediction_error()` deletion is safe (zero
importers, no star-import, no `getattr`/`importlib` lookup, both named analysis scripts confirmed to
mention it only in prose); no `KeyError` risk (every reader iterates, none indexes `["transport"]`);
and the env flip cannot auto-dispatch — traced through four independent layers
(`required_policy_gate`, `policy/evaluator.py`'s decision branches,
`execution_dispatch_policy.v1.yaml`'s `blocked_policy_decisions`, and the absent
`proposal_kind_to_cortex` route).

## Restart required

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-proposal-runtime up -d --build
```

`orion-proposal-runtime` is required for the flag flip to take effect — the value is read at boot.

## Risks / concerns

- **Severity: should-know. Hop 0 fires zero times on its current input, measured not predicted.**
  Replayed over the real 7-day `repair_pressure_appraisal_log`: 45 readings, 6 distinct values, 34 of
  them one identical constant, `0` sustained-trend ticks. The wiring is proven; the input series is
  not capable of expressing a trend. Follow-up is a better series for hop 0, not more hop plumbing.
- **Severity: HIGH, open. A substrate concept node cannot be deleted from the live graph.**
  Measured three ways above; unexplained. This blocks the physical half of any node retirement,
  transport's included. Needs its own investigation patch — the first question is which process
  replays a byte-identical frozen snapshot within 6s while logging nothing.
- **Severity: note. `node:substrate.harness_closure` is the same shape of zombie** — 7 days stale at
  a hardcoded `0.65`. It is already excluded from the domain map deliberately, so it does not
  contaminate the reducer, but its producer looks dead too. Not touched here; scope was transport.
- **Severity: note. `orion_metacog` still has no consumer** (~3,050 rows/day, of which ~1,275 come
  from the still-level-triggered `rpc_health` transport branch). Unrelated to this patch, still open.

## PR link

<pending>
