# Phase 5 — signal semantics: provenance, window, commensurability

**Mode:** Was design/scoping. **R1–R5b are all shipped and merged. The
fourth-axis question (R6) is resolved: investigated, no live victim found,
no rung, no code.** See "Open decisions". Kept as the arc's record rather
than closed, because most of what it now says was learned by building it.

**Date:** 2026-08-13, revised same day, 2026-08-15 with shipped status and
measured outcomes, 2026-08-16 once R5a's watch shipped, 2026-08-19 once R5b
(the actual guard) shipped and was live-verified, and again 2026-08-19 once
the R6 investigation closed — see "What this revision retracts" and "What
building it changed".

## Status

| rung | state | PR | what it actually cost |
|---|---|---|---|
| R1 provenance | **merged** | #1631 | review retracted `winner_is_unique` and three blind spots |
| R2 regime readout | **merged** | #1633, #1657 (Hub), #1682 (timestamps) | review caught **three false claims** in the first version |
| R3 commensurability | **merged** | #1638, #1654 + hourly cron | detector had drifted from production; decay-aware fix shipped uncommitted once |
| R4 definition-change alert | **merged** | #1666, #1680 (tooling), #1677 | shipped a **fabricated alert** in its own lock file |
| R5a feedback-credit watch | **merged** | #1686, #1694, #1700 | **first version measured the wrong thing entirely** -- 0 of 454 flagged windows over 6,000 ticks were real decay; rebuilt on write-evidence, not shape |
| R5b feedback-loop guard | **merged** | #1709 | review caught a duplicate-entry bug and an untested negative-credit path; live-verified on real production traffic post-deploy |

All five rungs shipped. Every one was materially wrong on first submission and
was corrected by review or by live measurement — recorded below rather than
smoothed over, because the pattern is the finding. One rung (R6, the fourth
axis) is unbudgeted and still open -- see "Open decisions".

## What this revision retracts

The first version of this doc framed phase 5 as **liveness** and treated the
four metric surfaces being different as a *scoping obstacle*. Both were wrong.

- "Liveness" is not the problem. Every metric in the incident ledger below was
  live. They moved, they varied, they were wrong anyway.
- The surfaces differing is the **design premise**, not an obstacle. A field
  channel, an inner-state scalar, an organ signal, and a bus channel are
  different kinds of thing and get different treatment by construction. The
  first draft presented that as a blocker to be resolved. It isn't one.
- It also proposed a second answer to bus traffic. Bus traffic is already
  answered by the bus synaptic graph over `orion-bus-mirror` actuals, which
  sees multi-hop turns rather than unweighted per-tick counts. Building a
  parallel `traffic` verdict would be a duplicate mechanism.

## The one problem

**A number reaches a consumer with no record of what it is.**

Three facts are missing at the point of use, and every incident in the record
is the absence of exactly one of them:

| axis | the missing fact | failure it produces |
|---|---|---|
| **Provenance** | which producer actually wrote this, and when | a decayed or dead value reads as a calm real one |
| **Window** | what interval and what transform this summarizes | "near max but steady" is indistinguishable from "spiking" |
| **Commensurability** | is this on the same scale/semantics as what it is combined with | a low-resolution input silently dominates a merge or a ranker |

This is not a taxonomy. Each axis has a mechanical detector and a measured
instance, below.

## Incident ledger, mapped

| incident | axis | status |
|---|---|---|
| `bus_synaptic_prediction_error()` permanent ~0.27 floor (`mean(\|z\|)` rests at `sqrt(2/pi)`, not 0) | Window | fixed; live min now 0.0039 |
| `node:substrate.route` decayed by 0.92/tick for 48h, read as calm | Provenance | fixed |
| `transport_prediction_error()` excluded from one consumer, still winning budget slots in a generic one | Provenance | retired |
| `perception_staleness` wired as a topology edge source, produced by nothing → fabricated `pressure=0.0/confidence=1.0` | Provenance | fixed 2026-08-13 by the perception P4 work; edge now maps `prediction_error` |
| `thermal_pressure` (18 distinct) beating capability `pressure` (1,325 distinct) on 91.76% of ticks | Commensurability | fixed by deleting one routing entry |
| **`node:substrate.codebase` dominating the merged `prediction_error` channel** | Commensurability | **FIXED** — merge staleness rule, 2026-08-14. `codebase` went 66.6% → 0.9% of merge wins; distinct values 98 → 564 on matched 6,000-tick windows; median 1.0000 → 0.1911 |
| `resource_pressure` reading calm during a producer outage, crediting the in-flight action with success | Provenance | **FIXED** — R5b guard, 2026-08-18 (#1709), live-verified |
| "very busy at near max but steady state, so actually peaceful" is not expressible | Window | **OPEN — no mechanism exists** |

Seven of eight are fixed. Every one was found **by hand** — by a person or a
review noticing. None was found by a gate. That was the thing to change, and
it has now changed at least once: on 2026-08-14 the R3 cron gate and the R4
definition gate each fired on main within hours of landing, on real edits
nobody had flagged (`field_coherence_warning`'s subnormal merge; a consumer
removed from `orion:spark:signal`). Both were true positives.

## Measured evidence (2026-08-13, live)

Source: `substrate_field_state`, 113,190 rows spanning 2026-08-11 → 2026-08-13.
**That is a 2.5-day window** (the corpus restarts at the disk-death of
2026-07-23), so distribution claims below are scoped to it and are not
long-run. Sample used: the most recent 20,000 ticks = 11.4h at a 2.04s median
tick.

### Commensurability: the merged `prediction_error` channel

`collect_field_channel_pressures()` merges every `PRESSURE_CHANNEL` by `max()`
across all sources. For `prediction_error` that is a max over 12 nodes:

| source | wins the merge | its own distinct values / 20,000 ticks |
|---|---|---|
| `node:substrate.codebase` | **54.2%** | **5** — effectively the constant 0.3357 |
| `node:substrate.execution` | 41.7% | 1,208 |
| `node:substrate.vision` | 4.1% | 4 (0.0 with 1.0 spikes) |
| `node:substrate.biometrics` | **0%** | 1,188 |
| `node:substrate.bus_synaptic` | **0%** | 341 |
| `atlas`, `circe`, `athena`, `prometheus`, `substrate.transport` | 0% | 1 each (permanent 0.0) |

Consequences:

- The channel has a **hard floor of 0.3357**, set by a near-constant from one
  node. It cannot read calm below that. Structurally the same defect as the
  0.27 floor, but produced by the *merge* rather than by a formula — so a
  formula-level review would never find it.
- Over the last 600 ticks (~20 min) it is **1 distinct value, exactly 1.0**.
- The two richest signals in the set (1,188 and 341 distinct) contribute
  **nothing, ever**.

`max()` across incommensurable sources does not select the most informative
one. It selects the highest-scaled one.

### Provenance: the discarded dict

`collect_field_channel_pressures()` already returns
`tuple[dict[str, float], dict[str, str]]` — values *and* a provenance dict
naming which source won each channel. `field_pressures()` discards it:

```python
channel_pressures, _provenance = collect_field_channel_pressures(field)
return map_channels_to_dimensions(channel_pressures)
```

The seam exists and is computed every tick. It is thrown away one line before
the consumer.

### Provenance: the feedback-loop trap

`orion/field/pressure.py:100-108` records, and a code review confirmed, that if
`services/orion-biometrics` goes quiet, decay drives every remaining input
toward 0, `resource_pressure` reads calm, and because
`config/feedback/feedback_policy.v1.yaml` lists `resource_pressure: decrease`
under `positive_delta_channels` (**verified present on main today**), the
in-flight action is credited with a positive outcome for a sensor outage.

Tracked to PR #1554, which **merged as docs only**. The guard was never built.

Checked for it directly: **no geometric-decay runs in the last 20,000 ticks** —
every producer stayed live across the retained window. So this is a **latent
trap, not a currently-firing bug**. It cannot be caught by monitoring, because
by the time it fires it has already written a false reward.

### Window: nothing expresses regime

Prototyped level / dispersion / drift / saturation over a declared window
against real ticks. It separates cleanly (e.g. `memory_pressure` level 0.813
with dispersion 0.001 — loaded and steady; `gpu_pressure` dispersion 0.258 and
touching both 0.0 and 1.0 — volatile and rail-saturated). No such reading
exists anywhere in the system today; consumers get one scalar.

Note the units trap this exposes: a "600-tick window" is 20 minutes at the
current 2.04s cadence, and nothing anywhere writes that down. **Windows get
declared in seconds.**

## Treatment by signal kind

Different kinds get different mechanisms. That is the point, not a compromise.

| surface | URNs | what it gets | why |
|---|---|---|---|
| `field_channel` | 38 | full treatment: provenance + regime + commensurability | real per-tick history in `substrate_field_state`; feeds the rankers |
| `inner_state` (substrate-runtime) | 37 | provenance + regime | per-signal tables exist; this is what feeds the autonomy rankers, so it is where mixing does the most damage |
| `organ_signal` | 252 | **definition-change alert only** | no persistence (in-memory window only; `substrate_organ_emissions` has 1.65M rows but one `organ_id`, and it is not in `ORGAN_REGISTRY`). Persisting it is a producer change, out of scope. Alert when someone adds to it. |
| `bus_channel` | 261 | **definition-change alert only** | bus synaptic over `orion-bus-mirror` actuals already owns traffic, including multi-hop. Alert when someone edits the defs. |

## Roadmap

Ordered. Each rung is independently shippable and independently useful.

### R1 — provenance survives the merge

Thread the already-computed provenance dict through `field_pressures()` to the
consumer. No new computation, no new schema concept, no behavior change.

*Acceptance:* for any dimension, name the source that won each contributing
channel this tick, from real stored state. **MET** (#1631). Existing functions
became thin wrappers over the provenance-carrying ones so the two can never
drift. Review retracted a `winner_is_unique` claim and three blind spots.

### R2 — regime readout over declared windows

Level, dispersion, drift, saturation as **separate** readings per channel,
windows declared in seconds. Surface on the lineage card and in `--json`.
Reuse `orion/bus/ewma.py::compute_ewma_update` and
`classify_channel_series()`; check `orion/substrate/prediction_error.py`,
`orion/metacog/trend_reducer.py`, and the phi autoencoder v2 running on field
signals before writing any new statistic — several already exist and this must
not add a sixth.

*Acceptance:* "near max but steady" and "volatile" produce different readouts
for two real channels. A saturated channel reports saturation rather than a
level. **MET** (#1633), live on the Hub glossary panel (#1657).

Cost, recorded because it is the point of this doc: review found **three false
claims** in the first version — a "derived" dispersion threshold that was
declared (206 of 208 windows had a channel inside the supposedly empty region);
a claim that `collect_field_channel_pressures()` polarity-corrects, which it
does not, so `confidence` at a healthy 0.8678 read `loaded_volatile` in 159 of
208 windows; and "mutation-checked", where I had hand-picked five mutants I
expected to fail. A real 31-mutant harness killed 7 and left 23 alive.

A second defect surfaced only in production (#1682): `channel_regime()` had
always accepted `updated_at` and the Hub never passed it, so all 38 channels
used the value-ratio fallback — blind exactly in the subnormal range that turned
out to be everywhere. Fixing it took **three** rules; the first two were both
wrong, and the second shipped a confidently-wrong authoritative verdict from
0.5% stamp coverage, which is worse than the blindness it replaced.

### R3 — commensurability detector

Flag any merge where one source wins >50% of ticks while contributing fewer
than N distinct values, and any consumer combining channels whose declared
window semantics differ.

*Acceptance:* fires on `substrate.codebase` in the `prediction_error` merge as
it exists today, and would have fired on `thermal_pressure` before the manual
catch. **MET** (#1638), ratcheted and gated (#1654), running hourly by cron.

Two things it taught. The detector had **drifted from production**: its merge
predated the staleness rule, so its first gate run reported `prediction_error`
dominated at 72.4% *after* production had stopped merging that way — fixed by
sharing `_stale_node_channels()`. And decay trails inflate distinct-value
counts, since 0.92/tick generates a fresh float every tick; `node:circe`'s 943
"distinct values" for `reasoning_load` were 99.8% decay steps.

First live catch on main, 2026-08-14: `field_coherence_warning`, where
`node:circe` won 100% of 6,000 merges on **one** informative distinct value.
Investigating it found the channel frozen at subnormal `3e-323` with a decay
ratio of exactly 1.0 — `0.92 × 3e-323` rounds back to itself, so it is
mathematically frozen forever.

### R4 — definition-change alert

Diff-triggered notification when an agent edits bus channel defs, organ defs,
or topology channel maps. Answers "tell me when someone starts messing in
there" without a verdict column. Subsumes what the first draft called slice C,
which was a narrow static assert on one config file and is already green.

*Acceptance:* editing a channel def in a PR surfaces the change to Juniper.
**MET** (#1666). Diffs the *resolved* definition layer (595 URNs), not the YAML
text, so a reordered list produces nothing and one real edit produces one named
line. The alert IS the lock diff: the gate is red until `--update` re-locks,
and re-locking is what writes the sentence.

First live catch, hours after landing: a consumer removed from
`orion:spark:signal`. True positive on a **correct** change — the gate's job is
to say it happened, not that it was wrong.

Two failures worth keeping. The first lock shipped **two fabricated
high-severity alerts**, residue of a mutation test that locked a mutated state,
reverted, and re-locked — so the block recorded the revert as the change. The
feature shipped carrying exactly the misinformation it exists to prevent. Fixed
by deriving the block from the merge base and having the gate recompute it, so
the sentence is a constraint rather than a convention. And the gate could never
be green on main (HEAD *is* the merge base there); caught by another session,
because I only ever exercised it from a branch.

R4 also exposed two defects in the **semantic layer it depends on** (#1680):
`visit_Subscript` ignored AST context, so a channel's own producer appeared in
its blast radius; and generic consumers — a `max()` over a whole vector, naming
no channel — were invisible, so `field_coherence_warning` read as
zero-consumer while attention read it every tick. Acting on that under "kill
means kill" would have deleted a live producer.

### R5a — the precondition watch (this roadmap) **MET** (#1686, #1694, #1700)

Report-only: can a credited dimension currently be fooled by a producer
outage right now? Not the guard -- the measurement a real guard needs first.

*Acceptance:* name, per credited dimension, whether there is write evidence
behind its current value inside the feedback loop's own window. **MET**, but
not on the first attempt -- the first version (#1686) reused
`classify_producer_liveness()`, a shape/monotonicity classifier, and a
post-merge review found it measured the wrong thing entirely: **0 of 454
flagged windows over 6,000 live ticks contained a single actual decay step.**
A genuine 0.9 -> 0.0 real fix read as "only the decay loop touched this" --
the exact opposite of the module's purpose. Rebuilt (#1700) on two
write-evidence signals instead: real node-write timestamps (reusing R2's own
tested mechanism) for node-vector-sourced channels, and per-tick
diffusion-contribution freshness (verified live against
`apply_diffusion`'s actual source, not assumed) for capability-routed ones. A
second review of the rebuild found one more real bug -- the per-channel
mechanism choice was a majority vote over the whole batch, so a real outage
in a channel's numerically-minority sourcing type was invisible to both
signals -- fixed before merge. Live result: 0 findings on the most recent
6,000 ticks, down from 3 fake findings on ~29 consecutive hourly runs.

Full account: `orion/field/credit_integrity.py`'s module docstring and
`docs/superpowers/pr-reports/2026-08-16-credit-integrity-rebuild-pr.md`.

### R5b — the guard itself **MET** (#1709)

A staleness guard on the dimension so a decayed-to-calm reading cannot be
credited as a positive outcome. Changed a learning loop, so it went through
proposal mode per CLAUDE.md §0A and shipped once Juniper said "build it".

*Acceptance:* an unbacked or stale reading is withheld from credit in both
directions, never silently dropped. **MET.** New primitive
`channel_write_backed()` in `orion/field/credit_integrity.py` reuses R5a's
exact two mechanisms (node-vector timestamp freshness, capability-routed tick
freshness) -- no new heuristic. `orion/feedback/builder.py` gates
`positive_delta_channels` through it; an unbacked reading is recorded in a new
`FeedbackFrameV1.withheld_evidence` field rather than credited or silently
dropped, so it stays inspectable. Rollback lever:
`FeedbackPolicyV1.write_evidence_guard_enabled` (default on).

Honest scope note, found by investigation before building: no live
action-value/ranker consumer of the feedback-credit signal exists today, so
this is not (yet) blocking RL-style reward hacking. The real, live consequence
of the trap this closes is corrupted pattern/expectation building in
`orion/consolidation/motif.py`, which reads `outcome_status` /
`negative_evidence` directly.

Review caught two real bugs before merge: `channel_write_backed()` was called
twice for `reliability_pressure` on the same tick, producing a duplicate
`withheld_evidence` entry (fixed by threading the already-computed verdict
through instead of recomputing); and the negative-credit direction had no
test coverage (fixed with a regression test). Mutation testing separately
caught a coverage gap where a `None` (unmapped-this-tick) verdict fell through
as if credited.

Live-verified post-deploy, not just tested: confirmed the new code path
actually loaded in the running container, then queried
`substrate_feedback_frames` in production Postgres directly and confirmed
`withheld_evidence` is round-tripping on genuinely fresh post-deploy frames
(`tick_f22ec0fdb8fe`, empty list that tick since all deltas were flat 0.0 --
correct, not a null result). Deploy verification also found and fixed an
unrelated, pre-existing live bug: `services/orion-feedback-runtime/.env` had
drifted from its own `.env_example` and every sibling service's `.env`,
running `ORION_BUS_URL=redis://bus-core:6379/0` instead of the mandated
tailscale address -- functionally fine inside the docker network, but a
direct violation of CLAUDE.md's explicit bus-URL rule and fragile to network
changes. Fixed and redeployed; confirmed via `docker inspect` and a direct
Redis reachability check.

Full account: `orion/field/credit_integrity.py`'s module docstring and
`docs/superpowers/pr-reports/2026-08-18-feedback-write-evidence-guard-pr.md`.

## Non-goals

- One classifier across all four surfaces.
- Any declared or persisted verdict column. Verdicts stay computed.
- Fixing any metric this finds. Detection is its own patch; each fix is another.
- Inferring theory anchors from data. A fabricated rest-point is worse than none.
- Changing the merge, the feedback policy, or any ranker in R1-R4.

## What building it changed

Answers to the questions this doc originally asked, plus what it did not
anticipate.

**Order (resolved):** R1 → R2 → R3 → R4, as written. R1's provenance dict
turned out to be load-bearing for all three later rungs — R3's detector, R4's
routing deltas, and R2's Hub timestamps all read it — so doing R3 first would
have meant building it twice.

**The named problem was mis-diagnosed, and the fix was elsewhere.** This doc
frames the `prediction_error` domination as *commensurability* — a coarse input
on the wrong scale. It is not. `codebase_prediction_error()` is already
correctly z-scored. The defect is **staleness**: `prediction_error` is absent
from `NODE_DECAY_CHANNELS`, so a slow producer's value persists byte-identical
and wins `max()` for thousands of ticks. The fix was a merge staleness rule, in
the provenance axis, not the commensurability one. The detector R3 built is
still right and still fires; the *instance* that motivated it belonged to a
different axis.

**A fourth axis showed up, unbudgeted: can this metric express rest?** Not
provenance, window, or scale — whether the number has a reachable calm state at
all. Three live instances, one pre-existing and two found here:
`bus_synaptic_prediction_error`'s permanent 0.27 floor; `field_coherence_warning`
frozen at subnormal `3e-323` where decay is mathematically a no-op; and
`check_field_coherence()` returning only nodes with `s > 0.0`, so a node that
recovers is never written and decays forever — **"calm" is unrepresentable for
it, and "sensor dead" is indistinguishable from "genuinely quiet."** R2's
`refresh_state` addresses the last of these only by accident.

**Detection tooling needs the same scrutiny as what it detects.** R4 depends on
the metric semantic layer, and that layer was wrong in both directions at once:
counting a channel's own producer as its blast radius, and missing every
generic `max(vector)` consumer. It reported "zero consumers" for a channel
attention reads every tick. A retirement made on that output would have killed
a live producer — the exact incident class this roadmap exists to prevent,
committed by the instrument.

**R5 timing (resolved by evidence):** hold, as written, and the wait paid. The
guard's trap — `resource_pressure` reading calm during a producer outage — is
now not hypothetical: an outage was observed 2026-08-14, and the fleet's
`expected_offline_suppression` mechanism, which exists for exactly that case,
is unreachable because every node in `config/biometrics/node_catalog.yaml` is
`expected_online: true`. **0 of 126,983 stored ticks** carry a nonzero value.

**Decision #1 below, resolved 2026-08-16 -- not by fixing that config surface
first.** Direct call: "offline suppression is binary when shit is variable...
not sure why we are so blocked on this dumb concept. circe is up sometimes,
not always and its variable. live with it." A binary expected-online flag
cannot describe a variably-up node, and R5a's guard never actually needed to
know whether a node was *expected* offline -- only whether a reading was
*observed* this window. R5a (above) was built directly against real write
evidence instead, with no `expected_offline_suppression` dependency at all.
That config gap is still real and still unfixed, but it is no longer on R5's
critical path.

## Open decisions for Juniper

1. ~~**R5 scope.**~~ **RESOLVED 2026-08-16, see above.** R5a shipped without
   waiting on `expected_offline_suppression`.
2. ~~**The fourth axis.**~~ **RESOLVED 2026-08-19 -- no rung, no code, park it.**
   Investigated properly (`scripts/check_metric_lineage.py` blast radius +
   direct Postgres history, then a full adversarial pass after the first
   pass's headline instance turned out to be wrong) before deciding, per this
   doc's own "measure before building" discipline.

   **The mechanism is real.** `apply_decay()`
   (`services/orion-field-digester/app/digestion/decay.py`) has no floor on
   any of its 28 `NODE_DECAY_CHANNELS` entries -- `vec[ch] = vec[ch] *
   decay_rate` forever, and float64 repeatedly multiplied by 0.92 hits a
   fixed point in the subnormal range before reaching true 0.0. Proved this
   is the only way `field_coherence_warning` reaches `3e-323`: its one write
   site (`worker.py:275`) can only ever write `round(hits/applicable, 4)`, a
   ratio of small integers rounded to 4dp -- never a subnormal float directly
   -- confirmed with a whole-repo grep for any other writer, not just the two
   modules already in view.

   **The instance that motivated it was wrong.** The first pass presented
   `node:circe`'s frozen `field_coherence_warning` as a live, currently-active
   defect. It wasn't checked whether circe was actually up. It is not: every
   real channel on `node:circe` (`thermal_pressure`, `cpu_pressure`,
   `gpu_pressure`, `staleness`, seven others) stopped writing within the same
   ~2-minute window, 10+ hours before the check, while `node:athena` /
   `node:atlas` / the `substrate.*` nodes all had writes within the prior
   1-2 minutes -- confirmed with a corrected *per-node* query after the first
   version of that query wrongly merged timestamps across all nodes and
   couldn't actually distinguish "circe specifically is down" from "the whole
   pipeline stalled". Circe going offline is already-documented expected
   behavior for that node (see R5 timing section above, direct quote:
   "circe is up sometimes, not always... live with it"), not an anomaly.

   **No live consumer was found to actually be fooled.** Checked the 5
   magnitude-sensitive sites out of the channel's 17 discovered generic
   whole-vector consumers (`check_metric_lineage.py --generic-consumers`):
   `pressure_delta()` (feeds R5b) treats a subnormal `before` the same as an
   exact-zero one (`after - before` is insensitive to the difference, plus
   its own `1e-6` deadband); `commensurability.py`'s existing
   `DECAY_RATIO_EPSILON` carve-out is unaffected either way; the
   `capability_vectors` decay branch is architecturally exempt in practice
   (its own 2026-07-12 comment: `apply_diffusion()` overwrites every live
   entry the same tick, so nothing decays away from a real value there);
   and Hub's `build_channel_series()` already documents this exact
   phenomenon by name ("blind in the subnormal range where decay stops
   producing a 0.92 ratio") as the reason R2's timestamp-based regime path
   exists -- and every stuck channel checked on circe carries a real
   `node_vector_updated_at` stamp, so all of them ride that already-fixed
   path, not the blind ratio-inference fallback it replaced.

   **What's still genuinely open, not chased:** R2's own docstring states
   only 22-25 of 38 channels get the timestamp fix; the remaining 13-16 still
   use blind ratio-inference and theoretically could misread a subnormal
   freeze as live data. No specific channel has been shown to hit this live.
   If this doc's thesis (a known defect with no mechanism behind it does not
   stay known) applies to itself: this paragraph is that mechanism for this
   one.

## Risk note — tooling, not code

`rg` output in this environment was caught on 2026-08-13 silently replacing the
searched-for identifier with a short token on large result sets, including
inside file paths (`channel_map` → `n`; `config/field/biometrics_lattice.yaml` →
`config/field/n.yaml`, a path that does not exist). Small result sets are
unaffected, so it is intermittent. Suspected source is RTK's ripgrep filter.

This matters to this roadmap specifically: every rung depends on reading
identifiers out of the repo accurately, and this corruption fabricates
plausible ones. Verify symbol and path spellings with a direct file read before
relying on grep output.

**Checked 2026-08-16 whether `~/.claude/hooks/rtk-fcc-gate.sh` already covers
this: it does not.** Read the script directly -- it only decides whether to
run RTK's rewrite hook at all, gated on FCC-vs-interactive session (checks
`ANTHROPIC_BASE_URL`/`ORION_FCC_MCP_TOOL_RESULT_MAX_CHARS`), unrelated to
identifier corruption. The earlier version of this note assumed it was the
same mechanism; it is not. A **third** occurrence, same session that wrote
this revision: a plain `grep -h "^POSTGRES_URI="` call was silently
intercepted and returned RTK's own help/usage text instead of running.

**2026-08-19: root-caused for real, not just observed, and both symptoms
turned out deterministic, not "unpredictable" as this note previously
guessed.** Two distinct bugs, not one:

1. **The `-h` swallow, reproduced live on demand.** RTK's clap-based arg
   parser globally reserves `-h`/`--help` (and `-V`/`--version`) for
   *itself* before the wrapped tool ever sees them. `grep -h`
   (`--no-filename`, an ordinary flag) never reaches real grep -- RTK's own
   help text comes back instead, silently, exit 0. 100% reproducible, not
   intermittent.
2. **The identifier corruption, root-caused via RTK's own public issue
   tracker, not reproduced locally.** [rtk-ai/rtk#1613](https://github.com/rtk-ai/rtk/issues/1613)
   (filed by someone else, open since April, still unfixed as of the
   latest release checked, v0.45.0): RTK's output parser assumes
   ripgrep/grep output is always `path:line:content`, via a naive
   `splitn(3, ':')`. Ripgrep omits the path prefix for a single-file
   search, so real output is `line:content`; when the matched content
   itself contains a colon (a YAML key like `channel_map:` is exactly this
   shape), the parser misassigns fields. Matches the originally recorded
   symptom precisely.

**Gated 2026-08-19.** `~/.claude/hooks/rtk-fcc-gate.sh` now blocks (exit 2,
explains why, points at `--no-filename`) any grep/rg call carrying a bare
`-h`-containing short-flag cluster before RTK's parser can swallow it, and
unconditionally injects `-H`/`--with-filename` into every plain grep/rg
rewrite so #1613's missing-path-prefix precondition can't occur -- purely
additive, safe on every existing call. Not this repo's code (this hook is
global, `~/.claude/`, not version-controlled, not part of this arc's usual
PR/test discipline), so it's recorded here rather than shipped as a normal
patch. Built and tested against 9 cases (blocks, safe-passthrough,
FCC-bypass) in the authoring session; installing it required a permission
grant this session didn't have, so as of this revision it is **written and
verified in isolation, not yet confirmed installed and live** -- check
`~/.claude/hooks/rtk-fcc-gate.sh` for the 2026-08-19 header comment before
trusting that it's active.

Recorded three times before this fix existed, acted on zero, across the
entire arc above -- the standing example of this doc's own thesis, until
now: a known defect with no mechanism behind it does not stay known.
