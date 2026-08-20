# Metric semantic layer: lineage, liveness, and edit-time compliance gates

**Mode:** Design/proposal. Tooling and observability scaffolding, not a
cognition change — but any *fix* to a metric this layer classifies as dead is
proposal-mode territory per CLAUDE.md §0A and needs its own sign-off.

**Date:** 2026-08-12, revised 2026-08-19 once phase 3 (the phase that was
never built) shipped, revised again 2026-08-19 once a scoped slice of phase 5
shipped.

**Status:** phases 1, 2, and 4 shipped 2026-08-12/13. **Phase 3 shipped
2026-08-19** — see its section below for what actually got built. **Phase 5
partially shipped 2026-08-19** — two candidates only (`attention_self_model.v1`
scalar fields, `l7_l11_ladder` throughput); see "Phase 5 (2026-08-19)" below.
The general "any surface, any URN" version remains deliberately deferred —
that per-surface data-source question is still bigger than the two cases
solved here.

## Arsonist summary

CLAUDE.md §0A already contains the correct rule. The "Metric quality gate"
spells out six mandatory steps — trace provenance to real code, check
independence, name a theory anchor, sanity-check live data, search for an
existing mechanism, assess reversibility — and demands they be re-run *every
time*, even for a metric that "seems obviously fine."

That gate is prose in a contract file, enforced by an agent choosing to
remember it. The failure record shows what that buys:

- `bus_synaptic_prediction_error()` shipped with a mathematically permanent
  ~0.27 floor. Caught only by hand-recovering the pre-aggregation statistic.
- `node:substrate.route`'s `prediction_error` decayed to subnormal because a
  generic staleness loop multiplied it by 0.92 per tick for 48+ hours. Looked
  identical to "genuinely calm at zero" until the exact geometric ratio was
  checked by hand.
- `transport_prediction_error()` was excluded from one consumer's active-domain
  set as known-dead, yet its tick kept running and its node kept winning real
  budget slots in `endogenous_curiosity.py` — a generic consumer nobody checked.
- A recorded session claimed a decay-artifact check that was never run, while
  the disproving 0.92 ratio sat in the data.

Every one of these is the same shape: **a consumer was built or modified
against a metric whose real upstream behavior nobody traced.** The gate did not
fail because it was wrong. It failed because nothing mechanically ran it, and
nothing put the answer in front of the agent at the moment of the edit.

The fix is not another taxonomy of metric names. Five registries already name
things. The fix is a **join across what already exists**, a **computed**
liveness verdict instead of a declared one, a **mechanically discovered**
consumer list, and a gate that fires **at edit time, in the agent's context**,
not in a doc the agent won't open.

## Current architecture

Five registries exist today. Each is real and load-bearing. None of them join.

| Registry | Granularity | Count | Carries | Gate |
|---|---|---|---|---|
| `orion/bus/channels.yaml` | bus channel | 261 | producers, consumers, `schema_id`, stability, since | `check_bus_channels.py`, `check_single_consumer_channels.py` |
| `orion/schemas/registry.py` | schema class | hundreds | `SchemaRegistration(model, kind)` name→class | `check_schema_registry.py` |
| `orion/inner_state_registry.py` | inner-state signal | 13 | producer service, cadence, composition status, cognition consumers | `check_inner_state_registry.py` |
| `orion/signals/registry.py` | organ | 30 | signal kinds, canonical dimensions, causal parent organs, bus channels | none found |
| `config/field/field_channel_glossary.v1.yaml` | field channel | 38 | level, category, meaning, `self_state_dimension`, `evidence_dimension` | Hub panel + live classifier |

Plus 11 `scripts/check_*.py` deterministic gates and 5 `scripts/platform/audit_*.py`
auditors, wired into `make` targets.

### What is already right, and should be reused rather than rebuilt

`orion/field/channel_glossary.py::classify_channel_series()` is the single best
asset here. It is a battle-tested liveness classifier whose verdict vocabulary
directly encodes the incident history above:

- `never_produced` — absent entirely, not even a reconciled default
- `dead` — present but all values subnormal (covers folded-away,
  unproduced-past-reconcile, and subnormal-noise alike)
- `ratchet_suspect` — monotonically non-decreasing with real net climb, and
  ≥ `RATCHET_MIN_SAMPLES` points; catches a `mode=add` channel missing from the
  decay channel sets
- `quiet` — wired and present, low variance this window
- `live` — real variance

The glossary YAML also gets a design decision exactly right, and says so in its
own header: it **deliberately omits a `verdict` field**, because a static
verdict column is precisely what went stale once already. Liveness is computed
live from `substrate_field_state`, not hand-maintained.

That principle — *verdicts are computed, never declared* — is the load-bearing
idea of this whole design. It exists today for 38 of the metric surfaces and
for none of the others.

### The actual gap

Three things are missing, and they are specific:

**1. No shared ID space.** A field channel `cpu_pressure`, an organ dimension
`biometrics/gpu_load#level`, and an inner-state field
`self_state.v1#reasoning_pressure` are the same *kind* of thing — a named
scalar with an upstream and downstream — but nothing can address them
uniformly, so nothing can ask a question that spans them.

**2. No reverse edge, and graphify structurally cannot supply it.** The
knowledge graph carries 81,046 links, including 7,984 `references` and 7,039
`uses`. None of them answer "who reads `cpu_pressure`", because
**metrics are string dict keys, not symbols** — verified: every access is of
the form `vector["cpu_pressure"]` / `.get("cpu_pressure")` / a YAML key. Symbol-level
extraction cannot see them. Consumer lists in `inner_state_registry.py` and
`ORGAN_REGISTRY` are therefore hand-maintained, and go stale silently.

**3. No edit-time surface.** Every gate today is a CI/`make` check that runs
*after* an agent has already written the code. Nothing intercepts the moment an
agent opens a file that reads a metric and tells it what that metric actually is.

### Confirmed staleness in the prior design

`docs/superpowers/specs/2026-07-10-cognition-metric-lineage-registry-design.md`
proposed 8 ideas against this same problem. None were built. It is now partly
stale: it centers on `config/self_state/self_state_policy.v1.yaml` and
`orion/self_state/`, both of which **no longer exist** — removed in the
2026-07-22 SelfStateV1 burn. Its ideas 1, 2, 3, and 4 survive conceptually and
are absorbed below; its `config/self_state`-specific mechanics do not.

## Missing questions

Answered by inspection, not posed to Juniper:

- *Does a metric registry already exist?* Five do, at four different
  granularities. A sixth would be a keyword cathedral. → **join, don't author.**
- *Can graphify supply consumer discovery for free?* No. String-key access is
  invisible to symbol-level extraction. → purpose-built scanner needed, and its
  cost is justified rather than assumed.
- *Is there a proven liveness classifier to reuse?* Yes,
  `classify_channel_series()`, scoped to field channels only. → generalize the
  data source, keep the classifier.
- *Is there a working edit-time hook precedent?* Yes. The graphify PreToolUse
  nudge fires on every Bash/Read call in this repo today. → the mechanism is
  proven; this design reuses its shape.

Genuinely open, and the reason this stops for a decision — see **Recommended
next patch**.

## Proposed schema / API changes

### Metric URN — the ID space

```
metric://<surface>/<producer>/<name>[#<field>]
```

```
metric://field_channel/orion-field-digester/cpu_pressure
metric://inner_state/orion-self-state-runtime/self_state.v1#reasoning_pressure
metric://organ_signal/biometrics/gpu_load#level
metric://bus_channel/orion-substrate-runtime/orion:substrate:brain_frame
```

This is a *projection key*, not a new registry. No URN is hand-authored. Every
URN is derived from an entry that already exists in one of the five registries.
A metric that resolves to no URN is unregistered — and that is itself the
finding.

### `MetricNode` — the resolved join (computed, not stored)

```python
@dataclass(frozen=True)
class MetricNode:
    urn: str
    surface: str                      # field_channel | inner_state | organ_signal | bus_channel
    producer_service: str
    producer_ref: str | None          # file:line of the function that computes it — §0A step 1
    upstream: tuple[str, ...]         # parent URNs — §0A step 2 independence
    downstream: tuple[str, ...]       # file:line consumers, MECHANICALLY discovered
    liveness: str | None              # computed via classify_channel_series, never declared
    liveness_window: str | None       # what window produced that verdict
    sampled_at: datetime | None
    registry_source: str              # which of the 5 registries this came from
    theory_anchor: str | None         # §0A step 3; None is a legitimate, visible answer
```

`liveness` is **never persisted into a config file.** It is computed at
resolve time or read from a dated report artifact. The glossary already
established this rule; this design does not break it.

### No bus/channel/schema contract changes

Nothing here adds a channel, changes a payload, or alters a schema. It is a
read-only projection over existing registries plus a live sample. That keeps
reversibility high (§0A step 6): if the layer is wrong, deleting it removes a
script, a hook, and a doc — nothing is baked into a schema, manifest, or
training default.

## Files likely to touch

**New:**

- `orion/metrics/lineage.py` — the resolver. Reads all five registries, emits
  `MetricNode`s. Zero hand-authored metric content.
- `orion/metrics/consumers.py` — mechanical downstream discovery. AST-based
  scan for string-literal dict-key access (`Subscript` with a `Constant` slice,
  `.get("...")` calls) over `orion/` and `services/`, cross-referenced against
  known metric names. Explicitly **not** a regex sweep (§0A "no regex swamp").
- `orion/metrics/liveness.py` — generalize `classify_channel_series()`'s data
  source beyond `substrate_field_state`. The classifier math is imported, not
  reimplemented.
- `scripts/check_metric_lineage.py` — the CI gate.
- `scripts/hooks/metric_lineage_nudge.py` — the edit-time PreToolUse hook.
- `tests/test_metric_lineage.py`, `tests/test_metric_consumers.py`.

**Modified:**

- `Makefile` — `check-metric-lineage` target, following the 11 existing ones.
- `.claude/settings.json` — register the PreToolUse hook.
- `CLAUDE.md` §0A — point the metric quality gate at the command that now runs it.

**Read-only inputs, unchanged:** all five registries.

## Non-goals

- **Not a sixth registry.** No new hand-authored list of metric names. If it
  can't be derived from an existing registry, it doesn't get a URN.
- **Not a static verdict column.** Liveness is computed. The glossary's header
  already explains why, from a real incident.
- **Not a metric taxonomy or ontology.** No categories, tiers, or semantic
  classes beyond what the five registries already carry.
- **Not a fix for any dead metric it finds.** Detection only. Every fix is a
  separate, individually signed-off patch.
- **Not a replacement for the existing gates.** It joins them; it does not
  subsume `check_bus_channels.py` et al.
- **Not blocking on unknown theory anchors.** `theory_anchor: None` is a
  legitimate, visible state — not a build error.

## Acceptance checks

Each is a runtime fact, not a config fact (§0A "runtime truth beats config truth").

1. `python scripts/check_metric_lineage.py --json` resolves ≥ 38 field
   channels + 13 inner-state signals + 30 organs without a hand-authored URN
   list, and prints the count per registry source.
2. For a known-live metric (`cpu_pressure`) and a known-dead one
   (`node:substrate.route` `prediction_error`, decay-to-zero, 2026-07-26), the
   computed verdicts are `live`/`quiet` and `dead` respectively — verdicts
   recovered from real stored history, not fixtures.
3. Consumer discovery finds `endogenous_curiosity.py` as a consumer of the
   generic node-metric path — the consumer that the `transport_prediction_error`
   retirement missed. This is the regression test for the incident that
   motivated the design.
4. The gate exits non-zero when a test fixture adds a new consumer of a
   metric whose latest computed verdict is `dead` or `never_produced`.
5. The PreToolUse hook, on an `Edit` to a file containing a registered metric
   key, emits that metric's lineage card (producer ref, upstream, downstream
   blast radius, verdict, sample age) into the agent's context — verified by a
   real hook fire with a log line, not by the hook merely being registered.
6. `make check-metric-lineage` is green on `main` at time of merge, with any
   pre-existing dead metrics listed as **reported, not fixed**.

## Recommended next patch

Build in this order. Each phase is independently useful and independently
reversible; phases 2 and 3 are where the user-visible payoff is.

**Phase 1 — resolver + URN join (no gate, no hook).**
`orion/metrics/lineage.py` + `tests/`. Read-only, additive, zero runtime blast
radius. Deliverable: `check_metric_lineage.py --json` dumps the joined graph.
This alone answers "where does this come from and what's upstream" in one place
for the first time.

**Phase 2 — mechanical consumer discovery.**
`orion/metrics/consumers.py`. This is the highest-value single piece: it
produces the **blast radius** — the thing an agent editing a widget currently
has no way to see. It also mechanically re-derives the hand-maintained consumer
lists in `inner_state_registry.py` and `ORGAN_REGISTRY`, and any disagreement
between hand-list and discovered-list is an immediate finding.

**Phase 3 — edit-time PreToolUse hook. SHIPPED 2026-08-19.**
The actual answer to "agents don't trace upstream." Puts the lineage card in
front of the agent *before* the edit lands, using the same mechanism as the
graphify nudge that already works in this repo.

`scripts/hooks/metric_lineage_nudge.py`, registered in `.claude/settings.json`
on the same `Edit|Write|NotebookEdit` matcher as `shared_checkout_edit_guard.py`.
Fails open throughout (matches RTK's and graphify's own "no match" shape):
malformed input, no matched token, or a missing cache all just print nothing.

**The one real design problem, solved by not solving it live.**
`scan_repo()` walks ~3900 files in ~13-14s (measured, both in the original
design doc and again live building this) — far too slow to run synchronously
on every Edit/Write. The hook does not call it. `scripts/refresh_metric_lineage_cache.py`
runs the exact same `build_graph()` + `scan_repo()` calls `check_metric_lineage.py`
already makes and persists them to `.cache/metric_lineage.json` (gitignored,
atomic write via temp-file + `os.replace`) instead of discarding the result
after one CLI invocation. The hook just reads that file — measured at ~65ms
per call including Python startup, not 14s.

Self-healing, not a hard dependency someone has to remember to run: if the
cache is missing entirely, or older than an hour, the hook kicks a refresh
off in the background (`subprocess.Popen(..., start_new_session=True)`, same
"hand the expensive part to a detached process" shape as
`scripts/hooks/stop_worktree_wip_snapshot.py`) and returns immediately either
way — first-ever edit in a fresh worktree shows nothing rather than paying a
14s synchronous cost, and self-heals within the one background refresh's
runtime. A cooldown lock (`.cache/metric_lineage.refresh.lock`, 60s) stops a
burst of edits from spawning a pile of concurrent scans.

Skips FCC subprocess turns (`ORION_FCC_SUBPROCESS`), same convention and same
reasoning as `graphify_hook_guard_gate.sh`: token-budget-constrained, no
code-navigation payoff.

Verified live, not just unit-tested: built a real cache against this repo
(601 nodes, 8,679 hits, 3,877 files, 14.4s) and fed the hook a real edit
touching `field_coherence_warning` — the exact channel this session's R6
investigation spent an hour manually running `check_metric_lineage.py` against
by hand. The hook produced the correct card (URN, meaning, blast radius,
consumer sites) in 65ms, unprompted. That gap — remembering to run the CLI
tool by hand — is precisely what phase 3 exists to close.

**Phase 4 — gate + `make` target.** SHIPPED 2026-08-13.

`make check-metric-lineage-gate`. Three checks, all provable from repo state:

1. **Declared-consumer existence** — a registry may not claim a consumer that
   does not exist, checked at *both* levels: the module file and the callable
   after the colon. Found three on first run: the known
   `orion-spark-introspector`, plus `orion-timeline` and `orion-evidence-index`,
   which appear **only** in `channels.yaml` across four channels with zero
   references anywhere else in the repo. This is the load-bearing check.
2. **Orphan ratchet** — registered metrics that feed nothing (no code consumer
   *and* no surviving declared consumer) may shrink, never grow. A metric that
   names something but feeds nothing is a keyword cathedral (§0A).
3. **Referential integrity** — causal parent organs must exist in
   `ORGAN_REGISTRY`.

**Not wired into CI, despite the name "gate".** No workflow in
`.github/workflows/` runs any `make check-*` target — not this one and not any
of the eleven that predate it. It is a local target run by hand or by an agent
following the contract, which means it can be skipped. Wiring it (and the
others) into a workflow is a real follow-up, not something this patch quietly
implies it did.

Pre-existing debt is carried in `config/metrics/orphan_baseline.json` — recorded
and visible, not silently waived. Fixing it means editing `consumer_services`
in `channels.yaml`, a contract change (§6) belonging in its own patch.

**What phase 4 deliberately does NOT catch:** newly introduced but
*unregistered* metrics. `arena_degeneracy` (PR #1604) shipped the same day and
is invisible to this layer. Catching that statically would mean guessing which
new string dict keys are "metric-shaped", and the only available signal is a
suffix list (`*_pressure`, `*_load`, `*_error`, …) — precisely the keyword
cathedral §0A bans, false-positive noisy, and trivially evaded. Registration is
enforced by review and by the phase 3 card, not by a heuristic pretending to be
a gate. Stated so the absence reads as a decision, not an oversight.

**Phase 5 — liveness generalization beyond field channels. PARTIALLY SHIPPED
2026-08-19** (two candidates only; the general version is still deferred —
see below).

The general question — "for any URN on any surface, where does the live
sample come from" — is still bigger than the first four phases combined, and
is not solved here.

**Correction 2026-08-20** (code review): this section previously credited
`docs/superpowers/specs/2026-08-13-phase5-liveness-scope.md`'s "R6 section"
with a systematic 48-URN walk finding "25 of 48 have retired producers".
Neither claim survived re-verification. That doc's R6 is an unrelated,
still-open question ("can a metric express rest") and contains no such walk.
Re-running `check_metric_lineage.py --json` live against current `main`
(2026-08-20) confirms 48 `inner_state` nodes (15 signals + 33 scalar fields)
is the real total, but only 3 signals (8 nodes) have a retired producer —
`self_state.v1`, `drive_state.v1`, `autonomy_state_v2` — not 25. Of the
remaining 12 signals: 4 were ruled out with a specific documented reason
(`mood_arc_corpus.v1`/`field_channel_corpus.v1` config-gated,
`chat_stance_disposition` categorical, `biometrics_cluster.v1` a registry-
flagged duplicate of `field_state.v1`), 2 were built (below), and 6 —
`field_state.v1`, `field_attention_frame.v1`,
`attention_broadcast_projection.v1`, `mood_arc_encoder.v1`,
`phi_heuristic.valence`, `phi_intrinsic_reward.v1` — were never actually
investigated in this pass, despite an earlier draft of this doc implying
full coverage. Named explicitly here rather than left to look covered.

Two had real backing data and a concrete case for building liveness now:

- **`attention_self_model.v1`** — a real live consumer
  (`orion-equilibrium-service`'s metacog gates), 19,426 rows, ~30s cadence.
- **`l7_l11_ladder`** — initially dismissed for having "no cognition
  consumer," which was a category error caught by a direct challenge ("why
  you sweeping l7 l11 ladder under rug"): it carries a live *mutating* route
  (`skills.runtime.builder_prune.v1`) that deletes host data, independent of
  whether anything downstream reads it cognitively. Its own registry notes
  already flag the `REHEARSAL` classification as stale given that route.

**What got built:** `orion/metrics/liveness.py`. The classifier math is
never reimplemented — every verdict still comes from the existing
`channel_glossary.classify_channel_series()`, exactly as designed. Only the
data source is new, and it's narrow by construction (two hardcoded cases, not
a generic per-surface resolver):

- `attention_self_model.v1`'s five scalar fields (`confidence`,
  `prediction_error_confidence`, `field_overall_salience`,
  `broadcast_lane_age_sec`, `heartbeat_mean_ratio`) are read straight off
  `substrate_attention_self_model.self_model_json ->> '<field>'`, ordered by
  `generated_at`, over a 1h window (~120 samples when healthy).
- `l7_l11_ladder` has no shared scalar — it's a pipeline
  (`ProposalFrameV1 -> ... -> ConsolidationV1`), not a signal — so it's
  reframed as **throughput** liveness: rows-per-bucket across each of its 5
  backing tables, fed into the same classifier. Bucket size is tuned per
  table's real cadence (measured live 2026-08-19, not assumed): the four
  ~2.1s-cadence stages use 1-hour windows / 1-minute buckets;
  `substrate_consolidation_frames` (~96min cadence, confirmed live) uses a
  48h window / 3h buckets — a 1h bucket would read ~0.6 rows/bucket on a
  *healthy* table, false "dead" territory. The 5 per-stage verdicts roll up
  to one via `_worst_of()`, which reuses `channel_glossary.CLEAN_VERDICTS`
  rather than a hand-invented severity order — **a real bug this caught**:
  an earlier draft ranked `quiet` above `live`, so 4 live stages + 1
  expectedly-quiet slow stage rolled up to an overall `QUIET`, which reads as
  a regression on every healthy tick. Fixed before merge; regression test
  added (`test_worst_of_live_and_quiet_mix_is_live_not_quiet`).

**Two review rounds found real issues, all fixed before merge** (full
findings/fixes are in the PR description, not duplicated here). Round 1:
`open_readonly_connection` had been reimplemented from scratch instead of
importing the repo's existing `scripts/analysis/_pg_readonly.py` helper —
moved to `orion/db_readonly.py` so both layers can share it without
inverting `orion/` -> `scripts/` dependency direction; a query-time DB
failure (not just a connect-time one) could crash the whole `--metric` CLI
instead of degrading to UNKNOWN; `LIVE_VARIANCE_THRESHOLD` reused unmodified
against row-count data trips `ratchet_suspect` on any ordinary busy burst,
and — caught by a second round of live testing, not assumed fixed —
normalizing the series to unit scale does NOT actually prevent that (a
monotonic climb clears the threshold at any scale), so `ratchet_suspect` is
now explicitly downgraded to `live` for throughput data, since that verdict's
real meaning has no equivalent for pipeline throughput; and the ladder's
`sample_count` was counting non-empty buckets (~1 per bucket) instead of real
rows, understating true volume by ~28x for the fast stages.

Round 2 (against the round-1 fixes) found the single most important issue in
the whole patch, and it was in this design doc, not the code: a claim that a
"48-URN systematic walk" was recorded in `2026-08-13-phase5-liveness-scope
.md`'s R6 section did not check out — that section is about a different,
unrelated question, and contains no such walk. Re-verified live 2026-08-20;
see the corrected accounting a few paragraphs up. Also found: the JSON
`--metric` output couldn't distinguish "DB unreachable" from "no source
registered" without parsing free text (fixed with a `liveness_status`
field); `connect_timeout` only bounded the connect phase, not a query
hanging after connect (fixed with a session-level `statement_timeout`);
`broadcast_lane_age_sec` — unlike its four `[0,1]`-bounded sibling scalar
fields — shares the ladder's exact borrowed-threshold exposure and wasn't
routed through the domain-safe classifier; `has_registered_source()` and
`liveness_for_node()` independently duplicated the same routing
conditionals, a silent-drift risk for a future third candidate (fixed with
one shared `_resolve_source_kind()`); and `ScalarFieldSource`'s row cap
could silently truncate a window with no visibility into it, unlike the
existing sibling reader this pattern is modeled on (fixed with a
`truncated` flag). One finding — `resolve_dsn()`'s env-var priority
disagreeing with a cited-as-matching precedent — was fixed by correcting the
docstring's claim rather than the code; the priority order itself
(`POSTGRES_URI` first) was the right call, matching `.env_example`'s
canonical key.

Wired into `check_metric_lineage.py --metric <name>` (and `--json`): a node
with a registered source now prints a real verdict with its sample count and
provenance string instead of the old blanket "NOT COMPUTED (phase 5)". A
node without one still prints an honest "NOT COMPUTED — no live data source
registered", never silently blank. A live-but-unreachable Postgres reports
`UNKNOWN`, distinct from both — found while testing the failure path itself
that `psycopg2.connect()` has no default timeout, so a genuinely
unreachable (not merely refused) host would otherwise hang the whole CLI
call; fixed with an explicit 5s `connect_timeout`.

Connection contract mirrors this repo's existing precedent
(`orion/substrate/felt_state_reader.py`,
`scripts/analysis/measure_attention_self_model_confidence_baseline.py`):
refuses a session that doesn't confirm `default_transaction_read_only = on`.
Host default is `postgresql://postgres:postgres@localhost:55432/conjourney`
(verified live 2026-08-19 — the docker-network hostnames those in-container
readers use, `orion-athena-sql-db`/`orion-sql-db`, do not resolve from the
host); `POSTGRES_URI`/`DATABASE_URL`/`ORION_SQL_URL` override, matching
`scripts/print_recent_turn_effects.py`'s existing fallback chain.

Verified live against real data, not just mocked in tests: `--metric
confidence` returns `QUIET` for `attention_self_model.v1#confidence` (matches
the ~0.04%-of-ticks branch-starvation finding already on record for that
field) and `LIVE` for `broadcast_lane_age_sec`/`heartbeat_mean_ratio`;
`--metric l7_l11_ladder` returns `LIVE` (4 stages live, consolidation quiet,
correctly rolled up after the severity-order fix above).

**Still deferred, and not attempted here:** every other surface (field
channels already have their own live classifier; `organ_signal` and
`bus_channel` have no liveness source registered at all), the 3 retired-
producer signals and 4 ruled-out-with-reason signals named above, and the
**6 signals never investigated in this pass** — `field_state.v1`,
`field_attention_frame.v1`, `attention_broadcast_projection.v1`,
`mood_arc_encoder.v1`, `phi_heuristic.valence`, `phi_intrinsic_reward.v1`.
That last group is a real gap, not a decision — see the 2026-08-20
correction above.

### Decisions taken (2026-08-12)

- **Scope of the first pass: phases 1+2.** Built and merged as one PR.
- **Phase 3 hook behavior: always show the full lineage card**, on every edit
  touching a registered metric — not only on dead/suspect verdicts. Juniper's
  call. The nudge-blindness risk (the agent-board Stop banner a live session
  ignored across ~20 turns) is accepted deliberately: a card that only appears
  for already-known-bad metrics cannot prevent the failure where nobody knew
  the metric was bad yet.

## Build findings (phases 1+2, 2026-08-12)

Two things the build changed about the design, both found by running the
scanner against the real repo rather than reasoning about it:

**1. `collection_member` is the access kind that matters most, and it was
missing from the original design.** The first scanner draft classified only
subscript / `.get` / dict-key / attribute access as high confidence. Under
that model `cpu_pressure` had **81 raw hits and a blast radius of zero** — no
explicit key read anywhere outside tests. Its real production consumers all
register it into a *collection that some loop iterates*:

```
orion/field/pressure.py:35                              (channel tuple, reducer-iterated)
orion/field_coherence.py:9,13                           (coherence rule pairs)
services/orion-field-digester/app/digestion/decay.py:31 (NODE_DECAY_CHANNELS)
services/orion-field-digester/app/tensor/channels.py:4  (channel definition set)
```

That `decay.py:31` line is the exact mechanism behind the 2026-07-26
decayed-to-zero incident in CLAUDE.md §0A — a generic staleness loop
multiplying a channel by 0.92 per tick. Membership in that list is now
first-class lineage rather than something to rediscover by hand.

This is also the generalized form of the `transport_prediction_error`
retirement miss: a metric "retired" from one *named* consumer keeps feeding
every *generic* one. A blast radius that ignores collection membership reports
empty for exactly the metrics most likely to be silently alive.

**2. Attribute access was systematically undercounting inner-state metrics.**
Field-channel and organ metrics are string dict keys; inner-state pydantic
scalars are attributes (`frame.recon_error`). The first run reported 9
inner-state metrics with zero consumers; adding `ast.Attribute` recovered 4 of
them (+153 hits). The remaining 5 (`delta_phi`, `delta_recon_error`,
`overall_confidence`, `recon_error`, `shuffle_baseline_loss`) appear to be
genuinely unread and are reported as findings, not fixed here.

### Measured, not asserted

```
bus_channel   261    scan tokens     386
field_channel  38    files scanned  3714
inner_state    36    consumer hits  4713
organ_signal  252    runtime         ~13s
TOTAL         587 URNs
```

### Known limits, stated

- Dynamic access (`vector[name_from_config]`) is invisible to any static
  scanner. Such call sites are **undercounted, never overcounted**.
- Python matching is exact-equality on the string constant, so prose that
  merely contains a metric name produces no hit at all. The YAML/JSON scan is
  substring-based and correspondingly noisier — it is tagged `config` and
  excluded from high-confidence blast radius.
- Liveness is still **not computed** (phase 5). Every lineage card says so
  explicitly rather than leaving a blank that reads as "fine".
