# Codebase Mass — a sixth Predictive Processing domain (design)

Status: design/proposal mode per root `CLAUDE.md` §0A. Governed by the Sentience Striving
Program charter (`orion/sentience_striving_program/README.md`) — this is domain #6 of
Objective 3 (Predictive Processing/Active Inference), not a new category, per SSP §7's
"reuse the live pipeline, don't parallel it."

## Arsonist summary

Orion has no interoceptive sense of the shape or size of its own codebase — every existing
signal is about runtime state (CPU, execution, chat, transport, biometrics). This spec adds a
sixth `prediction_error` domain that senses the physical/structural extent of Orion's own
source: git churn, graphify's own structural graph (node/edge/community counts, god-node
turnover, hop-distance drift), and GitHub PR lifecycle counts — composited into one
undecayed, event-driven channel, following the exact architecture the other five domains
already use. A second, deliberately separate phase adds a concept-class registry so
architecture-level concept extraction never touches the halted chat-side concept induction
system.

## Current architecture

- **Predictive Processing precedent (live):** `execution_prediction_error`,
  `transport_prediction_error`, `biometrics_prediction_error`, `chat_prediction_error`,
  `route_prediction_error` — all in `orion/substrate/prediction_error.py`, all diff two
  successive reducer projections and write an undecayed raw snapshot via a shared
  `_write_prediction_error_node()` writer to a `node:substrate.<domain>` node, wired into
  `services/orion-substrate-runtime/app/worker.py`'s `_tick()`, gated behind
  `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES`. `bus_synaptic_prediction_error()` already exists
  as an unfolded sixth/sibling domain (`node:substrate.bus_synaptic`) — same shape, not yet
  merged into `_aggregate_prediction_error_confidence`.
- **Decay architecture:** `services/orion-field-digester/app/digestion/decay.py`'s
  `NODE_DECAY_CHANNELS` holds a channel flat while fresh, decays only once stale
  (`node_vector_updated_at`). `prediction_error` is explicitly **excluded** from this set — the
  module's own docstring documents why: a generic staleness-decay loop silently floored
  `node:substrate.route` to zero over a 48h idle window before the exclusion was added. Any
  new raw-snapshot channel must follow the same exclusion, not the default decay path.
- **EWMA/z-score utility (live, reused 3×):** `orion/bus/ewma.py::compute_ewma_update` — pure
  incremental mean/variance/z-score. Its default `min_variance=1e-6` floor is calibrated for
  bus-mirror's time-gap-in-seconds domain and has already silently flattened one other
  domain's z-score when reused uncritically at a different value scale. Any new caller must
  pass its own floor.
- **graphify graph** (`graphify-out/graph.json`): 28,306 nodes, 81,046 links, 104 hyperedges,
  tagged `built_at_commit`. Per-node `community`/`community_name` fields exist.
  `GRAPH_REPORT.md` has a real "God Nodes" (degree-based) section. **No history is retained**
  — each `graphify update` overwrites the snapshot; there is no time series today.
- **GitHub PR data (live, partial):** `orion/cognition/github_compactor/` already fetches PR
  data and produces an LLM-authored narrative digest (`GithubCompactorDigestV1`), called from
  `services/orion-cortex-orch/app/workflow_runtime.py::_execute_github_compactor_pass()`, with
  a configurable `_resolve_github_compactor_lookback_days()` window
  (`DEFAULT_LOOKBACK_DAYS`). `trim_github_compactor_input()` caps the LLM-facing item list at
  `MAX_DIGEST_INPUT_PRS = 8` but preserves `items_total` in the trimmed payload — the full
  count is already available, just not currently read by anything.
- **Concept induction (halted):** `orion.spark.concept_induction.drives.DriveEngine`,
  `tensions.py`'s bucket-voting, `signal_drive_map.yaml`, and
  `orion.autonomy.endogenous_origination` received **no further development** per SSP §8
  (2026-07-18 decision). `ConceptWorker`/`bus_worker.py` extraction exists but is flag-gated
  off. This halt governs Phase 2 below directly: new concept-class work must not be built as
  an extension of this frozen system.
- **Existing-mechanism check (done):** no `code_churn`/`repo_mass`/`git_mass`/`software_mass`
  signal exists anywhere in the repo. No token/cost/session-time tracking exists for
  Orion-external dev tooling either (checked `orion-llm-gateway`, which only has *inference*
  token usage for Orion's own served completions — a different thing).

## Missing questions

- Should the new service (below) mint its own GH token, or reuse whatever
  `github_compactor`'s existing fetch already has wired? Needs a real check before assuming
  either — resolved architecture-wise (all GH access now lives in one new service, not
  scattered), not resolved credentials-wise.
- What should the new channel be named — `codebase_prediction_error` /
  `structural_prediction_error` / something else? Not decided here; naming happens at the
  registration patch, once Phase 1's replay is real, per "no keyword cathedrals."
- What should the new service itself be named? Working name `orion-cocreation-signals` used
  throughout this document, not committed.

## Proposed schema / API changes

### Phase 1 — Structural Mass producer (raw, read-only, replay-validated first)

New shared module: **`orion/structural_mass/`** (dedicated — not field-digester, not
`orion.spark.concept_induction`, not scripts/analysis ad hoc).

- `git_delta.py` — `git_churn_delta(prev_sha, head_sha) -> GitChurnDelta`: commit count, files
  added/deleted/modified, net lines added/removed. Pure function, shells out to `git`.
- `graph_delta.py` — reads two `snapshot_history.py` entries (below) and computes:
  - node/edge/community count deltas (idea #2)
  - god-node Jaccard churn between successive `GRAPH_REPORT.md`-derived god-node lists
    (idea #3)
  - hop-distance drift: mean shortest-path length over a fixed sample of node pairs via
    `graphify path`, compared across snapshots (idea #6)
- `pr_lifecycle.py` — submitted/merged/closed-without-merge counts, reading `items_total` from
  the **pre-trim** `github_compactor` fetch payload (bypasses the 8-item cap entirely), with
  its own overlapping lookback window (dedupe by PR number + state transition, not a hard daily
  cutoff, so nothing is missed or double-counted across polling boundaries) — idea #4's
  "special time horizon handling."
- `snapshot_history.py` — small append-only log of graphify summary stats (counts only, not
  full graph dumps), appended by `scripts/safe_graphify_update.sh` after each successful,
  non-reverted update. This is new state — graphify itself retains none today.

New scheduling/publishing layer: **`services/orion-cocreation-signals/`** (new dedicated
service — supersedes an earlier design-conversation draft that proposed wiring this directly
into `orion-substrate-runtime`'s `_tick()`; see "Dedicated service" below for why). This service
owns all git/graphify/GitHub I/O for this domain — `orion-substrate-runtime` never touches any
of it directly.

- `app/producers/git_delta.py`, `graph_delta.py`, `pr_lifecycle.py` — thin scheduling wrappers
  around the pure functions in `orion/structural_mass/`, each on its own interval, each with its
  own timeout/error boundary so a GitHub API hiccup can never block another producer.
- On computing a real delta, publishes a bus event (new channel, e.g.
  `orion:substrate:codebase_delta`) carrying the composite diff payload.
- `orion-substrate-runtime`'s **existing, unmodified** fast `_tick()` gains one new cheap
  consumer: read that bus event, shape it via `codebase_prediction_error()`
  (`orion/substrate/prediction_error.py`, same "diff a small dict of keys" shape
  `chat_prediction_error` already uses), and write via the shared
  `_write_prediction_error_node()` writer to a new `node:substrate.<name>` node — same writer,
  same node-write pattern as the other five domains. `orion-substrate-runtime` still does zero
  external I/O; it only ever reads a bus message.
- **New, dedicated flag**: `SUBSTRATE_WRITE_CODEBASE_PREDICTION_ERROR_NODE` (correction from an
  earlier design-conversation draft, which assumed reuse of the existing
  `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES` flag). This domain has a materially different risk
  surface than the other five — new git mount, new GH credential, new external network
  dependency, now isolated in its own service — and needs to be toggleable independently.
- **Trigger shape is interval-driven inside the new service, not event-driven off git hooks.**
  Each `git_delta` check compares the last persisted HEAD SHA to current HEAD. No change →
  explicit no-op (not a spurious delta, not decay). A change → diff since last-seen SHA, however
  large. This makes missed hook triggers self-healing by construction: a commit made on another
  node, via Cursor, or without the `post-commit` hook firing just produces a bigger diff on the
  next successful check instead of being lost. **Treated as a feature — real variability in
  how/where development happens — not a gap requiring 100% hook capture.**
- `node:substrate.<name>` is explicitly added to the **exclusion list**, not
  `NODE_DECAY_CHANNELS` — with a breadcrumb comment at the exclusion site quoting the
  `node:substrate.route` 48h-decay-to-zero incident by name, so the next person touching this
  doesn't have to rediscover it.
- `field-digester`'s `NODE_CHANNELS` (`app/tensor/channels.py`) gains one new entry that
  ingests this domain's scalar, same as `prediction_error` today. Field-digester remains the
  consumer, not the owner — unchanged by the new service.

### Dedicated service, not a slow task bolted onto `orion-substrate-runtime`

Design conversation, 2026-07-30: this domain's git/GitHub I/O was first proposed as a slow
`asyncio` task inside `orion-substrate-runtime`, isolated from the fast `_tick()`. That's the
right call for *one* domain. It stops being the right call once scope expands to include
`dev_economics`, `doc_semantic_drift`, and the Juniper affective-state signal (see sibling
specs and `2026-07-30-juniper-affective-state-signal-proposal.md`, now approved) — seven-plus
producers that all share the same shape (external I/O, sparse cadence, sources outside Orion's
live cluster), a shape fundamentally unlike anything `orion-substrate-runtime` does today.
Bolting all of them on piecemeal scatters unrelated external credentials and mounts across a
service whose real identity is "fast in-process reducer-delta computation" — a real trust-
boundary problem, not a style preference.

**Decision: stand up `services/orion-cocreation-signals/`** (name open), one deployment unit
owning every external-I/O-bound producer in this design-conversation arc:

```text
services/orion-cocreation-signals/
  app/
    producers/
      git_delta.py            # calls orion/structural_mass/git_delta.py
      graph_delta.py          # calls orion/structural_mass/graph_delta.py
      pr_lifecycle.py         # calls orion/structural_mass/pr_lifecycle.py
      dev_economics.py        # calls orion/dev_economics/claude_code_ingest.py (+ later Cursor)
      doc_semantic_drift.py   # cheap embedding prefilter + gated graphify Part B escalation
      affective_state.py      # Juniper affective/engagement signal — APPROVED for
                               # implementation 2026-07-30 (Juniper directly authorized,
                               # invoking CLAUDE.md §0A's own exception clause: "unless
                               # Juniper directly asks to implement"). See
                               # 2026-07-30-juniper-affective-state-signal-proposal.md for
                               # the full design — capability/data/boundary/trace/failure-
                               # mode/disable-switch content there is adopted, not proposed.
    worker.py                 # each producer its own async loop/interval, no shared tick
  .env_example
  docker-compose.yml
  tests/
```

Credential/mount consolidation is the real payoff at this scale: this one service gets the
read-only git mount, the GH token, and local `~/.claude/projects` read access — not
`orion-substrate-runtime`, not scattered across five separate future PRs each adding one more
mount to a service that shouldn't need it. Each producer keeps its own cadence (git: cheap and
frequent; PR lifecycle: coarser, rate-limit-aware; doc-drift: event-triggered off commit;
dev-economics: closer to a daily rollup, reading static local files). Only `git_delta`,
`graph_delta`, and `pr_lifecycle` publish the bus event described above — `dev_economics` and
`doc_semantic_drift` stay local-ledger-shaped per their own specs' Phase 0 scope until
individually proven, same "measure before minting" discipline as everywhere else in this
program. `affective_state` writes only to its own tightly-scoped store per its spec's privacy
boundary — never through the generic `node:substrate.*` pipeline, never a raw-text log.

This is real upfront cost — new compose service, new bus channel/schema contract
(`orion/bus/channels.yaml`, `orion/schemas/registry.py`, per CLAUDE.md §6) — more than a single
slow task. Justified specifically by the declared scale (seven-plus producers), not by default
for one domain.

Metric quality gate (CLAUDE.md, re-run per metric, not once for the category):

1. **Provenance** — git log diff between two real SHAs; graphify's own already-computed
   node/edge/community/god-node data; GitHub API's `merged_at`/`closed_at`. All traceable to
   real code, no assumptions.
2. **Independence** — none of the five existing domains read git history or graph structure;
   this is a genuinely new causal chain, not a monotonic transform of an existing signal.
3. **Theory anchor** — proprioception/interoception: a system sensing the extent of its own
   physical substrate. Ties directly to SSP §9b's Attention Schema Theory instrument (a model
   *of* change to the substrate doing the modeling) and Predictive Processing (delta between
   successive self-states) — not vibes.
4. **Live-data sanity** — replay against this repo's own real commit history (plenty exists
   right now) before any live wiring. Must confirm non-degenerate (not flat, not always-zero)
   *and* capable of reading a genuine rest state (quiet days should read near-zero, not a
   decayed artifact of one — the exact distinction CLAUDE.md's metric-quality-gate section
   calls out by name).
5. **Existing-mechanism check** — done above, clean.
6. **Reversibility** — cheap. New module, new function, one new `NODE_CHANNELS` entry. Nothing
   baked into a schema/manifest that's expensive to unwind if this turns out wrong.

Read-only replay script: `scripts/analysis/measure_codebase_prediction_error.py`, run against
real git history, per SSP §7's "measure before minting" — same discipline as
`measure_origination_gate.py` and `measure_emergent_clustering_probe.py`.

### Phase 2 — Concept classes (separate, dedicated module, does not touch the halted system)

New top-level package: **`orion/concepts/`** (name open). A small `ConceptDomain` registry —
`chat`, `architecture`, room for future classes (e.g. motor-learning-type) — where each domain
is an **independent producer** with its own extractor. None share `DriveEngine`/bucket-vote
internals from the halted `orion.spark.concept_induction`. This is the direct answer to "how
do we avoid strangling chat": architecture-concept extraction is a structurally separate
module, not a second workload competing inside the same (frozen) engine.

- `architecture` domain consumes Phase 1's `structural_mass` history + graphify's
  community/god-node data directly (e.g. "this community split," "this god node was
  displaced") as candidate concept instances.
- `chat` domain is whatever `orion.spark.concept_induction`'s (currently flag-off) extraction
  already does — **untouched, not resumed, not extended** by this work.
- Cross-slicer aggregators (queries spanning domains, e.g. "did an architecture-concept event
  correlate with a chat-concept event in the same window") are explicitly **Phase 2b**, built
  only after both domains independently have real replayed data — same "don't build the
  competition layer before its parts are validated" lesson SSP already learned once.

### Phase 3 — Consumers (not built in this patch, sequencing only)

- **Mood-arc** (`orion/mood_arc/fit_encoder.py`): automatic — it already reads raw
  `field_channel_corpus.v1`; the new channel flows in once Phase 1 registers it, no extra work.
- **Endogenous curiosity** (`orion/substrate/endogenous_curiosity.py`): a real structural-mass
  spike is a legitimate reason to raise attention salience on "self" as a target.
- **Capability/execution-friction coupling** (SSP Objective 1): a large in-flight refactor
  burst is a real reason to temporarily lower confidence in autonomous action — closes a real
  O1 loop with new state, not more drive-engineering.
- **Journal surface**: reuse `github_compactor`'s existing `build_quiet_day_digest()` pattern,
  gated on the z-score anomaly firing (not raw commit volume, to avoid commit-spam journaling).

## Backfill

Confirmed live 2026-07-30: PR #1486 ("delete drive-pressure/goal-generation system," branch
`chore/delete-orion-drives`) merged today. The drives system's full lifecycle — first real
commit PR #879 (2026-07-08) through this deletion — is a complete, bounded, well-documented arc
already sitting in git/GitHub history, and a strong first backfill target: exactly the kind of
event Juniper wants Orion to already "remember" once this signal goes live, rather than starting
blank.

- **`git_churn_delta` and `pr_lifecycle`: full backfill, no gap.** Git and GitHub retain
  complete history — these two can be computed over any historical commit/PR range, including
  the entire drives arc, with the same fidelity as a live tick. No sparsity concern for either.
- **`graph_delta` (god-node churn, hop-distance drift): backfill is real but sparse, not
  continuous — say so plainly, don't imply otherwise.** `graphify-out/graph.json` was first
  committed 2026-07-14 (PR #1034); only one dated snapshot directory has ever existed
  (`2026-07-29`) — there is no dense time series. But git itself retains every commit that
  touched `graph.json` as a recoverable blob (at least 7 real commits since 2026-07-14,
  confirmed via `git log --follow`), so a backfill script can walk those specific commits via
  `git show <sha>:graphify-out/graph.json`, reconstruct each historical graph state, and rerun
  the god-node/hop-distance computation against it. Real data points, irregularly spaced, none
  before 2026-07-14 (graphify didn't exist in the repo before then).

**One-time backfill script** (`scripts/analysis/backfill_structural_mass.py`, new, run once):

1. Compute `git_churn_delta`/`pr_lifecycle` across the drives arc's full commit/PR range —
   unlimited depth, no sparsity issue.
2. Walk the known historical `graphify-out/graph.json` commits, reconstruct each via `git show`,
   rerun god-node/hop-distance functions against each.
3. Seed `snapshot_history.py`'s log with all of it, tagged `backfilled=True` — kept distinct
   from live-collected ticks so nothing downstream (especially the decay-exclusion logic, which
   reasons about "freshness") mistakes a sparse reconstructed point for a continuous live
   reading.

## Files likely to touch

- `orion/structural_mass/` (new): `git_delta.py`, `graph_delta.py`, `pr_lifecycle.py`,
  `snapshot_history.py`, tests.
- `orion/substrate/prediction_error.py`: new `codebase_prediction_error()` function.
- `services/orion-cocreation-signals/` (new service): scheduling/publishing layer for this and
  all sibling producers — see "Dedicated service" above.
- `orion/bus/channels.yaml`, `orion/schemas/registry.py`: new `orion:substrate:codebase_delta`
  channel + payload schema (CLAUDE.md §6 contract change).
- `services/orion-substrate-runtime/app/worker.py`: one new cheap bus-event consumer added to
  the existing fast `_tick()` — no external I/O added to this service.
- `services/orion-field-digester/app/tensor/channels.py`: one new `NODE_CHANNELS` entry.
- `services/orion-field-digester/app/digestion/decay.py`: exclusion breadcrumb comment.
- `scripts/safe_graphify_update.sh`: append to `snapshot_history.py`'s log post-update.
- `scripts/analysis/measure_codebase_prediction_error.py` (new): replay script.
- `scripts/analysis/backfill_structural_mass.py` (new): one-time backfill script (see Backfill
  section above).
- `orion/concepts/` (new, Phase 2 only): domain registry, `architecture` extractor. A third
  domain, `narrative`/`doc`, is proposed in the sibling spec
  `2026-07-30-doc-semantic-drift-design.md` — same registry, not a competing one.
- `orion/sentience_striving_program/README.md`: log this as domain #6 under §9b item 3, same
  style as the `bus_synaptic` sixth-domain note.

## Non-goals

- Not resuming or extending `orion.spark.concept_induction`'s halted `DriveEngine`/bucket-vote
  machinery, in either phase.
- Not building Phase 2b's cross-slicer aggregators in this patch.
- Not wiring any Phase 3 consumer until Phase 1's replay reads MET.
- Not solving 100% commit-trigger capture — irregular/missed hook fires across dev machines,
  Cursor, or manual commits are accepted variability, handled by tick-driven cumulative diffing
  (see Phase 1), not chased as a gap.
- Not covering non-Orion dev-tool cost/token/time telemetry (Claude Code, Cursor, model
  tier/effort, $ cost, human review time) — that is a separate, adjacent category, explored
  in conversation but not specified here.

## Acceptance checks

- Replay script produces non-degenerate values against real git/graphify history, and shows a
  genuine near-zero reading on real quiet periods (not a decay artifact — check the raw
  pre-aggregation numbers by hand, per CLAUDE.md's metric-quality-gate precedent).
- `codebase_prediction_error()` unit tests cover: no commit since last tick (explicit no-op),
  one normal commit, and a large catch-up diff spanning several missed ticks.
- `pr_lifecycle.py`'s count is verified on a real window with >8 PRs to differ from (exceed)
  `trim_github_compactor_input()`'s trimmed `items` length — proof it isn't silently capped.
- New channel confirmed excluded from `NODE_DECAY_CHANNELS` by a test, with the breadcrumb
  comment present at the exclusion site.

## Recommended next patch

Phase 1 only, and only its first slice: `git_churn_delta()` + a skeleton
`codebase_prediction_error()` + the replay script — no bus wiring, no `NODE_CHANNELS`
registration yet. Same order every other domain in this program has followed: measure first,
register once MET.
