# Codebase Mass — a sixth Predictive Processing domain (design)

Status: design/proposal mode per root `CLAUDE.md` §0A. Governed by the Sentience Striving
Program charter (`orion/sentience_striving_program/README.md`) — this is domain #6 (documented
as the seventh domain node in the SSP README's own running count, since `bus_synaptic` already
claimed "sixth" — naming here is about Objective 3's domain family, not a literal ordinal) of
Objective 3 (Predictive Processing/Active Inference), not a new category, per SSP §7's
"reuse the live pipeline, don't parallel it."

## Status, 2026-07-30 (updated same day as the original spec)

Phase 1's three producers and the composite scoring/bus contract are **shipped, tested,
replay-verified against real data — not yet wired into a live tick.** Four PRs:

- `orion/structural_mass/git_delta.py` (PR #1496).
- `orion/structural_mass/pr_lifecycle.py` (PR #1500).
- `orion/structural_mass/graph_delta.py` + `snapshot_history.py` (PR #1502).
- `orion:substrate:codebase_delta` bus channel/schema (`orion/schemas/codebase_delta.py`) +
  composite `codebase_prediction_error()` in `orion/substrate/prediction_error.py` (PR #1515).

Real deviations from this spec's original assumptions, found during implementation (not
guessed — confirmed against real code/data each time):

- **`pr_lifecycle.py` does not reuse `github_compactor`'s fetch, contrary to this spec's
  original "Missing questions" framing.** That fetch
  (`services/orion-cortex-exec/app/verb_adapters.py::GithubRecentPullRequestsVerb`) turns out to
  hardcode `per_page=20` (no further pagination) and unconditionally drop every PR with no
  `merged_at` — it can never report "submitted" or "closed-without-merge" counts, only merged
  ones. `pr_lifecycle.py` fetches independently via the `gh` CLI instead (`gh pr list --search
  "sort:updated-desc"` — plain `gh pr list` sorts newest-created-first only, which would miss an
  old PR merged recently; verified live).
- **`graph_delta.py` implements ideas #2 (count deltas) and #3 (god-node Jaccard churn) only.**
  Idea #6 (hop-distance drift via `graphify path`) is deliberately deferred — it needs live graph
  traversal calls, a materially different cost class from the other two (pure functions over
  already-computed snapshot summaries), and isn't validated the way the shipped two now are. Real
  follow-up, not dropped.
- **The composite `codebase_prediction_error()` normalizes each domain to its own EWMA z-score
  independently, then averages the z-scores** — not the single "diff a small dict of keys" shape
  this spec originally proposed borrowing from `chat_prediction_error`. That shape assumes one
  set of raw values on one common scale; git churn (thousands of lines), PR counts (single
  digits), and graph node/edge deltas (hundreds to hundreds of thousands) are not on a common
  scale, so combining raw magnitudes first would reintroduce an arbitrary cross-scale weighting.
  Normalize-then-average avoids it, mirroring `bus_synaptic_prediction_error`'s own "aggregate
  many already-normalized z-scores" shape.
- **Real live-data validation exceeded this spec's own ask.** The `graph_delta` replay
  independently reproduced this repo's actual 2026-07-14 destructive-graph-update incident
  (`GRAPH_REPORT.md`'s "God Nodes" parser also got a real bug found+fixed by code review: it
  could silently misreport a genuine parse failure as "nothing changed" — see PR #1502). The
  composite scoring replay caught its own real calibration bug (a borrowed variance floor that
  would have flattened the PR domain's real signal) before merge, not after — see PR #1515.

**What's still not live:** no `orion-cocreation-signals` service exists, `orion-substrate-runtime`
has no consumer for the new channel, `node:substrate.codebase` is not registered in
`NODE_CHANNELS`, and it does not appear in the Attention Organ tab
(`services/orion-hub/scripts/attention_organ_routes.py`) or any other UI — deliberately, since a
node with no live producer would be an empty/dead panel row (confirmed via investigation
2026-07-30: Causal Geometry, the other Hub tab considered, turned out to be a structurally
unrelated signal family — causal-graph-edge structure, not domain-level `prediction_error`
scalars — not a fit for this domain at all). See "Producer + consumer patch design" below for
the concrete next-patch elaboration.

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

- ~~Should the new service mint its own GH token, or reuse `github_compactor`'s?~~ **Resolved
  by implementation, 2026-07-30:** `pr_lifecycle.py` doesn't reuse `github_compactor`'s fetch at
  all (see "Status" above) — it shells out to the local `gh` CLI, same as this repo's own
  dev/CI environments already use for PR management. The credentials question for the *service*
  (below) is still open: does `orion-cocreation-signals` get its own `GITHUB_TOKEN`/mount, or
  invoke `gh` the same way the replay script does (relying on an already-authenticated `gh` CLI
  in its container)? See "Producer + consumer patch design" below.
- ~~What should the new channel be named?~~ **Resolved, PR #1515:** `orion:substrate:codebase_delta`
  (channel), `CodebaseDeltaV1` (schema), `codebase_prediction_error()` (scoring function name).
- ~~What should the new service itself be named?~~ **Still open.** `orion-cocreation-signals`
  remains a working name only, not committed — no service directory exists yet under this or any
  other name. Worth a final gut-check at the producer patch, not before (per "no keyword
  cathedrals," naming happens once there's a real thing to name).
- **New question, raised by shipping the composite scoring function (PR #1515):** where does
  `CodebaseMassBaseline` (the three-domain EWMA state `codebase_prediction_error()` needs threaded
  across ticks) actually persist? Every other domain in `orion/substrate/prediction_error.py`
  stores its baseline directly on a persisted reducer-projection object
  (`ExecutionTrajectoryProjectionV1.prediction_error_baseline_ewma`, etc.) — this domain has no
  such projection (it's bus-event-driven, not reducer-driven), so there is nothing to mutate in
  place. See "Producer + consumer patch design" below for the concrete proposal and why it needs
  verification before implementation, not just a design-mode assertion.

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

### Producer + consumer patch design (elaborated 2026-07-30, after Phase 1 shipped)

Phase 1's pure functions and composite scoring are real and tested (see "Status" above). This
section elaborates the next two patches (producer, then consumer, per CLAUDE.md §5) now that
there's real code to design against, instead of a placeholder file tree.

**Producer patch scope, narrowed from the original file tree.** The original "Dedicated
service" section above sketched six eventual producers (`git_delta`, `graph_delta`,
`pr_lifecycle`, `dev_economics`, `doc_semantic_drift`, `affective_state`) to justify standing up
a dedicated service instead of a slow task. That scale justification still holds, but only three
of those six have shipped code today — `dev_economics.py`, `doc_semantic_drift.py`, and
`affective_state.py` are each still their own separate, unimplemented sibling spec. **The
producer patch should scaffold `services/orion-cocreation-signals/` with exactly the three real
producers (`git_delta`, `pr_lifecycle`, `graph_delta`)**, in a shape that doesn't need
re-architecting to add the other three later (independent per-producer async loops already
gives each new producer its own file/interval/credentials without touching the others) — not
build placeholder stubs for producers that don't exist as real code yet, which would be
schema-valid-but-empty-shell scaffolding, the same failure class this program's metric quality
gate exists to catch.

**GH credentials, resolved by how `pr_lifecycle.py` actually works (see "Status" above):** the
service doesn't need a minted `GITHUB_TOKEN` secret of its own — `pr_lifecycle.py` shells out to
the local `gh` CLI, same as `scripts/analysis/measure_pr_lifecycle.py` already does successfully
in this repo's own dev environment. The service's container needs a working, authenticated `gh`
CLI (mount `~/.config/gh` read-only, or an equivalent `GH_TOKEN` env var `gh` itself honors) —
an operator/deployment concern, not a code-level credential the service manages itself. Simpler
than the original "mint its own GH token" framing assumed.

**Producer scheduling, one interval per producer, matching each domain's real cadence:**

- `git_delta`: cheap and frequent (proposed default: 60s) — a `git rev-parse HEAD` check against
  the last-persisted SHA is nearly free; only a real SHA change triggers the actual
  `git_churn_delta()` diff + publish.
- `pr_lifecycle`: coarser, rate-limit-aware (proposed default: 15min) — `gh pr list` is a real
  network call; no need to poll GitHub every minute for PR lifecycle counts.
- `graph_delta`: event-triggered off graphify updates, not a fixed interval — the original spec's
  own framing (`scripts/safe_graphify_update.sh` appending to `snapshot_history.py`'s log after
  each successful, non-reverted update) is still the right shape; this producer's async loop
  polls `snapshot_history.py`'s log for a new entry it hasn't diffed yet, rather than re-deriving
  "did graphify run" from `graph.json`'s mtime (fragile — a checkout/rebase touches mtimes without
  a real content change).

**Container mounts:** a read-only mount of this repo's own working tree (for `git_delta.py`'s
`git` subprocess calls and `graph_delta.py`'s `git show`/graphify-out reads) — the same repo the
service's own code lives in, mounted read-only into itself, matching how
`scripts/analysis/measure_*.py` already operate directly against a real checkout. No separate
git-clone-inside-the-container step needed.

**Baseline persistence — resolved, 2026-07-31, by reading the real code (not building against the
earlier proposal blind).** The originally proposed answer ("piggyback on `node:substrate.codebase`'s
own metadata") is **wrong** — verified directly against `_write_prediction_error_node()`'s real body
(`services/orion-substrate-runtime/app/worker.py`): it builds a **fresh** `metadata` dict on every
call containing only `source_kind`/`prediction_error`/`reducer_key`, plus whatever's explicitly
carried forward from `DYNAMICS_ENGINE_OWNED_METADATA_KEYS`
(`orion/substrate/falkor_codec.py` — a narrow, semantically-specific allowlist owned by
`SubstrateDynamicsEngine.tick()`: `dynamic_pressure`/`dynamic_pressure_reason`/`dormant`/
`dormancy_updated_at`) and `contributing_turn_ids`. A custom `codebase_mass_baseline` key is not
on that list — it would be **silently dropped on the very next write**, exactly the "tick clobbers
a field it doesn't own" failure class this repo has already hit three times and documented by name
(`execution_load` cross-lane stomp PR #1338, field-digester's generic decay clobber,
`SubstrateDynamicsEngine.tick()`'s `bus_synaptic` clobber PR #1449 — all three cited verbatim in
`services/orion-substrate-runtime/app/store.py::save_attention_self_model()`'s own docstring).

**Real answer: a dedicated, single-writer, append-only table — the exact pattern this repo already
uses for the same problem.** `substrate_attention_self_model`
(`services/orion-sql-db/manual_migration_attention_self_model_v1.sql`,
`store.py::save_attention_self_model()`) is the direct template: one `INSERT ... ON CONFLICT DO
NOTHING` per tick (no read-modify-write, no shared row to clobber), retention-pruned, with a
`generated_at DESC` index for reading the latest row back
(`store.py::get_latest_field_attention_frame()` is the existing "read most recent row" precedent to
mirror). The consumer patch should add:

- `services/orion-sql-db/manual_migration_codebase_mass_baseline_v1.sql`: `create table if not
  exists substrate_codebase_mass_baseline (baseline_id text primary key, generated_at timestamptz
  not null, baseline_json jsonb not null, created_at timestamptz not null default now())` + a
  `generated_at desc` index — same shape as the self-model migration, one field renamed.
- `store.py::save_codebase_mass_baseline()` / `get_latest_codebase_mass_baseline()`: append +
  read-latest, same shape as `save_attention_self_model()`/`get_latest_field_attention_frame()`.
  `baseline_json` is `CodebaseMassBaseline.to_json_dict()`-shaped (needs adding a
  `to_json_dict()`/`from_json_dict()` pair to `CodebaseMassBaseline`, matching
  `GraphSnapshotStats`'s existing convention in `orion/structural_mass/snapshot_history.py`).
- This table has **exactly one writer** (the new consumer tick) and **no pre-existing occupant** —
  the same structural guarantee `save_attention_self_model()`'s docstring names explicitly, not
  "avoided by convention."

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

- ~~`orion/structural_mass/`~~ **Shipped** (PRs #1496, #1500, #1502): `git_delta.py`,
  `graph_delta.py`, `pr_lifecycle.py`, `snapshot_history.py`, tests.
- ~~`orion/substrate/prediction_error.py`~~ **Shipped** (PR #1515): composite
  `codebase_prediction_error()`.
- ~~`orion/bus/channels.yaml`, `orion/schemas/registry.py`~~ **Shipped** (PR #1515):
  `orion:substrate:codebase_delta` channel + `CodebaseDeltaV1` schema.
- `services/orion-cocreation-signals/` (new service, not yet built): scheduling/publishing layer
  — see "Producer + consumer patch design" above for the narrowed, real scope.
- `services/orion-substrate-runtime/app/worker.py`: one new cheap bus-event consumer added to
  the existing fast `_tick()` — no external I/O added to this service. Blocked on the
  `CodebaseMassBaseline` persistence question above.
- `services/orion-field-digester/app/tensor/channels.py`: one new `NODE_CHANNELS` entry.
- `services/orion-field-digester/app/digestion/decay.py`: exclusion breadcrumb comment.
- `scripts/safe_graphify_update.sh`: append to `snapshot_history.py`'s log post-update.
- `services/orion-hub/scripts/attention_organ_routes.py`: add `"codebase"` to
  `KNOWN_PREDICTION_ERROR_DOMAINS`, and decide whether it joins
  `ACTIVE_INFERENCE_DOMAINS` (`orion/substrate/attention_self_model.py`) — a separate semantic
  call from just displaying it, same distinction the `bus_synaptic` note makes. **Not before the
  consumer patch ships real data** — confirmed via investigation 2026-07-30 that adding this row
  today would render an empty/dead panel (no live node yet).
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

~~Phase 1 only, and only its first slice...~~ **Superseded, 2026-07-30 — Phase 1 is done.** The
producer patch is next: scaffold `services/orion-cocreation-signals/` with exactly the three real
producers (`git_delta`, `pr_lifecycle`, `graph_delta`), per "Producer + consumer patch design"
above — no consumer wiring yet, so the service can be built and its own tests/docker-compose
validated in isolation before `orion-substrate-runtime` reads anything from it. The consumer patch
(bus-event consumption, `NODE_CHANNELS` registration, decay exclusion, `substrate_codebase_mass_
baseline` migration) can follow once the producer patch is real — the `CodebaseMassBaseline`
persistence question that previously blocked it is now resolved (see "Producer + consumer patch
design" above, 2026-07-31 update: dedicated single-writer append-only table, same pattern as
`substrate_attention_self_model`).
