# Field-native goal-provenance producer, goal-context staleness, and decision-lattice observability

Status: design/proposal mode per root `CLAUDE.md` §0A and `orion/sentience_striving_program/
README.md` §2 ("every phase below still requires explicit sign-off before implementation").
Nothing implemented. This does not authorize any change to `capability_policy.py`,
`goal_context.py`, or any Hub route. Written up from a design conversation; every claim below
was traced against real code/config/docs in this session, not recalled from memory.

## Arsonist summary

Three threads, initially raised separately, turned out to be one patch:

1. **SSP §6 Objective 3 names the actual gap**: *"replacing `goal.drive_origin` with a
   field-native goal-provenance concept — this is what actually unblocks item 6"*
   (`README.md:262-263`). This is unbuilt. The channel built for it,
   `orion:memory:goals:proposed`, has `producer_services: []` since the 2026-07-30 drives
   deletion (`orion/bus/channels.yaml:1833-1841`) — nothing publishes to it. The one place
   that still constructs a `GoalProposalV1` at all, `policy_act.py:541`'s
   `goal_proposal_from_episode_intent()`, builds it **synthetically from episode intent**,
   not from field competition — it exists purely to satisfy `capability_policy.py`'s
   `requires_goal_status` schema check on three capability rules. That's the concrete,
   present-tense instance of "performative": the shape of a motivated decision with no real
   motivational content behind it.
2. **`GoalContextStore` (`goal_context.py`) has no time-based expiry**, by its own
   docstring's admission: *"MVP proxy... A full 'highest-priority among all live goals' query
   needs a goal set with expiry — a documented follow-on"* (`goal_context.py:13-15`). It only
   clears on a terminal-status transition of the *currently held* goal
   (`goal_context.py:43-48`). If Part A's producer ever stalls, a stale goal would bias
   voluntary attention forever with nothing to correct it.
3. **The operational surface for all of this is real but scattered and mis-scoped.**
   `services/orion-hub` already has 14+ tabs organized by subsystem name, not by altitude.
   Two looked like the missing "decision engine" mid-level view; only one actually is:
   - `"Pressure"` (`/static/pressure-analytics.html`) calls exclusively
     `/api/substrate/mutation-runtime/*` (`pressure-analytics.html:130-141`) — the **code
     self-mutation proposal system** (`orion/substrate/mutation_proposals.py`/
     `mutation_trials.py`), unrelated to `FieldStateV1`/attention/`orion-proposal-runtime`.
     A naming collision worth fixing on its own, cheaply (Part D).
   - `"Substrate Lattice"` (`substrate_lattice_routes.py`) is the real thing, already built:
     `_load_transport_proof_chain()` (`substrate_lattice_routes.py:166-359`) is a genuine
     M3→M4→M5→L7→L8→L9→L10→L11 join — transport bus projection → `FieldStateV1` capability
     vector → attention frame buckets → proposal candidates → policy decisions → dispatch
     counts → feedback → consolidation motifs, all from real Postgres tables, all with
     per-layer freshness metadata (`_layer_meta()`). It is hardcoded to one lane
     (`"transport"`, via string filters like `"transport" in c.get("target_id", "")`), but
     `_LANES` (`substrate_lattice_routes.py:27-49`) **already lists `biometrics` and
     `execution` as `"status": "planned"`** — the multi-lane generalization was anticipated
     in the data model and never finished, not something to invent from scratch.

None of this needs a new architecture. It needs: one small producer (Part A), one small
expiry check (Part B), and a parameterization of code that already exists (Part C) — plus one
cheap disambiguation (Part D). Consistent with SSP §7's own rule: *"Reuse the live pipeline,
don't parallel it."*

**Same-day context that changes the shape of Part A**, found while grounding this doc: the
field competition Part A reads from got materially richer *today* (2026-07-30,
`docs/superpowers/specs/2026-07-30-candidate-b-hosts-capabilities-live-wiring.md`, status
"implemented, tested, verified against live data"). `select_capability_targets()` no longer
returns `[]` unconditionally — it's live-scored via Candidate B's `novelty_scorer()`, and a
new `select_host_targets()` gives the five physical hosts
(`node:athena`/`atlas`/`circe`/`prometheus`/`rpc_timeout`) real coverage too.
`build_attention_frame()` (`orion/attention/field_attention/builder.py:45-51`) merges node
(Candidate A) + host (Candidate B) + capability (Candidate B) + system targets into one
ranked, capped `dominant_targets` list (`builder.py:79`). Part A's producer should key off
this real, already-merged winner — `frame.dominant_targets[0]` — not just Candidate A's six
`node:substrate.*` domains as an earlier draft of this conversation assumed.

---

## Part A — Field-native goal-provenance producer (SSP §6 Objective 3)

### Current architecture

- **Real, live competition**: `build_attention_frame()` produces `FieldAttentionFrameV1`
  every tick, `dominant_targets` ranked by `salience_score` (Candidate A precision-weighted
  for `node:substrate.*`, Candidate B novelty-only for hosts/capabilities, EWMA-zscore for
  `field:recent_perturbations`). Real, persisted to `substrate_attention_frames`.
- **Real, dead link**: `goal_context_listener.py` subscribes to `orion:memory:goals:proposed`
  and calls `goal_context.py::set_active_goal()` on receipt — code live, flag-gated on
  (`ORION_ATTENTION_TOPDOWN_ENABLED`), channel producer-less. Zero real events since
  2026-07-30.
- **Real, synthetic link**: `policy_act.py:541`'s `goal_proposal_from_episode_intent()`
  builds an in-memory `GoalProposalV1` from `episode_intent` (itself derived from
  `curiosity_signals`/`world_coverage_gap`, `orion/substrate/endogenous_curiosity.py`) to
  satisfy `capability_policy.py`'s `requires_goal_status` check on `web.fetch.readonly`,
  `recall.query.readonly`, `journal.compose.episode`. This path is real and live, but the
  "goal" it carries has no field-competition provenance — it's a schema-shaped stub.
- **`GoalProposalV1`** (`orion/core/schemas/drives.py:156`) lives in the file the 2026-07-30
  deletion was supposed to empty. Kept "write-never" per the PR report — this producer would
  be its first new writer since the deletion, a real question (see Missing questions).

### Missing questions

1. **Where does the producer run** — `orion-attention-runtime` (owns the tick that produces
   `dominant_targets`, would need a new bus-publish side effect added to a process that's
   currently read/compute-only) or `orion-substrate-runtime` (owns `goal_context_listener.py`
   already, would need a new Postgres read of `substrate_attention_frames` it doesn't
   currently do)? Real cross-service seam call, not free either way.
2. **What threshold turns "won the competition" into "worth proposing a goal about"?** Per
   SSP §7 ("measure before minting"), this needs a real measurement pass — reusing the
   z-score/EWMA discipline already validated for `recent_perturbation_zscore` and
   `bus_synaptic`'s `gap_zscore` is the obvious starting point, but the specific target/domain
   and threshold value isn't free to assert here.
3. **Schema fork, real tradeoff**: reuse `GoalProposalV1` (minimal churn against
   `goal_context.py`/`capability_policy.py`'s existing expectations, but writes a new
   producer into a schema module whose own name (`drives.py`) signals a retired system) or
   mint `FieldGoalProvenanceV1` in a new module (clean break, consistent with "kill means
   kill," but touches every consumer). **This document's recommendation, disclosed, not
   asserted as settled**: new schema. `GoalProposalV1`'s only other real field,
   `drive_origin`, is permanently `None` going forward (confirmed dead per the PR report);
   inheriting a schema half of whose fields are dead weight for a genuinely new, field-native
   producer repeats the exact "formalize before validating" pattern SSP's own §6 re-sequencing
   note names as the mistake to avoid.
4. **Does this replace `goal_proposal_from_episode_intent`'s stub, or run alongside it?**
   "Kill means kill" argues replace. SSP's phased discipline argues shadow-measure the new
   producer against real field data first — the stub currently gates three real, live
   capabilities, and breaking that isn't free. This document recommends shadow-only for the
   first patch (see Recommended next patch).
5. **Does Objective 6 (capability_policy ↔ salience coupling) become scoped work once this
   ships?** No — SSP explicitly gates Objective 6 behind Objective 3 closing *and* Objective
   2's AST/HOT reducer being *proven*, not just computed correctly (`README.md:415-419`,
   `README.md:213-217`). This patch closes half of that gate at most.

### Proposed schema / API changes

New schema, `orion/schemas/field_goal.py::FieldGoalProvenanceV1`, registered in
`orion/schemas/registry.py`:

```python
class FieldGoalProvenanceV1(GraphReadyArtifact):
    artifact_id: str
    field_target_id: str          # e.g. "node:substrate.biometrics", "capability:memory"
    target_kind: str              # "node" | "capability" | "system" -- from FieldAttentionTargetV1
    salience_score: float         # the real dominant_targets[0].salience_score that triggered this
    source_field_tick_id: str     # FieldStateV1.tick_id
    source_attention_frame_id: str  # FieldAttentionFrameV1.frame_id
    priority: float                 # normalized salience_score, feeds GoalContext.priority directly
    proposal_status: str            # reuses the same ProposalStatus literal goal_context.py's
                                     # _ACTIVE_STATES already checks against -- no new status vocabulary
    generated_at: datetime
```

No `drive_origin`, no taxonomy label — per O4 (`README.md:123-125`), if named categories ever
emerge they're a *report* on which `field_target_id`s recur, derived later, not asserted here.

Channel: repoint `orion:memory:goals:proposed`'s `schema_id`/`message_kind` in
`orion/bus/channels.yaml:1833-1841` to `FieldGoalProvenanceV1`/`memory.field_goals.proposed.v1`
— a disclosed, intentional contract change (per CLAUDE.md §6), not a silent repoint. The
channel already has the right name and the right consumer wired (`goal_context_listener.py`);
it's the producer and the payload shape that need to change, not the channel itself.

`goal_context_listener.py` and `goal_context.py::GoalContextStore.update_from_goal()` updated
to accept `FieldGoalProvenanceV1` instead of `GoalProposalV1` — same `_ACTIVE_STATES`
status-gating logic, same "latest wins" semantics, different input type.

---

## Part B — Goal-context staleness dead-man's-switch

Not continuous decay. `GoalContextStore`'s "latest wins" replace-on-injection semantics
(`goal_context.py:34-54`) already means a live producer naturally supersedes stale goals —
reaching for a leaky-integrator decay here would invite the exact failure class CLAUDE.md's
metric-quality-gate names twice (`bus_synaptic_prediction_error()`'s permanent 0.27 floor,
`node:substrate.route`'s decay-to-zero artifact) and SSP's own README names as a standing
lesson (§8: *"a decay mechanism's injection cadence must be reconciled against its own decay
rate, or it saturates"*). That tuning cost isn't justified by the actual risk here.

The actual risk is narrower: **if the producer stalls, nothing clears a stale goal**, since
the only clear path is a terminal-status transition of the currently-held goal, which also
requires a live producer. A hard staleness timeout is a dead-man's-switch, not a decay
mechanism — no injection/decay-rate tuning, no rest-state ambiguity.

### Proposed schema / API changes

`GoalContext` (`orion/substrate/attention/top_down.py:83-86`), additive field:

```python
@dataclass
class GoalContext:
    priority: float
    goal_artifact_id: Optional[str] = None
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # new
```

`GoalContextStore.current()` (`goal_context.py:56-57`), staleness check at read time (not
write time — avoids needing a background sweep):

```python
def current(self) -> Optional[GoalContext]:
    if self._current is not None:
        age = (datetime.now(timezone.utc) - self._current.received_at).total_seconds()
        if age > _MAX_GOAL_AGE_SEC:  # new env-configurable constant, real default TBD by measurement
            self._current = None
    return self._current
```

`_MAX_GOAL_AGE_SEC`'s real value is a Missing Question, not asserted here — needs to be long
enough that a normal producer cadence never trips it accidentally (same discipline as
`SUBSTRATE_LATTICE_FRESHNESS_THRESHOLD_SEC`'s existing 60s pattern in
`substrate_lattice_routes.py:649-650`, likely a much longer window here since goals aren't
expected to tick every 30s the way field state does).

---

## Part C — Decision-lattice observability (generalize Substrate Lattice)

### Current architecture

- `_LANES` (`substrate_lattice_routes.py:27-49`): a static list, one real entry
  (`transport`, `"status": "live"`), two placeholder entries (`biometrics`, `execution`,
  `"status": "planned"`) with no `trace_prefix`/`field_capability_id`/`attention_target_id`
  filled in. `GET /api/substrate-lattice/lanes` already returns this list as-is.
- `_load_transport_proof_chain()` (`substrate_lattice_routes.py:166-359`): a real,
  already-built M3→L11 join —
  - M3: `substrate_transport_bus_projection` (latest row) + `substrate_reduction_receipts`
    filtered `WHERE reducer_name LIKE '%transport%'`
  - M4: `substrate_field_state`'s `capability_vectors.get("capability:transport", {})`
  - M5: `substrate_attention_frames`, target buckets filtered for
    `target_id == "capability:transport"`
  - L7: `substrate_proposal_frames`, candidates filtered `"transport" in c.get("target_id",
    "")`
  - L8/L9/L10/L11: `substrate_policy_decision_frames` / `substrate_execution_dispatch_frames`
    / `substrate_feedback_frames` / `substrate_consolidation_frames` — read whole, not
    lane-filtered (these frames aren't per-lane in the current schema)
  - `_layer_meta()` attaches per-layer freshness (stale/fresh vs.
    `SUBSTRATE_LATTICE_FRESHNESS_THRESHOLD_SEC`) to every stage
  - A dead L6 (`SelfStateV1` `transport_integrity`) was already removed 2026-07-22 with an
    explanatory comment (`substrate_lattice_routes.py:335-341`) rather than left stale-forever
    — the precedent this generalization should follow for any lane whose data genuinely
    doesn't exist yet.
- Four routes, all transport-specific: `/transport/latest`, `/transport/gates`,
  `/transport/simulate` (`SimulateRequest`), `/transport/draft-policy-patch`.
- `_compute_gates()`, `_compute_salience()`, `_compute_verdict()` — all read `_TRANSPORT_
  CHANNELS`/hardcode `"transport"` string literals directly.
- Frontend (`substrate-lattice.js`): calls `GET /api/substrate-lattice/lanes` and renders
  `lane.lane_id` generically (line 130) — the lane *list* is already generic — but 28 total
  `"transport"` references elsewhere in the same file mean the actual proof-chain rendering,
  gate display, and simulate/draft-patch panels are still lane-specific, not lane-parameterized.
- **What changed today, materially for this generalization**: before the Candidate B patch
  (Part A's "same-day context" above), `capability_targets` was *never* non-empty in a real
  persisted `substrate_attention_frames` row (confirmed in that doc's own live-data section:
  `"capability_targets has never been non-empty in a real persisted frame, since
  select_capability_targets returned [] unconditionally before this patch"`). M5's real data
  for any capability-keyed lane other than transport (which used the M4 field-vector path, not
  M5's target buckets, for its "has data" check) effectively didn't exist until today.
  Generalizing now has real data behind it in a way it wouldn't have a day ago.

### Missing questions

1. **Which lanes, concretely?** Candidates with real backing data today: `transport` (live,
   unchanged), `biometrics`/`execution`/`chat`/`route` (real `node:substrate.*`
   prediction-error domains, Candidate A), `bus_synaptic` (real, 6th domain, live since
   2026-07-25), plus now capability-keyed lanes (`capability:memory`, `capability:vision`,
   etc., Candidate B, live since today) and host-keyed lanes (`node:athena`, etc., also
   Candidate B, live since today). That's potentially 10+ real lanes — a real scoping
   decision, not "generalize to all of them in one patch."
2. **Do L8-L11 need real lane-filtering, or is whole-frame display honest enough?** Unlike M5/
   L7 (which have per-target/per-candidate `target_id` fields to filter on), policy decision,
   dispatch, feedback, and consolidation frames are read whole in the current transport chain
   — they don't carry a lane-scoped view even for transport today. Generalizing might mean
   accepting the same whole-frame display for every lane (honest, matches current transport
   behavior) rather than inventing new per-lane filtering these frames don't structurally
   support yet.
3. **`_compute_gates()`/`_compute_salience()`/`/simulate`/`/draft-policy-patch`** are built
   around `config/substrate-lattice/transport_lattice_policy.v1.yaml`'s specific
   `dimension_weights`/`channels` shape. Do the other real domains get their own per-lane
   policy YAML (`config/substrate-lattice/<lane>_lattice_policy.v1.yaml`), or is gate/simulate
   functionality transport-only for now, with other lanes getting proof-chain-only (read-only,
   no simulate/draft-patch) coverage in a first pass?

### Proposed schema / API changes

Genericize the join function: `_load_transport_proof_chain(freshness_threshold_sec)` →
`_load_lane_proof_chain(lane: LaneDef, freshness_threshold_sec)`, where `LaneDef` (extending
`_LANES`' existing dict shape) carries the filter predicate each stage currently hardcodes:

```python
@dataclass
class LaneDef:
    lane_id: str
    status: str  # "live" | "planned"
    field_vector_kind: str      # "capability_vectors" | "node_vectors"
    field_vector_key: str       # e.g. "capability:transport", "node:substrate.biometrics"
    attention_target_id: str    # what to match in M5's target buckets
    proposal_target_match: str  # substring match for L7 candidates (transport's current "in" check)
    reducer_name_like: str | None = None  # M3-equivalent, None for lanes with no bus-projection stage
```

Routes generalize from `/transport/*` to `/{lane_id}/*`:
`GET /api/substrate-lattice/{lane_id}/latest`, `GET /api/substrate-lattice/{lane_id}/gates`
(404/`gates_not_configured` for lanes with no policy YAML yet, honest per-lane rather than a
blanket capability), `POST /api/substrate-lattice/{lane_id}/simulate`.
`_LANES`' `biometrics`/`execution` placeholder entries get filled in with real
`field_vector_key`/`attention_target_id` values instead of being dropped, and new entries
added for `chat`/`route`/`bus_synaptic` and the capability/host lanes per Missing Question 1's
scoping decision.

---

## Part D — Pressure Analytics naming disambiguation (cheap, non-blocking)

Rename the Hub tab label from `"Pressure"` (`index.html:101-106`, `id="pressureAnalyticsTabButton"`)
to something that doesn't read as `FieldStateV1` pressure — e.g. `"Code Proposals"` or
`"Self-Mutation"` — and update `pressure-analytics.html`'s own subtitle
(`"Signals → Pressure → Proposals → Trials → Decisions"`) to name the `mutation-runtime`
system explicitly. No backend change, no route change — a label-only fix that removes a real
source of the exact confusion this design conversation hit directly.

---

## Files likely to touch

```
orion/schemas/field_goal.py                          # new -- FieldGoalProvenanceV1
orion/schemas/registry.py                             # register it
orion/bus/channels.yaml                               # repoint goals:proposed schema_id/message_kind
orion/substrate/attention/goal_context.py             # accept new schema, staleness check
orion/substrate/attention/top_down.py                 # GoalContext.received_at
services/orion-substrate-runtime/app/goal_context_listener.py   # new payload type
services/orion-attention-runtime/app/ (or orion-substrate-runtime)  # new producer, per Missing Q1
scripts/analysis/measure_field_goal_provenance_*.py    # shadow-measurement script, SSP §7 pattern
orion/sentience_striving_program/README.md §6 item 3   # status update, same changeset per its convention

services/orion-hub/scripts/substrate_lattice_routes.py # _load_lane_proof_chain generalization
services/orion-hub/static/js/substrate-lattice.js      # de-hardcode the 28 "transport" refs
services/orion-hub/static/substrate-lattice.html       # lane selector UI
config/substrate-lattice/*.yaml                        # new per-lane policy files, per Missing Q3
services/orion-hub/tests/test_substrate_lattice_routes.py

services/orion-hub/templates/index.html                # Part D label rename
services/orion-hub/static/pressure-analytics.html      # Part D subtitle rename
```

## Non-goals

- Not implementing anything in this patch — design/proposal mode only.
- Not hand-authoring a new drive taxonomy or category list anywhere in this design (§7, §10).
- Not flipping `capability_policy.py`'s `required_domain_surprise_below` on any rule — a
  separate, already-scoped design (`2026-07-24-efe-capability-gate-design.md`) with its own
  sequencing, untouched here.
- Not reaching for SSP Objective 6 (capability_policy ↔ salience coupling) — explicitly gated
  behind Objective 2 being *proven*, which this patch doesn't touch.
- Not building continuous decay on `GoalContext` — a staleness timeout only (Part B).
- Not generalizing Substrate Lattice to every possible lane in one patch — Missing Question 1
  is a real scoping decision, not "all of them."
- Not giving every lane simulate/draft-policy-patch functionality — only lanes with a real
  per-lane policy YAML get that; others get read-only proof-chain display.
- Not touching L8-L11's frame schemas to add lane-scoping fields they don't have — accepting
  whole-frame display for those stages, matching current transport-lane behavior, not
  inventing new filtering.
- Not merging or renaming the `mutation-runtime` system itself in Part D — label-only fix,
  the underlying code self-mutation pipeline is untouched and out of scope for this document.

## Acceptance checks

**Part A:**
1. Producer publishes real `FieldGoalProvenanceV1` records over a live 48–96h shadow-measurement
   window, each traceable by id to a specific `FieldAttentionFrameV1`/`FieldStateV1` tick
   (spot-check N random emissions against the source rows).
2. `GoalContextStore` shows at least one real non-empty active goal sourced from the new
   producer (currently: zero, ever, since the 2026-07-30 deletion).
3. Trigger rate is non-degenerate — not always-firing (would recreate `recent_perturbations`'
   old 99.98% monoculture), not never-firing (would recreate the orphaned-channel state it
   replaces). Same live-data-sanity-check gate CLAUDE.md requires for any new metric.
4. `evaluate_capability()` replayed against historical `curiosity_signals` with the new
   producer's goal swapped in for `goal_proposal_from_episode_intent`'s stub returns the same
   or better outcome distribution — regression-safe before any live swap of the stub itself.
5. O2 falsifiability trace: for at least one real dispatched action in the measurement window,
   produce the full chain — field tick → salience winner → goal-provenance record → capability
   decision → dispatch — with real ids at every hop, not narrated.

**Part B:**
6. Staleness check is read-time only, adds no background sweep/new tick.
7. `_MAX_GOAL_AGE_SEC` chosen from real observed producer cadence once Part A has live data —
   not asserted before that data exists.

**Part C:**
8. Each newly-generalized lane's `/latest` endpoint reports honest per-layer freshness
   (`_layer_meta()`'s existing stale/fresh semantics) rather than fabricating data for a stage
   that has none yet — same discipline as the existing L6-removal precedent.
9. No lane's proof chain silently reuses another lane's data (regression test per lane,
   mirroring `test_substrate_lattice_routes.py`'s existing transport coverage).

**Part D:**
10. No behavior change — verified by the fact that this is a label/subtitle-only diff.

## Recommended next patch

Not implementation — still design/proposal mode. In order:

1. **Part D first** — zero-risk, immediately removes the exact confusion this conversation
   hit, no sign-off complexity.
2. **Part A, shadow-only** — smallest real slice: node-target domain lanes only (reuse
   Candidate A's already-computed winner, no new scoring math), publish to the repointed
   channel, wire `goal_context.py` to actually read it, staleness check (Part B) included in
   the same patch since it's two lines and directly de-risks Part A. Do **not** touch
   `policy_act.py`'s synthetic stub or `capability_policy.v1.yaml` in this same patch — shadow-
   measure the new producer against real field data first, per SSP §7.
3. **Part C**, once Part A has real goal-provenance data to display, scoped to the lanes named
   in Missing Question 1's real answer (not all of them) — this is what turns Part A's
   shadow-measurement into something a human can actually watch happen, not just query.

This is cognition-loop-adjacent per CLAUDE.md §0A and SSP's own charter (every phase is a
sign-off gate) — say the word on sequencing/scope and I'll start on whichever slice you pick.

## Source material

- `orion/sentience_striving_program/README.md` §6 item 3, §8 — the named unbuilt objective and
  the deletion's own accepted-consequence language.
- `docs/superpowers/pr-reports/2026-07-30-delete-orion-drives-pr.md` — full deletion scope,
  the death-certificate quote.
- `orion/bus/channels.yaml:1833-1841` — the orphaned channel's own comment.
- `orion/autonomy/policy_act.py:61-98,541-608` — the synthetic-goal stub, traced call sites.
- `orion/autonomy/capability_policy.py`, `config/autonomy/capability_policy.v1.yaml` — the real
  gate chain the stub feeds.
- `orion/substrate/attention/goal_context.py`, `top_down.py:83-86` — `GoalContextStore`'s own
  documented MVP gap.
- `orion/attention/field_attention/builder.py:21-86`, `selectors.py:335-351` — the real,
  merged `dominant_targets` competition.
- `docs/superpowers/specs/2026-07-30-candidate-b-hosts-capabilities-live-wiring.md` — same-day
  patch that materially changes what Part A should read from.
- `services/orion-hub/scripts/substrate_lattice_routes.py:27-359,636-712` — the real M3-L11
  join and its `_LANES` placeholder entries.
- `services/orion-hub/static/js/substrate-lattice.js`, `pressure-analytics.html:130-141` — the
  frontend hardcoding and the naming-collision evidence.
- `config/substrate-lattice/transport_lattice_policy.v1.yaml` — the one real per-lane policy
  file, the template for Missing Question 3.
- `docs/superpowers/specs/2026-07-24-efe-capability-gate-design.md` — sibling design in the
  same program, referenced for what's explicitly *not* touched by this document.
