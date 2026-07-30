# Goal-system remaining gaps: design session

Date: 2026-07-30
Status: design only, not implemented
Depends on: `docs/superpowers/specs/2026-07-30-goal-provenance-and-decision-lattice-observability-design.md` (Parts A/B/D shipped, PR #1517), `orion/sentience_striving_program/README.md` §6

## Arsonist summary

Juniper asked, after PR #1517 shipped: "is goals shit we wanted to burn... does it use shit design thats ungrounded." The answer given at the time was: the channel-level producer (`FieldGoalProvenanceV1`) is real and grounded, `GoalProposalEngine`/`drive_origin` gating is already deleted (PR #1486). That answer was correct but incomplete — it covered one lane (`goal_context.py`'s top-down attention bias) and left three more "goal" surfaces unaudited. Asked to "hit it all," this doc audits the rest with the same rigor: trace to real code, don't assert.

Four real gaps found, each independently gated (this doc proposes work, does not authorize it):

- **E. `capability_policy.py`'s live gating still runs on a synthetic stub**, not real goal state — `evaluate_capability()`'s `ctx.goal` is populated exclusively by `policy_act.py::goal_proposal_from_episode_intent()`, a function that fabricates a `GoalProposalV1` from an episode intent on every call rather than reading anything `orion-attention-runtime` or `goal_context.py` produces. This is the exact gap SSP §6 Objective 6 names and gates on Objective 3 being proven first — Objective 3 (this session's PR #1517) is now proven, so Objective 6 is unblocked, not yet started.
- **F. `drive_origin` is a zombie required field, write-only in ~10 files**, since `capability_policy.py` stopped reading it 2026-07-30 (`chore/delete-orion-drives` Wave 2a). Every writer sets it to a hardcoded literal (`"predictive"`, `"supervisor"`) because the schema still requires it, not because it carries real information. This is the keyword-cathedral shape CLAUDE.md §0A names directly: a label with a producer but no live consumer.
- **G. The default autonomy-repository backend points at deleted infrastructure.** `build_autonomy_repository(backend="graph")` is the default in both call sites that matter (`chat_stance.py`, `autonomy_ctx.py`, both via `AUTONOMY_REPOSITORY_BACKEND` env, defaulting to `"graph"` when unset or invalid) — a Fuseki-backed SPARQL repository. There is no `services/orion-fuseki` directory left in this repo (confirmed via `ls services/`), consistent with the 2026-07-28 full RDF/Fuseki service deletion already on record. The `"local"` fallback backend is a pure stub that always returns `availability="empty"` regardless of subject — not a real alternative, a permanent no-op. Net effect: the live chat-stance goal/autonomy lookup path defaults to a dead endpoint with a no-op fallback, degrading silently (graceful error classification exists — `_classify_query_error`/`_bounded_reason` — so this doesn't crash chat, it just always returns empty) rather than loudly. This is CLAUDE.md §0A's "no empty-shell cognition" pattern: a real-looking degraded-mode return masking that no goal context has reached chat stance since Fuseki was removed.
- **H. Two disclosed-but-unclosed loose ends from PR #1517 itself**: `ORION_GOAL_PROVENANCE_MIN_STREAK=3` is an uncalibrated debounce constant, and Part C (generalizing Hub's Substrate Lattice UI to show real goal-provenance ticks) was explicitly deferred.

None of these four are new problems this session created — G and F predate PR #1517 by weeks (drive_origin removal was 2026-07-30's `chore/delete-orion-drives`; Fuseki deletion was 2026-07-28). This doc's job is naming them with evidence, not implementing fixes.

## Current architecture

### E. capability_policy.py's real vs. synthetic goal input

```text
orion/autonomy/policy_act.py::goal_proposal_from_episode_intent(intent)
  -> fabricates GoalProposalV1 in-memory, never published to any bus channel
  -> proposal_status hardcoded "proposed", drive_origin = intent.drive_origin (itself
     hardcoded "predictive" at every call site, see Part F)
  -> fed into CapabilityEvaluationContext.goal
  -> orion/autonomy/capability_policy.py::evaluate_capability() reads
     ctx.goal.proposal_status against rule.requires_goal_status (missing_goal /
     goal_status_insufficient / requires_promote reason codes) -- REAL gating,
     just fed a synthetic, always-"proposed", never-"planned"/"executing" input
```

Meanwhile, `orion-attention-runtime` (PR #1517) now produces real `FieldGoalProvenanceV1` artifacts with a real `proposal_status` field, consumed by `orion/substrate/attention/goal_context.py::GoalContextStore`, which biases top-down attention. **These two goal notions do not talk to each other.** `capability_policy.py`'s gate and `goal_context.py`'s attention bias are two independent, parallel "what is Orion's current goal" answers, one real-field-driven, one synthetic-per-call. `_EPISODE_JOURNAL_CAPABILITY`'s and `_READONLY_CAPABILITY`'s auto-execute gates therefore never actually reflect what the field says is salient.

### F. drive_origin's remaining footprint (write-only)

Confirmed via `rg -n "drive_origin"` across `orion/` and `services/`, excluding tests:

| File | Role | Status |
|---|---|---|
| `orion/autonomy/policy_act.py:473,493,517` | writes `"predictive"` literal into store slot key + synthetic goal | write-only |
| `services/orion-world-pulse/app/services/curiosity.py:91` | writes `"predictive"` literal, own comment says "no gate reads this anymore" | write-only, self-documented dead |
| `services/orion-cortex-exec/app/autonomy_goal_execute.py:38,74,143` | `drive_origin: str = "supervisor"` field default, logged | write-only |
| `services/orion-cortex-exec/app/supervisor.py:1085` | reads from a graph row, defaults to `"supervisor"` | reads a value nothing writes meaningfully anymore |
| `orion/autonomy/summary.py:148-176,278` | `dedupe_goal_headlines_by_drive_origin()` — dedup key for chat-visible goal headlines | **live consumer**, but the key it dedupes on is a near-constant now (mostly `"predictive"`/`"supervisor"`), so dedup is close to a no-op |
| `orion/autonomy/goal_actions.py`, `orion/autonomy/repository.py`, `orion/autonomy/goal_archive.py` | SPARQL `orion:driveOrigin` predicate, read/write | queries a Fuseki backend that no longer exists (see Part G) |
| `orion/schemas/attention_frame.py:99` | `goal_drive_origin: str | None = None` | already `None`-by-default post Wave 2b per `top_down.py`'s own comment (line 245-246) |
| `orion/core/schemas/drives.py:160,188` | `GoalProposalV1.drive_origin: str`, required (no default) | the schema constraint forcing every writer above to keep inventing a value |

`orion/autonomy/summary.py`'s `dedupe_goal_headlines_by_drive_origin` is the one place this still does real (if degenerate) work — it's a live consumer, so `drive_origin` is not 100% dead, just close to it.

### G. AutonomyRepository's graph backend and Fuseki

```text
AUTONOMY_REPOSITORY_BACKEND env (default unset) -> "graph" (both call sites coerce
  invalid/missing values to "graph", chat_stance.py:1909-1911, autonomy_ctx.py:171-173)
  -> build_autonomy_repository(backend="graph") -> GraphAutonomyRepository
  -> SparqlHttpClient against a Fuseki endpoint
  -> services/orion-fuseki/ : DOES NOT EXIST (confirmed via ls services/)
```

`services/orion-hub/scripts/drives_analytics_queries.py:664` hardcodes `backend="graph"` directly (not env-gated at all). `LocalAutonomyRepository.get_latest()` (lines 222-226) is the only fallback and is a pure stub — `state=None, availability="empty"` unconditionally, not a real local-storage read path.

This is the same "goal" surface that feeds `chat_stance.py`'s stance construction — i.e., whatever residual "what does Orion want" text ever reached a chat turn through this path has most likely been reading empty/unavailable since 2026-07-28, silently.

## Missing questions

1. **Should `capability_policy.py` read `GoalContextStore.current()` directly, or should there be a dedicated adapter that maps `FieldGoalProvenanceV1` -> `GoalProposalV1`-shaped context?** `evaluate_capability()`'s `CapabilityEvaluationContext.goal` is typed `GoalProposalV1 | None` and reads `.proposal_status` — `FieldGoalProvenanceV1` already has `proposal_status` (same `ProposalStatus` literal type) but not `goal_statement`/`proposal_signature`/`drive_origin`. Two paths: (a) change `ctx.goal`'s type to `FieldGoalProvenanceV1 | None` and update `evaluate_capability()`'s few field reads (small, since it only ever reads `.proposal_status`), or (b) keep the synthetic-shape contract and adapt. (a) is smaller and removes the last `GoalProposalV1` dependency from this file's live path. Needs a decision, not a default.
2. **Does killing `drive_origin` as a required field break `orion.autonomy.summary.dedupe_goal_headlines_by_drive_origin`'s real (if degenerate) behavior, and does anyone care?** If `GoalProposalV1` itself is being phased toward deletion (Part E may make it unreferenced outside `goal_actions.py`/`repository.py`/`goal_archive.py`'s dead SPARQL path), this dedup function's fate needs a call: keep dedup keyed on something else (e.g. `field_target_id` once goal headlines are field-native), or accept it becomes moot once headlines come from `FieldGoalProvenanceV1`.
3. **Is Fuseki actually, fully gone, or does an out-of-repo Fuseki instance still answer these queries?** This doc treats "no `services/orion-fuseki/` directory" plus the prior session's 2026-07-28 full-deletion finding as sufficient evidence, per the "converge, don't chase" discipline — but the exact next patch should open with a live check (`curl` the configured `AUTONOMY_REPOSITORY_ENDPOINT`/equivalent, or grep the running container list) before deleting `GraphAutonomyRepository`, since deleting a repository class that's silently still working would be a real regression, not a cleanup.
4. **What should replace the graph backend as chat-visible goal/autonomy state?** If Fuseki is confirmed dead, `LocalAutonomyRepository`'s stub needs either a real backing store (Postgres, matching `goal_context_listener.py`'s `GoalContextStore` persistence) or the whole `AutonomyRepository` abstraction needs to be pointed at `GoalContextStore`/`FieldGoalProvenanceV1` directly, collapsing two "what does Orion want" reads into one. This is architecturally the same shape as Part E's question, at a different layer (chat-stance narration vs. capability gating) — worth deciding together, not separately, to avoid building two different field-native goal adapters.
5. **`min_streak` calibration**: what real signal would validate 3 as the right debounce vs. 2 or 5? SSP's own "measure before minting" precedent (item 5's emergent-clustering probe) suggests replaying `substrate_attention_frames` history once a few days of real Candidate-A node-target competition has accumulated, and checking streak-length distribution empirically rather than picking a number a priori twice.

## Proposed schema / API changes

None proposed by this doc directly — each part below needs its own follow-up design-or-implement decision. Sketching the shape for scoping purposes only:

- **Part E**: `CapabilityEvaluationContext.goal: FieldGoalProvenanceV1 | None` (was `GoalProposalV1 | None`); `capability_policy.py` deletes its `GoalProposalV1` import. Real caller change: `policy_act.py`'s three `evaluate_capability()` call sites (`_READONLY_CAPABILITY`, `_RECALL_CAPABILITY`, `_EPISODE_JOURNAL_CAPABILITY`) read `GoalContextStore.current()` instead of calling `goal_proposal_from_episode_intent()`. Open sub-question from Missing Question 1: `GoalContextStore.current()` returns at most one active goal (global), while today's three call sites each construct a per-episode synthetic goal — collapsing to one global goal changes semantics (a capability gate keyed to "is there currently a dominant field target" rather than "is there a goal for this specific episode"). That's a real behavior change needing explicit sign-off, not a mechanical swap.
- **Part F**: once Part E ships and `GoalProposalV1` has no remaining non-dead reader of `drive_origin` outside `orion.autonomy.summary`, either (a) make `drive_origin: str | None = None` (soft-deprecate, matches `attention_frame.py`'s `goal_drive_origin` precedent) or (b) delete the field outright if `dedupe_goal_headlines_by_drive_origin` is also retired in the same patch (kill-means-kill). No schema change proposed until Missing Question 2 is answered.
- **Part G**: no schema change; a repository-backend default flip (`AUTONOMY_REPOSITORY_BACKEND` default `"graph"` -> `"local"` or a new field-native backend) plus, if going the field-native route, a new `FieldAutonomyRepository` reading `GoalContextStore`/Postgres directly instead of SPARQL.
- **Part H**: no schema change for min_streak (already a settings field); Part C's Hub UI change is additive (a real-ticks panel on the existing Substrate Lattice page, scoped to node-target lanes per the original design doc).

## Files likely to touch (per part, once each gets its own implement sign-off)

**E:**
```text
orion/autonomy/capability_policy.py
orion/autonomy/policy_act.py
orion/autonomy/models.py (CapabilityEvaluationContext usages elsewhere, if any)
services/orion-world-pulse/app/services/curiosity.py (second evaluate_capability call site)
orion/autonomy/tests/test_policy_act.py
orion/autonomy/tests/test_capability_policy.py
```

**F:**
```text
orion/core/schemas/drives.py
orion/autonomy/summary.py
orion/autonomy/policy_act.py
services/orion-world-pulse/app/services/curiosity.py
services/orion-cortex-exec/app/autonomy_goal_execute.py
services/orion-cortex-exec/app/supervisor.py
orion/schemas/registry.py (if GoalProposalV1 usage narrows enough to reconsider registration)
```

**G:**
```text
orion/autonomy/repository.py
orion/substrate/relational/adapters/autonomy_ctx.py
services/orion-cortex-exec/app/chat_stance.py
services/orion-hub/scripts/drives_analytics_queries.py
services/orion-cortex-exec/.env_example, services/orion-hub/.env_example (AUTONOMY_REPOSITORY_BACKEND default)
```

**H:**
```text
services/orion-attention-runtime/app/settings.py (ORION_GOAL_PROVENANCE_MIN_STREAK — if calibration changes the default)
scripts/analysis/measure_goal_provenance_streak_distribution.py (new, mirrors measure_emergent_clustering_probe.py's shape)
services/orion-hub/ (Substrate Lattice panel, per original design doc's Part C scoping)
```

## Non-goals

- This doc does not implement anything. Each part (E/F/G/H) needs its own explicit "go" per CLAUDE.md §0A's proposal-mode requirement for cognition/autonomy-adjacent changes — capability gating and chat-visible goal state both qualify.
- Not proposing to resurrect Fuseki. Part G's question is what replaces it, not whether to bring it back.
- Not proposing to delete `GoalProposalV1` in this doc — Missing Question 2 must be answered first, and `orion.autonomy.summary`'s live (if degenerate) consumer must be accounted for.
- Not re-opening Objective 4/5/7 (consciousness-theory instruments, emergent-clustering, integration) — those are sequenced after Objective 6 in the charter and untouched by this doc.
- Not proposing a redesign of `evaluate_capability()`'s rule structure (`capability_policy.v1.yaml`'s shape) — only its goal-context input source.

## Acceptance checks

- **E done** means: `evaluate_capability()`'s three real call sites read `GoalContextStore.current()` (or its field-native successor), `goal_proposal_from_episode_intent()` is deleted or demonstrably still needed for a documented reason, and a live replay (or at minimum a targeted unit test using real recorded `FieldGoalProvenanceV1` rows) shows at least one capability decision changing outcome (allowed/denied/requires_promote) between the old synthetic input and the new real one — proving this wasn't a no-op swap.
- **F done** means: every remaining `drive_origin` write site either reads from something real or the field is removed/defaulted; `rg drive_origin` across the repo (excluding historical docs/PR reports) returns only genuinely-load-bearing hits.
- **G done** means: a live check confirms Fuseki's status one way or the other (not assumed), `AUTONOMY_REPOSITORY_BACKEND`'s default no longer points silently at dead infra, and `chat_stance.py`'s goal/autonomy narration has a real, traceable, non-"always empty" data path — verified with a live chat-turn trace showing the new path populated, not just code review.
- **H done** means: `min_streak` has a real distribution-based justification (or is explicitly kept at 3 with the measurement showing why), and Hub's Substrate Lattice shows real `FieldGoalProvenanceV1` ticks with a visible trace ID back to the producing attention-runtime tick.

## Recommended next patch

**Part G first**, not E. Reasoning: G is the one with an active, currently-silent failure mode in a chat-visible surface (goal/autonomy narration defaulting to empty against dead infra) — that's the closest thing to "shit design that's ungrounded" actually running live right now, versus E/F which are internal gating correctness issues with no evidence yet of user-visible harm. G also directly informs Missing Question 4, which E's implementation should ideally already know the answer to before building a second, possibly-redundant field-native adapter.

Concretely: live-verify Fuseki's status (Missing Question 3) as a fast, cheap, read-only first step — either it's confirmed dead (expected) and the fix is flipping `AUTONOMY_REPOSITORY_BACKEND`'s default plus giving `LocalAutonomyRepository` (or a new field-native repository) a real backing read, or it's alive somewhere unexpected and this whole part's severity drops to zero. Either outcome is fast to reach and changes what "next" means, so it should run before committing to an implementation plan for G.
