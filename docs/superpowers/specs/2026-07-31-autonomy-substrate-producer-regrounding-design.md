# Regrounding the `autonomy` substrate producer on real goal state

Date: 2026-07-31
Status: design only, not implemented. Revised after an adversarial review pass —
see "Review findings incorporated" below before reading the rest at face value.

## Arsonist summary

`orion/substrate/relational/adapters/autonomy_ctx.py`'s `autonomy` producer is registered in `ProducerRegistryV1` at `graphdb_durable` tier, meaning the belief-unification layer treats it as a real, authoritative, durable source. It has been a silent no-op for as long as `AUTONOMY_GRAPH_BACKEND=disabled` has been the live default — confirmed via direct log inspection of the real `orion-athena-cortex-exec-background` container: `autonomy_graph_backend_blocked consumer=autonomy_ctx_adapter ... fallback=skip_adapter` fires on every call, no exception, no signal, nothing downstream ever notices. **Corrected on review: this default has been live since 2026-05-15, roughly two and a half months, not "confirmed dead today" as this doc originally implied by proximity to the same day's unrelated `GraphAutonomyRepository` deletion (PR #1530).** Zero complaints or noticed impact have surfaced in that window — the real motivation for this doc is "the repair material happens to already exist," not a demonstrated live cost of leaving it broken. Its sibling `self_study` (same tier, same dead GraphDB dependency) is in the same state.

The arsonist question: does `autonomy_ctx.py` deserve to exist as a registered producer at all, or is this another `identity_yaml` — machinery around nothing? Traced to its real remaining shape (post the 2026-07-30 drives deletion, which already stripped its drive/tension branches as unreachable dead code): it does exactly one thing — map `AutonomyStateV1.goal_headlines` into `GoalNodeV1` substrate nodes. That's a real, load-bearing shape, more so than this doc originally showed: `GoalNodeV1` nodes are extracted by `_project_autonomy_from_beliefs()` in `chat_stance.py` into `summary.proposal_headlines`, which review traced one hop further than the original draft — that value reaches `ctx["chat_stance_inputs"]`, which is rendered **verbatim into `stance_react.j2`, the real LLM-facing prompt template**. If this producer worked, real goal text would reach the actual prompt Orion generates from, not just an internal summary object. That part of the case is stronger than originally argued, not weaker.

But "the pipe is real" and "what's proposed to flow through it is real" are different claims, and review found the second one false as originally scoped — see below.

This is not the same failure mode as `identity_yaml` (Design finding, 2026-07-31 conversation, not yet written up separately): `identity_yaml` computes nothing and never could regardless of backend — flat hardcoded confidence/salience wrapped around static ctx strings. `autonomy_ctx.py`'s shape is real; only its *data source* is dead. That's a repairable gap, not a keyword cathedral by itself — but review found the originally-proposed repair would have reintroduced the identity_yaml failure mode in a new spot (see below), not avoided it. This repo already built the real repair material without connecting it here: `orion.autonomy.goal_state.get_active_goal()`, a real, live, bus-subscribed cache of `FieldGoalProvenanceV1` (SSP §6 Objective 6, PR #1530), proven live via direct evidence (3 confirmed Redis subscribers on `orion:memory:goals:proposed`, clean `goal_state_listener_started` log lines, zero errors) in two other services already.

## Review findings incorporated

An adversarial review pass (full prompt and findings in this doc's originating conversation) checked this doc's claims against the actual code rather than accepting its own framing. Two real problems, one overstatement, one omission, all now fixed in the sections below:

1. **`goal_text=goal.field_target_id` was wrong-shaped, not wrong-in-spirit.** `field_target_id`'s real live values are raw internal identifiers — `"node:substrate.biometrics"`, `"capability:memory"` — not human-readable goal prose. The field it was replacing held real sentences (`"Reduce predictive uncertainty for hardware_compute_gpu"`). Shipping the raw ID into a slot that (per the corrected trace above) reaches the actual chat prompt would have replaced "silently absent" with "silently wrong-shaped" — the identity_yaml failure mode with a different data source. **Fixed below** by pairing the ID with real windowed dominance metadata instead of pretending it's prose.
2. **`goal.confidence` in the original code sketch was fake.** `FieldGoalProvenanceV1` inherits a flat `confidence: float = Field(default=0.7, ...)` from `GraphReadyArtifact`; the real producer (`orion-attention-runtime`) never sets it. The original sketch's own claim — "real, not decorative, the same bar the audit held the other 10 producers to" — was true for `salience_score` and false for `confidence`, stated as if both held equally. **Fixed below**: `confidence` dropped from the signal bundle sketch, or left at the schema default with an explicit comment that it is not yet real.
3. **Urgency was overstated** — corrected in the Arsonist summary above (2.5 months disabled, not "today," zero noticed impact).
4. **"Delete the producer instead" was never weighed as a competing option**, despite this doc's own evidence that the current dead-path already degrades safely and `chat_stance.py` already has a working, proven fallback. Added as an explicit alternative in Missing Questions below — not resolved, but no longer silently absent from consideration.

## Current architecture

```text
ProducerRegistryV1 (orion/cognition/projection_builder.py)
  producer_id="autonomy", trust_tier=GRAPHDB_DURABLE, anchor_scopes=(orion, relationship, juniper)
  adapter_fn=map_autonomy_ctx_to_substrate (orion/substrate/relational/adapters/autonomy_ctx.py)

map_autonomy_ctx_to_substrate(ctx):
  -> resolve_autonomy_graph_read_plan(ctx)  [orion/autonomy/graph_gate.py]
  -> plan.mode == "disabled" (AUTONOMY_GRAPH_BACKEND=disabled, confirmed live)
  -> returns None, every call, unconditionally

CognitiveUnificationLayer.beliefs_for_stance() [orion/substrate/relational/layer.py]
  -> "autonomy" anchor is permanently cold (no durable node ever lands)
  -> re-fans-out to the dead adapter on every single call (never warms, never caches
     anything, since there is nothing to cache)

chat_stance.py::_project_autonomy_from_beliefs(beliefs, ctx)
  -> reads sl.goals (GoalNodeV1 nodes, node_kind="goal") per anchor
  -> always empty -> function's own `if not any([drives, goals, tensions, snapshots]):
     return None` fires -> chat_stance.py falls through to _load_autonomy_state(ctx),
     which (per today's separate PR #1530) also correctly falls back to an honest
     identity_yaml-hazard-tagged summary

Separately, live and proven today:
orion.autonomy.goal_state.get_active_goal() -> Optional[FieldGoalProvenanceV1]
  populated via orion.autonomy.goal_state_listener subscribed to
  orion:memory:goals:proposed, real producer: orion-attention-runtime (PR #1517)
  Currently consumed by: orion-spark-concept-induction, orion-world-pulse
  (each runs its own bus-subscribed local cache -- see PR #1530's commit history
  for why: GoalContextStore, the substrate-runtime original, is in-process-only)
```

`orion-cortex-exec` (where `autonomy_ctx.py` actually executes) does not currently run a `goal_state_listener` — it has no infrastructure today that would let `map_autonomy_ctx_to_substrate` call `get_active_goal()` and get a real answer. Its bus-integration pattern also differs from the two services PR #1530 wired: `orion-spark-concept-induction`/`orion-world-pulse` use a simple FastAPI `lifespan()` + a manually-managed `OrionBusAsync` connection (mirroring the existing `heartbeat_chassis` pattern in each). `orion-cortex-exec` instead uses a heavier `Rabbit`/`Hunter` chassis (`orion.core.bus.bus_service_chassis`) with several module-level `Hunter(...)` listener instances (`trace_listener`, `core_event_listener`, `embodiment_perception_listener`, `world_context_capsule_listener`) started together in a `starters` list around line 1061 of `services/orion-cortex-exec/app/main.py`. A new goal-state subscription here would most naturally be a new `Hunter`, not a reuse of `goal_state_listener.py`'s bespoke `subscribe()`/`get_message()` loop as written — that loop was built for the simpler lifespan pattern and hasn't been proven against `Hunter`'s handler-callback shape.

## Missing questions

1. **Does `orion-cortex-exec` need its *own* local goal-state cache, or can it read one that already exists?** `orion-attention-runtime` and `orion-substrate-runtime` both already hold real goal state in-process (`GoalContextStore`, the original). `orion-spark-concept-induction`/`orion-world-pulse` each got their own copy because they're separate processes with no shared memory. `orion-cortex-exec` is *also* a separate process, same constraint — but this is the third service asking the same question, which is worth naming explicitly rather than silently building a third near-identical listener: is three independent in-process caches of the same 4-hour-stale-tolerant, single-active-goal state the right shape, or is this the point where a shared read surface (Redis key, small HTTP status endpoint) stops being premature and starts being the honest fix? `goal_state.py`'s own docstring already flags this exact tension as "a documented follow-on, not solved by this module" — this design doesn't have to solve it, but should decide for or against a third copy with that question named, not by default.
2. **`Hunter` vs. `goal_state_listener.py`'s existing loop — adapt one or the other?** `goal_state_listener.py`'s `run_goal_state_listener()`/`start_goal_state_listener()` shape (own `subscribe()` context manager, own polling loop, own stop-event) was built and tested against the lifespan pattern in two services already. Wrapping `_handle_bus_message` in a `Hunter(_cfg(), handler=..., patterns=[...])` instead would be less code in `orion-cortex-exec` specifically but means the "same bus-listener shape" claim across all three services stops being true. Needs a decision, not an assumption either way.
3. **What does `map_autonomy_ctx_to_substrate` do with `get_active_goal()` returning `None`?** Today's dead-backend path returns `None` unconditionally, which is indistinguishable (from `beliefs_for_stance`'s perspective) from "no real goal currently dominant." That's actually the *correct* honest behavior to preserve — the fix here is not "always produce a GoalNodeV1," it's "produce a real one when one exists, honestly emit nothing when it doesn't." Confirm the replacement adapter should keep exactly that shape (return `None` when `get_active_goal()` is `None`), not invent a placeholder.
4. **`self_study`'s fate is a sibling problem, not solved by this doc.** Same tier, same dead backend, same silent-no-op symptom — but its target data (a self-study named graph queried via `SELF_STUDY_NAMED_GRAPH`/GraphDB SPARQL) has no obvious "already built, already live" replacement the way `autonomy` does. Scoping it into this patch would be solving a problem this doc has no real answer for. Left explicitly out — worth its own investigation, separately, into whether `self_study`'s underlying concept has moved to a different live mechanism already (concept induction? something in `orion.spark`?) or whether it should simply be deleted like `GraphAutonomyRepository` was.
5. **Should `identity_yaml`'s retiering (to `snapshot_ephemeral`, matching `social.py`'s already-correct precedent) ride in the same patch, or ship separately?** They're unrelated fixes to sibling producers found in the same investigation. Bundling risks conflating two different classes of fix (misconfigured tier vs. dead data source) in one PR's review; splitting means two small PRs instead of one. Not a blocking question, just a sequencing call.
6. **Delete the producer instead of fixing it — genuinely weighed, not resolved.** The dead-path already degrades safely (`return None` every call, zero exceptions, zero downstream breakage in ~2.5 months) and `chat_stance.py` already has a working, proven fallback (`_load_autonomy_state`, regrounded in PR #1530). Against that: deleting throws away a real, load-bearing pipe (goal text → `proposal_headlines` → the actual LLM prompt, per the corrected trace above) for the sake of removing ~15 lines of already-inert code, when the real repair (`get_active_goal()`) already exists and is proven live elsewhere. The case for fixing over deleting rests entirely on Missing Question 7 resolving well — if the windowed-metadata content ends up too thin to be worth surfacing, delete becomes the honest choice, not fix.
7. **What should real windowed dominance metadata actually contain, and is it enough to make `goal_text` honest?** `orion/attention/field_attention/goal_provenance.py`'s `DominanceStreak`/`update_dominance_streak()` already computes a real consecutive-tick count for the currently-dominant target — it gates whether a goal gets emitted at all (`min_streak` debounce) but is discarded after that boolean check; it never reaches the published `FieldGoalProvenanceV1`. Surfacing it (streak length, and the tick_id where the current streak began) gives real, non-invented context — "this has been the dominant target for N consecutive ticks," not a bare opaque ID. That does not make `field_target_id` into prose, and this doc is not proposing to fake that it does — see the schema section below for the honest scope of what this actually buys.

## Proposed schema / API changes

**Revised on review.** The original version of this section claimed "none" and sketched `goal_text=goal.field_target_id` as if that were a complete, honest fix. It wasn't — see "Review findings incorporated" above. There is one small, real, additive schema change needed to make this honest, plus a corrected (smaller) adapter sketch.

**`FieldGoalProvenanceV1` addition** (`orion/schemas/field_goal.py`) — surfaces data that already exists and is already computed in `orion-attention-runtime`, currently discarded after the emit-gate check:

```python
class FieldGoalProvenanceV1(GraphReadyArtifact):
    # ...existing fields unchanged...
    # How many consecutive real field ticks the SAME target has won the node-target
    # sub-competition, including this one -- the exact value update_dominance_streak()
    # already computes to decide whether to emit at all (min_streak gate). Currently
    # thrown away after that boolean check; this exposes the real number instead of
    # re-deriving or inventing one. Always >= min_streak at emission time (that is
    # what makes emission happen).
    dominance_streak_ticks: int = Field(ge=1)
    # The source_field_tick_id of the FIRST tick in the current streak -- i.e. this
    # record's own source_field_tick_id is the window's end, this is the window's
    # start. Lets a consumer compute real elapsed wall time from two real tick
    # records, not a fabricated duration.
    window_start_field_tick_id: str
```

`orion-attention-runtime/app/worker.py::_maybe_build_goal()` already holds `self._node_streak` (a `DominanceStreak` with `.count` and `.target_id`) at the exact point it decides to emit — it needs to also remember the tick_id of the streak's first tick (not currently tracked; `DominanceStreak` only tracks `target_id`/`count` today) and pass both through to the `FieldGoalProvenanceV1` constructor. Small, additive, non-breaking — existing consumers of this schema ignore fields they don't read.

**Adapter sketch** (`orion/substrate/relational/adapters/autonomy_ctx.py`), corrected:

```python
def map_autonomy_ctx_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
    # ...existing skip-gate checks (verb/lane/skip_unified_beliefs) unchanged...
    goal = get_active_goal()  # orion.autonomy.goal_state
    if goal is None:
        return None
    node = GoalNodeV1(
        anchor_scope="orion",  # real goal-provenance is not yet subject-differentiated;
                                # see PR #1530's own Objective 6 status note on this
        # Honest, not invented prose: a real internal id paired with real elapsed
        # dominance, not dressed up as a human sentence the way the old
        # AutonomyStateV1.goal_headlines[i].headline used to be. See Missing
        # Question 7 -- if this reads too thin in practice, that is real evidence
        # for "delete the producer" (Missing Question 6), not a reason to fake prose.
        goal_text=f"{goal.field_target_id} (dominant {goal.dominance_streak_ticks} ticks)",
        priority=goal.priority,
        temporal=make_temporal(observed_at=goal.ts),
        provenance=_make_prov(subject="orion"),  # source_kind/channel updated to reflect
                                                   # goal_state, not sparql.graph
        # confidence deliberately omitted from the signal bundle here: the real
        # producer never sets FieldGoalProvenanceV1.confidence (it silently inherits
        # GraphReadyArtifact's flat 0.7 default) -- passing it through would repeat
        # exactly the "fake field labeled real" mistake this doc's own review caught.
        # salience is real: the actual per-tick field competition value.
        signals=SubstrateSignalBundleV1(confidence=0.5, salience=goal.salience_score),
        metadata={
            "proposal_signature": goal.artifact_id,
            "source_field_tick_id": goal.source_field_tick_id,
            "window_start_field_tick_id": goal.window_start_field_tick_id,
            "dominance_streak_ticks": goal.dominance_streak_ticks,
        },
    )
    return SubstrateGraphRecordV1(anchor_scope="orion", nodes=[node])
```

`confidence=0.5` here is a placeholder constant, disclosed as such in-line — not claimed as real. If a real confidence signal for goal-provenance emerges later (e.g. derived from `target.confidence_score`, the qualifying-observation-count gate `goal_provenance.py` already enforces before a target can win at all), it should replace this constant then, not before.

## Files likely to touch

```text
orion/schemas/field_goal.py                             # dominance_streak_ticks,
                                                          # window_start_field_tick_id
orion/attention/field_attention/goal_provenance.py       # DominanceStreak needs to also
                                                          # track the streak's first tick_id,
                                                          # not just target_id/count
services/orion-attention-runtime/app/worker.py           # _maybe_build_goal() passes the
                                                          # new fields through
services/orion-attention-runtime/tests/                  # streak/window field coverage
orion/substrate/relational/adapters/autonomy_ctx.py      # the rewrite itself
services/orion-cortex-exec/app/main.py                  # new Hunter (or adapted listener)
                                                          # + settings for CHANNEL_GOAL_PROPOSAL
services/orion-cortex-exec/.env_example                 # CHANNEL_GOAL_PROPOSAL, matching
                                                          # the other two services' pattern
orion/bus/channels.yaml                                 # add orion-cortex-exec to
                                                          # orion:memory:goals:proposed's
                                                          # consumer_services list
orion/substrate/relational/tests/test_adapters.py       # existing autonomy_ctx tests,
                                                          # currently presumably testing
                                                          # the dead SPARQL path -- audit
                                                          # before rewriting
```

## Non-goals

- Not fixing `self_study` (Missing Question 4) — separate investigation.
- Not fixing `identity_yaml`'s tier misconfiguration in this same patch (Missing Question 5) — a smaller, independent, already-understood one-line fix (retier to `snapshot_ephemeral`) that doesn't need this design doc's weight.
- Not resolving Missing Question 1 (shared vs. per-process goal-state cache) as part of implementing this — naming it is this doc's job; deciding it is a separate call, ideally made once rather than per-service.
- Not touching `chat_stance.py`'s own separate autonomy-graph-backend path (`_load_autonomy_state`, already regrounded in PR #1530 to `LocalAutonomyRepository`) — that's a different consumer of a different (already-dead, already-deleted) backend, unrelated to this producer.
- Not re-opening SSP Objective 6's larger scope (capability-gating semantics) — this is purely about what reaches the belief-unification layer for chat-stance narration, not about auto-execute gating.

## Acceptance checks

- `dominance_streak_ticks`/`window_start_field_tick_id` land on real, live `FieldGoalProvenanceV1` records — verified against `orion-attention-runtime`'s own logs/Postgres history, not just unit tests, per this repo's runtime-truth discipline.
- Live evidence that `autonomy_ctx_adapter` produces a real `GoalNodeV1` when a real goal is active: a log line or debug field showing `goal_text`/`priority`/the streak fields matching a real, independently-observed `FieldGoalProvenanceV1` from `orion-attention-runtime`'s own logs at the same timestamp.
- Live evidence it correctly returns `None` (not a stale/fake node) when `get_active_goal()` is `None` — checked via a window where no real goal has been dominant recently.
- `summary.proposal_headlines` (chat_stance.py's autonomy summary) shows real goal text — `"{field_target_id} (dominant N ticks)"`, not a bare ID — during a real chat turn, verified via the same `autonomy_lookup_turn`/equivalent debug-log pattern PR #1530 already used to verify `_load_autonomy_state`. **Read it and judge honestly whether it's worth having** — this is Missing Question 7's real test, not a formality.
- Existing `orion/substrate/relational/tests/test_adapters.py` autonomy coverage updated and passing against the new data source, not the deleted SPARQL path.

## Recommended next patch

Not this patch yet, as scoped. Answer Missing Questions 6 and 7 first, in that order — both are cheap (no code, or a small spike) and both can kill the rest of this doc:

1. **Question 7 first, concretely**: build the `dominance_streak_ticks` schema addition and worker-side wiring alone (small, isolated, low-risk on its own regardless of what happens next), let it run against real field ticks for a day, then actually read what `goal_text=f"{field_target_id} (dominant N ticks)"` looks like against real accumulated data. If it reads as genuinely informative, proceed. If it reads as noise dressed as insight, that's real evidence, not speculation, for—
2. **Question 6**: delete the producer instead, and consider whether `self_study` should go the same way, since both have now sat disabled for 2.5 months with zero noticed cost.

Only after that: answer Missing Questions 1 and 2 (the `orion-cortex-exec` wiring shape) before writing the adapter rewrite itself — they change the actual diff shape (new `Hunter` vs. adapted listener; one more per-process cache vs. a shared surface), and shouldn't be decided by default just because the first two services happened to need their own cache.
