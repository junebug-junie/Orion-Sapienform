# SelfStateV1 / phi / endogenous-origination burn — spec

Status: **spec, not implemented.** Supersedes the brainstorm doc
(`2026-07-22-self-state-phi-burn-brainstorm.md`) with concrete decisions and a rollout order.
Nothing here has been built yet.

## Decisions (Juniper, this thread)

1. `SelfStateV1` — the module, the service, the schema, the table, the hand-picked coefficients —
   killed outright. Reform (equal-weighting) was prototyped and reverted; not the direction.
2. `orion/autonomy/endogenous_origination.py` — killed outright. No other data source exists for it;
   there is nothing to strip down to.
3. Phi (`orion-spark-introspector`'s `inner_state.py`) — **stripped, not retired.** Keep the 4 real
   cognitive features (`recall_gate_fired`, `reasoning_present`, `execution_load`, `reasoning_load`),
   drop the 4 self-state-derived trainable slots (`agency_readiness`, `execution_pressure`,
   `reasoning_pressure`, `overall_intensity`) and all 10 `FELT_DIMENSIONS` rows, retrain the encoder
   on the smaller feature space. `honest_headline()` has zero non-self-state inputs today — it does
   not survive as-is; either dropped or rebuilt from scratch on the 4 cognitive features once there's
   a real formula for that, not assumed here.
4. `orion-topic-foundry` (the service) — confirmed zero self-state dependency, untouched, not
   discussed further in this spec.
6. `orion/proposals/scoring.py` (Layer 7 of the canonical pipeline) — **in scope, added after the
   spec's first pass.** It loses its only input when self-state dies, and needs a live replacement
   regardless of whether the L7-L11 ladder itself is ever revived. Point it at `FieldStateV1` directly
   instead of self-state's now-dead compression of the same data — consistent with the
   already-established field-native-over-self-state direction (see "Confirmed non-impacts and a
   scoped follow-up" below for the full reasoning). This fixes Layer 7's input; it does not revive
   the ladder (that's still out of scope, see below).
5. `orion/spark/concept_induction/` (`tensions.py`, `drives.py`, `bus_worker.py`) — **open, defaulted
   below, override if wrong.** Not the same system as topic-foundry. Its own `CLAUDE.md` already
   halted new development here on 2026-07-18 (found to be "a poorer reimplementation of Layers 4-9
   of the already-live canonical pipeline"), kept only until replacement wiring lands — that halt
   predates and is independent of this burn. `tensions.py::extract_tensions_from_self_state()` reads
   `SelfStateV1` directly and is wired to endogenous-origination
   (`tests/test_endogenous_origination_wiring.py`). **Default**: strip its self-state input the same
   way everything else loses one — `extract_tensions_from_self_state()` and its
   endogenous-origination wiring go away — but the broader `DriveEngine`/bucket-voting machinery is
   NOT additionally touched by this spec; it was already halted before this thread started and this
   spec doesn't newly decide its fate. If you want the whole directory left alone instead
   (self-state input included, even though it'll be reading a dead schema), say so before this
   ships.

## Rollout order (contract-first, per CLAUDE.md §5)

Multi-service change — sequencing matters so nothing crashes reading a producer that's already gone.

1. **Consumer patch first, producer last.** Every consumer below gets its self-state dependency
   removed/degraded *before* the producer (`orion-self-state-runtime`) stops running — never the
   reverse, or every live consumer errors on a sudden missing input mid-flight.
2. Stop `orion-self-state-runtime` writing `substrate_self_state` (flag-gated off, not deleted yet —
   a rollback window before the irreversible step).
3. Confirm no consumer errors for a real observation window (the runtime health monitor already
   watches `substrate_self_state` staleness — repurpose that window as the confirmation signal).
4. Delete `orion/self_state/`, `services/orion-self-state-runtime/`, schema, registry entry.
5. **DB drop is a separate, explicitly-gated step** (`substrate_self_state`,
   `self_state_predictions`, `identity_snapshots` — confirmed all three are self-state-owned, written
   only by this service's own `store.py`, not shared with a broader identity system). Per CLAUDE.md
   §13, `DROP TABLE` needs explicit approval at execution time regardless of this spec being signed
   off — this spec proposes it, it does not authorize running it.

## Per-consumer disposition

| Consumer | Current null-handling | Disposition |
|---|---|---|
| `orion/collapse/service.py` | **Already null-guards** (`self_state: SelfStateV1 \| None`, explicit `is None` check, returns `None` cleanly) | Near-zero work — delete the dead branch, done |
| `orion/proposals/scoring.py` | No guard | **In scope (added after first pass, decision 6).** Remove `dimension_score`/`dimension_confidence` self-state helpers, replace Layer 7's input with a direct `FieldStateV1` read. Not a like-for-like port — self-state's dimension scores were a (broken) compression of field-channel data; the new read should go at raw field channels, not re-derive self-state's dead formulas on the new source |
| `orion/substrate/metacog_trigger_signals.py` | No guard | Remove the `overall_condition` reflection-trigger reason; other trigger reasons (if any) unaffected |
| `orion-cortex-exec/chat_stance.py` | **Already no-ops** — `_project_self_state_from_beliefs()` returns `None` the moment `anchor.concepts` has no `self:`-prefixed nodes (existing guard, line ~1310-1312) | Zero code change needed. Once the sole producer (`self_state_ctx.py`) is deleted, no `self:` nodes ever exist, and this function no-ops on its own. Confirmed single producer/consumer pair — no third party reads that belief node |
| `orion/autonomy/endogenous_origination.py` | N/A | Deleted outright (decision 2) |
| `orion/substrate/relational/adapters/self_state_ctx.py` | N/A | Deleted outright — it's the sole producer of the belief node above |
| `orion/consolidation/motif.py`, `tensorize.py` | No guard | Remove self-state-keyed motif matching / tensorization fields |
| `orion/identity/snapshot.py` | No guard | Remove `overall_condition`/`overall_intensity` fields going forward; existing persisted snapshots keep their old values, untouched |
| `orion-hub` observability/lattice routes | No guard | Remove routes. Per CLAUDE.md §9: check the rendered Hub template too, not just the route — if a panel links to it, that panel needs removing/replacing, not left pointing at a 404 |
| `orion/spark/concept_induction/tensions.py` | No guard | See decision 5 above — defaulted, not fully resolved |

## Phi retrain scope

- Drop from `FELT_DIMENSIONS` iteration entirely (not just from the trainable subset) —
  `build_inner_state_features()`'s required `ss: SelfStateV1` argument goes away; the function's
  signature changes to take the trajectory/reasoning-activity projections directly, no self-state
  object at all.
- `honest_headline()` — no replacement formula proposed here. Either the `InnerStateFeaturesV1`
  payload ships without a headline field populated (explicit, honest "not computed" rather than a
  fabricated placeholder — CLAUDE.md's "no empty-shell cognition" applies directly here), or a new
  formula gets designed separately, later, from the 4 real cognitive features. Not decided in this
  spec.
- Encoder retrain: seed-v4's trainable set drops from 8 to 4 dims
  (`recall_gate_fired`, `reasoning_present`, `execution_load`, `reasoning_load` only). This is a new
  features-version (`seed-v5`?), not a live in-place change to `seed-v4`'s already-fit weights —
  same precedent as the existing `seed-v2`/`seed-v3`/`seed-v4` versioning already in the file.
- `metadata.overall_condition`/`trajectory_condition`, `self_state_id`, `liveness.self_state` — all
  self-state-sourced fields on `InnerStateFeaturesV1`, removed or defaulted to an honest absent state.

## README annotation plan

Every deleted/modified file's replacement location gets a dated note explaining what was here and
why it's gone, not just silent removal — per Juniper's "annotate the shit out of readmes"
instruction from earlier in this thread. At minimum:

- `orion/self_state/README.md` — deleted with the module, but its content (especially the
  `inner_state_registry.py` composition-status taxonomy: COMPOSED/SHADOW/DUPLICATE/REHEARSAL) is
  real prior art worth preserving somewhere before the file goes, not just discarded.
- `services/orion-self-state-runtime/README.md` — deleted with the service.
- `services/orion-spark-introspector/README.md` — dated note: seed-v4→seed-v5 feature-set cut,
  why (self-state proven pinned/flat + hand-tuned, 2026-07-22), what survived and why (4 real
  cognitive features), what didn't get a replacement yet (`honest_headline`).
- `orion/autonomy/README.md` — dated note: `endogenous_origination.py` removed, why (no data
  source left once self-state is gone), pointer to this spec for the full trace.
- `orion/proposals/`, `orion/consolidation/`, `orion/identity/` — wherever each has a README,
  a one-line note on what field/helper was removed and why, pointing back to this spec rather than
  re-explaining the whole trace in each location.
- New top-level pointer: this spec + the brainstorm doc are the canonical trace; every touched
  README should link back here rather than duplicate the reasoning.

## Confirmed non-impacts and a scoped follow-up (added after Juniper's review pass)

- **`AutonomyStateV2`: zero impact.** `endogenous_origination.py` does not reference
  `AutonomyStateV2` or any state store — confirmed via grep, no match at all. It only reads
  `SelfStateV1` and (presumably) publishes proposals when it decides to originate. `AutonomyStateV2`
  is produced by the fully separate homeostatic-drives/deviation-gated reducer. Killing self-state
  and endo-origination doesn't touch it.
- **L7-L11 ladder / field-digester swap — the input fix is now in scope (decision 6), reviving the
  ladder itself is not.** The ladder (`ProposalFrameV1 → ... → ConsolidationV1`, 5 services) is
  independently confirmed `REHEARSAL` in `inner_state_registry.py`: `EXECUTION_DISPATCH_MODE=dry_run`
  live, every external reader is a self-labeled Hub debug route except `orion-thought` reverie
  grounding, which "only appends an inert ID tag to an already-generated thought." That deadness is a
  **downstream consumption problem**, independent of what feeds Layer 7 (Proposal) — swapping
  `orion/proposals/scoring.py` onto `FieldStateV1` (decision 6) fixes Layer 7's input, since proposal
  scoring needs a live replacement regardless of self-state's fate, but it does not revive the
  ladder. Reviving the ladder needs real dispatch mode + real downstream consumers, a separate, much
  bigger project, still out of scope here.

## Open items / call-outs not yet resolved

- **`orion/self_state/inner_state_registry.py` collateral damage.** This registry tracks *every*
  "what does Orion currently feel/perceive" signal in the repo, not just `SelfStateV1` —
  `DriveStateV1`, `AutonomyStateV2`, phi (both halves), `BiometricsClusterV1`, the L7-L11 ladder,
  `mood_arc_corpus.v1` all have entries here. Deleting `orion/self_state/` wholesale takes this
  registry down with it, along with its CI gate
  (`scripts/check_inner_state_registry.py`/`make check-inner-state-registry`) that currently guards
  against silent signal duplication repo-wide. That gate has real, independent value unrelated to
  whether `SelfStateV1` itself is good — recommend moving the registry (not its self-state entry, the
  file itself) to a neutral location (e.g. `orion/inner_state_registry.py`) before deleting
  `orion/self_state/`, rather than losing the gate as a side effect. Not decided — flagging before
  this ships.
- **`orion/spark/concept_induction/` fate** — defaulted above (decision 5), not confirmed.
- **DROP TABLE timing** — proposed, not authorized; needs a separate explicit go-ahead at execution
  time per CLAUDE.md §13.
- **Live service cutover risk** — `orion-self-state-runtime` is currently deployed and running;
  section "Rollout order" above sequences consumer-first specifically to avoid a window where a live
  consumer queries a service that's already gone. Docker/restart commands to be listed at
  implementation time, not here.
- **Hub frontend** — flagged in the consumer table; not independently verified against the actual
  rendered template yet (CLAUDE.md §9 requires checking the template, not just the route, before
  calling frontend work done).

## Non-goals

- Not deciding `orion/spark/concept_induction/`'s DriveEngine/bucket-voting fate beyond its
  self-state input — that halt predates this thread.
- Not designing phi's replacement headline formula.
- Not executing the DB drop.
- Not touching `orion-topic-foundry` — confirmed unrelated, out of scope.
