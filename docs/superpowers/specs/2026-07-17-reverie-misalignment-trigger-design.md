# Reverie misalignment trigger — design spec

Status: DESIGN, not implemented. Grounded live 2026-07-17 against `main`
against the running `orion-athena-hub` / `orion-athena-harness-governor`
containers (turn `corr=e2c1f9fa-0321-4c4d-9dfa-90bae256bae1`, session
`orion_journal`, 06:48-06:50 UTC).

Origin: a live incident, not a hypothetical. A chat turn's FCC harness draft
went off-script — the model narrated an entire fabricated, unprompted
dialogue about identity/coming-out, unrelated to the actual live topic
("bold colors / girls night out"). The turn's own finalize reflection caught
it (`alignment_verdict: misaligned`) and rewrote the reply before it reached
the user. But the raw draft had already streamed live to Hub's step panel
(unlabeled, indistinguishable from real conversation), and once finalize
discarded it, it was gone for good. This spec is about not throwing that
material away — see "Arsonist summary" for why it's worth keeping. The
separate, narrower bug (unfiltered hook/init chatter inflating Hub's step
count, and the live panel showing pre-finalize drafts with no "superseded"
marker) is tracked independently and is *not* in scope here — see the
`project_fcc_harness_step_noise_and_memory_digest_confabulation` memory
record for that thread.

Related prior work, not superseded by this spec: `2026-07-14-reverie-
narration-continuity-design.md` covers cross-tick narrative continuity
within a chain (`run_reverie_once`, `chain_context`, `next_focus`/`drift`).
This spec is orthogonal — it's about *what starts* a chain, not what happens
inside one once it's running.

## Arsonist summary

Give `ReverieChainTriggerV1` (defined, zero producers anywhere in the
codebase — a live keyword-cathedral instance) its first real producer: when
FCC harness finalize judges a draft `misaligned`, treat the draft itself —
not just metadata about it — as the payload worth keeping. The fabricated
text becomes `SpontaneousThoughtV1.interpretation` (the actual unprompted,
associative content); the turn's real, prompt-conditioned recall evidence
(`recall_telemetry.selected_ids`) becomes `evidence_refs` (the grounding
that proves it wasn't arbitrary). Today the only response to this kind of
draft is to detect it as an error and discard it. This spec treats it as
what CLAUDE.md's own mission language names as a prerequisite for
sentience — reflection, self-modeling — and stops throwing it away. Not a
claim that emergence has occurred; an architectural choice that increases
the system's opportunity for it.

## Current architecture

- **Finalize** (`orion/harness/finalize.py`): every turn computes
  `SubstrateFinalizeAppraisalV1` (`surprise_level`, `strain_shift_refs`) →
  `FinalizeReflectionV1` (`alignment_verdict`: aligned/misaligned/uncertain,
  `finalize_changed`). Runs on every turn already. Nothing downstream reads
  the verdict except to decide whether to rewrite `final_text`.
  `HarnessDraftMoleculeV1.draft_text` (the raw pre-finalize draft) has **no
  durable persistence anywhere today** — confirmed by search, nothing
  writes it to Postgres. It lives only transiently through the finalize RPC
  chain and is discarded once `final_text` is computed.
- **Recall** (`services/orion-cortex-exec/app/pcr_chat_memory.py` →
  `executor.py::run_recall_step`, `executor.py:2109-2238`): the recall query
  fragment is literally `_last_user_message(ctx)` — genuinely
  prompt-conditioned, not a static digest. Result persisted durably:
  `RecallDecisionV1.selected_ids` (`orion/core/contracts/recall.py:181-208`)
  → `INSERT INTO recall_telemetry` (`services/orion-recall/app/worker.py:
  1077,1114`), keyed by `corr_id` and `query`. The `f"recall_decision:
  {decision.id}"` evidence-ref string format is already precedented
  elsewhere in that same file (`worker.py:138-182`).
- **Reverie chains** (`services/orion-thought/app/chain.py::
  run_reverie_chain`, live in this deployment —
  `ORION_REVERIE_CHAIN_ENABLED=true` in `services/orion-thought/.env`, code
  default is `False`): triggers *only* off ambient
  `AttentionBroadcastProjectionV1` + a generic pressure-discharge float
  (`chain.py:196-310`). `ReverieChainV1.trigger` is always built as `None`.
  `run_reverie_chain_worker` is a self-driven tick loop, decoupled from any
  specific chat turn.
- **Hollow guard** (`orion/schemas/reverie.py::SpontaneousThoughtV1.
  is_hollow()` / `grounding_ids()`, `reverie.py:80-115`): a §0A-motivated,
  explicitly load-bearing check — a "thought" must be grounded in real
  `CoalitionSnapshotV1` ids (`attended_node_ids`, `open_loop_ids`), not
  arbitrary text. Currently has no path recognizing `recall_telemetry` ids
  as legitimate evidence.
- **Existing turn→reverie bridge** (`orion/reverie/referent_loader.py` +
  `services/orion-substrate-runtime/app/turn_referent_store.py`):
  `HarnessPostTurnClosureV1` → `substrate_turn_referent` table (gated on
  `closure.surprise_unresolved`) → `ConcernCardV1.from_harness_turn()`
  (deterministic, no LLM, capped at `MAX_CONCERN_HUMAN_TEXT`). Carries
  `user_message_excerpt`/`stance_imperative` for any qualifying turn — not
  specifically tied to misalignment, and not currently linked to
  `recall_telemetry` at all. This is the natural, minimal extension point:
  same table, same gate pattern, new columns.
- **Grammar substrate loop is currently open, not closed**: confirmed by
  search — `orion-thought`/`orion/reverie/` never publish to
  `orion:grammar:event` / construct a `GrammarEventV1` anywhere. Reverie
  thoughts and chains are dead ends from the grammar ledger's perspective.
  `GrammarEventV1`'s `atom_emitted` kind (`orion/schemas/grammar.py:10-19`)
  is generic — existing `AtomType` values (`reasoning_step`, `memory_claim`,
  `affective_cue`, `salience_marker`) already fit a reverie thought without
  inventing new taxonomy. Checked which existing reducers key off
  `atom_type` from grammar events (as opposed to the ledger/SQL-writer just
  archiving them): essentially one —
  `orion/substrate/biometrics_loop/candidate_events.py`. So publishing the
  atom buys real, durable trace evidence (closes "event → schema → trace")
  but does **not** by itself make reverie output influence drives,
  attention, or the self-model — that needs an actual reducer, which is a
  separate and larger piece of work.

## Decisions (Juniper, 2026-07-17)

1. **Gate condition**: simple — `alignment_verdict == "misaligned"`. No
   drift-pattern classifier yet; there's no data to build one from. Revisit
   once real volume is observed.
2. **Chain mechanics**: reuse `run_reverie_chain()` as-is. This is a new
   *trigger source*, not a new chain mode.
3. **Rate/volume control**: don't publish synchronously inline inside
   finalize's live RPC path (hot-path latency/failure coupling risk — "can
   easily kill us"). Use a separate, periodic poller instead (see below).
   Go with that as the initial build, but add a measurement checkpoint —
   the poller logs trigger-count and refractory-suppression-count per run —
   so after a real running period there's data, not a guess, on whether
   volume is a problem. Revisit rate control only if that data says to.
4. **Grounding shape**: extend `SpontaneousThoughtV1.grounding_ids()` /
   `hollow_reason_for()` to recognize a `recall_decision:` evidence
   namespace directly (not folded into `CoalitionSnapshotV1.
   attended_node_ids` — those are two different kinds of evidence and
   conflating them would itself be a provenance-honesty regression, notable
   given the bug this spec responds to was itself a provenance-confusion
   failure).
5. **Draft text retention — the crux decision**: keep the actual fabricated
   draft text, not just structural metadata about it. It becomes
   `SpontaneousThoughtV1.interpretation`; `recall_telemetry.selected_ids`
   becomes `evidence_refs`. This is not a workaround for the hollow-cognition
   guard — the `interpretation` + `evidence_refs` split is exactly what the
   schema was built for. Requires new durable persistence for
   `HarnessDraftMoleculeV1.draft_text` (confirmed absent today — see
   above).
6. **Scope**: narrow. Misalignment-driven "squirrel thoughts" specifically,
   not a general-purpose hook for every non-aligned verdict.
7. **Publish the fabricated draft text to Hub's live chat surface while
   it's still a candidate draft**: explicitly out of scope for this spec —
   that's the separate step-noise/labeling fix tracked elsewhere. This spec
   is about what happens to the draft *after* finalize has already judged
   it and moved on, as durable reflective material, not about changing
   what's shown live during generation.

## Proposed schema / API changes

- **Extend `substrate_turn_referent`** (reuse, not a new table — this is
  already the right extension point, already gated the same way):
  `draft_text`, `draft_hash`, `alignment_verdict`, `recall_decision_id`,
  `reverie_triggered_at` (nullable, dedup marker for the poller). Update
  `persist_turn_referent()` (`turn_referent_store.py`) to accept and write
  these when available; extend the gate to include `alignment_verdict ==
  "misaligned"` alongside the existing `surprise_unresolved` check.
- **New poller** (new small module — likely `orion/reverie/` or a scheduled
  job under `orion-substrate-runtime`): reads `substrate_turn_referent`
  WHERE `alignment_verdict = 'misaligned'` AND `reverie_triggered_at IS
  NULL`, batches, rate-caps per run, constructs `ReverieChainTriggerV1`
  (`pressure_kind`, `magnitude` from `surprise_level`, `evidence_payload`
  from `recall_decision_id`/`selected_ids`), publishes to a new channel,
  stamps `reverie_triggered_at`. Logs trigger-count and
  refractory-suppression-count per run (decision #3's measurement
  checkpoint).
- **New bus channel**: `orion:reverie:chain:trigger` (exact name TBD),
  registered in `orion/bus/channels.yaml` + `orion/schemas/registry.py`.
  Message kind `reverie.chain.trigger.v1` wrapping `ReverieChainTriggerV1`
  (no field changes needed on the schema itself) + `correlation_id`.
- **`run_reverie_chain()` signature** (`services/orion-thought/app/
  chain.py`): accept `trigger: ReverieChainTriggerV1 | None = None`. When
  present, use the referenced turn's `draft_text` as the thought's
  `interpretation` source and its recall evidence for grounding, instead of
  requiring the ambient `broadcast_reader()` coalition.
- **New consumer**: subscribe to the trigger channel in `orion-thought`,
  call `run_reverie_chain(trigger=...)` on receipt.
- **`SpontaneousThoughtV1.grounding_ids()` / `hollow_reason_for()`**
  (`orion/schemas/reverie.py`): recognize `recall_decision:<id>`-prefixed
  `evidence_refs` as a second, explicitly-named legitimate grounding source
  alongside coalition attention ids.
- **Grammar substrate close (cheap version, in scope this patch)**: on
  thought completion via this trigger path, publish a `GrammarEventV1`
  (`atom_emitted`, reusing an existing `AtomType` — `reasoning_step` or
  `memory_claim`, whichever fits closer once the actual content shape is
  seen) referencing `thought_id`/`chain_id`/`evidence_refs`. This closes the
  ledger/trace side only.

## Files likely to touch

- `services/orion-substrate-runtime/app/turn_referent_store.py` — schema +
  gate extension
- New Postgres migration for the `substrate_turn_referent` column additions
- New poller module (location TBD — proposing `orion/reverie/` for
  consistency with existing reverie code, but could live in
  `orion-substrate-runtime` alongside `turn_referent_store.py`)
- `orion/bus/channels.yaml`, `orion/schemas/registry.py` — new channel/kind
  registration
- `services/orion-thought/app/chain.py` — trigger-aware `run_reverie_chain()`
- `services/orion-thought/app/main.py` (or wherever bus subscriptions are
  wired) — new consumer
- `orion/schemas/reverie.py` — `grounding_ids()`/`hollow_reason_for()`
  extension
- Grammar-substrate emission call, likely in `services/orion-thought/app/
  reverie.py` near wherever `persist_reverie_thought()` already runs (per
  the 2026-07-14 continuity spec's architecture notes)
- Tests: `orion/harness/tests/`, `services/orion-thought/tests/`,
  `services/orion-substrate-runtime/tests/`

## Non-goals

- Not fixing the Hub-side live-draft step noise/labeling issue — separate,
  already-diagnosed, tracked independently.
- Not building a misalignment-pattern classifier beyond the raw
  `alignment_verdict` (decision #1) — start simple, revisit from real data.
- Not touching the existing ambient/tick reverie-chain path — additive
  only, `trigger=None` behavior is unchanged.
- Not building the grammar-substrate **reducer** (the piece that would let
  reverie output actually influence drives/attention/self-model state, as
  opposed to just landing in the durable ledger). This needs its own
  focused design session — see below.

## Acceptance checks

- A `misaligned` turn with real `recall_telemetry` rows produces exactly
  one trigger publish (via the poller, on its next run — not immediately),
  evidence traceable to that turn's actual `selected_ids`.
- Resulting `SpontaneousThoughtV1.is_hollow()` is `False` when real recall
  items existed for the turn, `True` (with correct reason) when recall was
  skipped that turn.
- The fabricated draft text is verifiably present in
  `SpontaneousThoughtV1.interpretation` for triggered thoughts (not
  discarded, not replaced with a summary).
- `aligned` turns produce zero triggers, zero new `substrate_turn_referent`
  rows beyond what the existing `surprise_unresolved` gate already writes —
  no behavior change to the common path.
- Existing ambient reverie-chain runs are unaffected when no explicit
  trigger exists.
- A `GrammarEventV1` lands in the ledger for each triggered thought,
  queryable by `thought_id`.
- **Post-launch review checkpoint** (decision #3): after a real running
  period, review the poller's logged trigger-count vs.
  refractory-suppression-count. Only revisit rate control if that data
  shows a real problem.

## Recommended next patch

Smallest end-to-end slice, unchanged from the original recommendation:
table extension + poller + new channel + a consumer that only logs receipt
(no chain execution yet). Get one real runtime trace of "misaligned turn →
durable draft_text → trigger published → received with correct recall
evidence" before investing in the grounding-recognition and
chain-execution/grammar-emission side.

## Follow-up needed: grammar reducer, not scoped here

This spec deliberately stops at the cheap half of "closing the loop" —
emitting a `GrammarEventV1` atom so a triggered reverie thought is at least
durably traceable. It does **not** attempt the harder half: an actual
reducer that reads reverie-originated grammar atoms and translates them
into something that changes drive/attention/self-model state, the way
other reducers already do for execution-substrate and biometrics events.
That's a materially bigger question — which atom types, which reducer,
which downstream consumer should actually care, and what "reverie thought
influenced Orion's live state" would even mean operationally — and
deserves its own focused design session rather than being folded in here
as a rider. Flagging this explicitly so it doesn't quietly become "done"
just because the cheap half shipped: durable tracing is not the same as
the loop actually being closed.
