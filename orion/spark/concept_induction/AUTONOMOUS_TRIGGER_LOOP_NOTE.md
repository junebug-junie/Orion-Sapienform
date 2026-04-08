# Autonomous Event-Driven Concept Induction Trigger Loop

## Why this exists

`concept_induction_pass` remains a bounded **reader/reviewer** of concept profiles.
Profile generation still lives in `ConceptInducer.run(...)`, orchestrated through `ConceptWorker.run_for_subject(...)`.

The missing runtime behavior was autonomous triggering from live Orion activity. This patch adds a bounded trigger loop inside `ConceptWorker.handle_envelope(...)` so normal activity can evolve concept profiles without manual review commands.

## Trigger sources (v1)

Trigger source kinds are derived deterministically from envelope kind + intake channel:

- `chat_turn` (`orion:chat:history:log`, `orion:chat:social:stored`)
- `journal_write` (`journal.entry.*` kinds)
- `dream_result` (`dream.result.v1` and `dream.*` kinds)
- `self_review_result` (`self.review*` / `self_review*` kinds)
- `metacog_tick` (`orion:metacognition:tick`, `metacognition.tick*` kinds)
- `cognition_trace` (`orion:cognition:trace`)
- `collapse_event` (`orion:collapse:*`)
- fallback: `generic_activity`

## Trigger contract (bounded)

`ConceptInductionTrigger` carries:

- `source_kind`
- `source_event_id`
- `correlation_id`
- `subjects`
- `trigger_reason`
- `event_timestamp`
- optional `salience`

## Subject set and deterministic selection (v1)

V1 supported subjects remain:

- `orion`
- `juniper`
- `relationship`

Selection is deterministic and inspectable:

- Explicit canonical subject from payload is respected.
- Metacog / cognition trace / self-review sources select `orion`.
- Chat/journal/dream sources select by deterministic markers (`role`, `user`, text mentions of Orion/Juniper).
- Multiple subjects may be selected for one trigger.

## Bounded debounce/cooldown behavior

Added controls:

- Per-event dedupe window (`CONCEPT_TRIGGER_DEDUPE_SEC`)
- Per-subject cooldown (`CONCEPT_TRIGGER_COOLDOWN_SEC`)
- Inflight suppression/coalescing for the same subject
- Bounded retained trigger decisions (`CONCEPT_TRIGGER_RECENT_DECISIONS`)

Decision outcomes are explicit:

- `triggered`
- `coalesced`
- `queued`
- `skipped_due_to_cooldown`
- `skipped_no_window`

## Existing generation path preserved

When a trigger survives controls, the loop invokes the existing `run_for_subject(...)` path.
That path still performs:

1. local profile save via `LocalProfileStore`
2. profile/delta publication
3. RDF materialization via `rdf.write.request`

No alternate generation engine is introduced.

## Observability and operator verification

Structured, grep-friendly logs:

- `concept_induction_trigger_received`
- `concept_induction_trigger_decision`
- `concept_induction_subject_selected`
- `concept_induction_generation_enqueued`
- `concept_induction_generation_skipped`

Operator surface:

- `GET /debug/concept-induction`
- returns intake channels, configured subjects, cooldowns, inflight subjects, per-subject last induce timestamp, and recent trigger decisions.

## Quick verification

1. Produce chat/metacog activity on intake channels.
2. Confirm trigger logs above.
3. Confirm `memory.concepts.profile.v1` and `rdf.write.request` publishes.
4. Confirm graph profile queries begin returning profile rows.
