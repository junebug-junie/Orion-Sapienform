# Stance disposition: inner-state registration and path forward

Status: registered as `REHEARSAL`, not composed. No code change to `SelfStateV1`, `pressure_hints`, `self_state_policy.v1.yaml`, or `orion-field-digester` in this patch. This doc exists so the decision *not* to wire it in yet is traceable, not silently absent — the exact gap `orion/self_state/inner_state_registry.py` was built to catch.

## What this signal is

`orion/hub/turn_orchestrator.py::execute_unified_turn` resolves a Thought stance decision every turn — `proceed` / `defer` / `refuse`, with `disposition_reasons` and a `boundary_register` flag (`orion/schemas/thought.py::ThoughtEventV1`). Since `feat/unified-turn-grammar-trace` (merged 2026-07-13), this is captured as a `stance_disposition` atom in the `hub.chat:` `GrammarEventV1` trace, parsed by `orion/substrate/chat_loop/grammar_extract.py::extract_chat_turn_state` into three new fields on `ChatTurnStateV1` (`orion/schemas/chat_projection.py`): `stance_disposition`, `stance_disposition_reasons`, `stance_boundary_register`.

It is genuinely new: no prior signal in this repo recorded Orion's own engage/decline decisions as durable, queryable data.

## Where it actually reaches today (traced, not assumed)

```
Thought.disposition → ChatTurnStateV1.stance_disposition
  → active_chat_session projection (Postgres JSONB, substrate-runtime)
  → nothing further
```

Confirmed by reading the code, not inferring from schema shape:

- `orion/substrate/chat_loop/grammar_extract.py::compute_chat_pressure_hints` computes `conversation_load`/`repair_pressure`/`topic_coherence` only. It has no reference to `stance_disposition` — the field is populated on `ChatTurnStateV1` but never enters `pressure_hints`.
- `services/orion-field-digester/app/ingest/state_deltas.py::delta_to_perturbations`'s `chat_turn` case (the only place a chat-lane `StateDeltaV1` becomes a `substrate_field_state` perturbation) reads exclusively from `after["pressure_hints"]`. Since `stance_disposition` never reaches that dict, it never reaches `substrate_field_state`, `SelfStateV1`, or anything downstream of Layer 6.
- No HTTP read surface exists either: `services/orion-substrate-runtime/app/main.py` exposes `GET /projections/execution_trajectory` only — not `chat_session`, not `route_arbitration`.

So today it is real (computed, correctly, per turn) and completely inert beyond the raw ledger row. Not a bug — the field-digester/self-state layers were never told this fact exists yet. This doc is that telling.

## The considered path, and why it was rejected

The obvious move: add a `boundary_pressure` channel to `compute_chat_pressure_hints` (e.g. `1.0` on `defer`/`refuse`, `0.0` on `proceed`) and map it to the `social_pressure` dimension in `config/self_state/self_state_policy.v1.yaml` — reusing the exact pipeline `repair_pressure`/`conversation_load` already use (both already mapped to `social_pressure`, confirmed live in that file).

Rejected after tracing where `social_pressure` actually goes:

- `services/orion-spark-introspector/app/inner_state.py::SEEDV4_THEATER_FELT` already contains `social_pressure` — under the live `seed-v4` feature version, this dimension is deliberately excluded from φ's trainable feature set and recorded only in `InnerStateFeaturesV1.infra` ("provenance only... NEVER read by φ"). That exclusion was itself a dated design decision (`docs/superpowers/specs/2026-07-09-phi-seedv4-feature-set-design.md`), not an oversight.
- So the wiring would not reach the deployed encoder at all — while still mutating the *provenance recording* of a dimension a prior design pass explicitly evaluated and excluded, with no record of why the numbers moved.
- `mood_arc_corpus.v1` (`docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md`, item 1) started real corpus collection on 2026-07-13 — the same day. Introducing composition drift into an existing tracked `SelfStateV1` dimension at the exact moment hours-of-data collection begins is the specific failure mode a training corpus cannot recover from after the fact.
- `orion/self_state/inner_state_registry.py`'s own docstring: *"five independent times in one session, a real, computed inner-state signal turned out to either silently duplicate another one, or never reach a cognition consumer at all."* Composing `stance_disposition` into `social_pressure` without a registry entry would have been occurrence six, by the same person who was told about the first five.

## Path forward (none chosen — this is the menu, not a decision)

1. **Give it its own `SelfStateV1` dimension**, rather than folding into `social_pressure`. Requires a deliberate weight/formula design pass, the same kind of work `2026-07-09-phi-seedv4-feature-set-design.md` did for the current feature set. Not started.
2. **Feed the mood-arc corpus directly** instead of `SelfStateV1`. `mood_arc_corpus.v1` already exists specifically to accumulate real hours of data ahead of a training decision, off the hot path, with no cognition consumer required yet (`REHEARSAL` is its correct, accepted status per its own registry entry). Adding a `stance_disposition`-derived field to `MoodArcCorpusRowV1` would follow the exact precedent that schema was built for, without touching `SelfStateV1`'s live composition at all. Lowest blast radius of the three options.
3. **Defer to a future seed-v5-style feature-set redesign.** By the time that happens, real accumulated data (via option 2, if taken) would exist to evaluate whether stance/boundary signal actually correlates with anything the encoder's latent space finds useful — the same evidence-based process that resolved `phi_heuristic.valence`'s SHADOW→COMPOSED status on 2026-07-13, rather than a guess made before any data exists.

No recommendation is made between these here — that is a design call for whoever picks this back up, informed by whether `mood_arc_corpus.v1` collection is actually running by then and what it shows.

## Non-goals of this patch

- No change to `SelfStateV1`, `self_state_policy.v1.yaml`, `pressure_hints`, or `delta_to_perturbations`.
- No new `/projections/*` HTTP endpoint (a separate, smaller, still-open idea — dev-only visibility, does not by itself solve the composition question above).
- No change to `MoodArcCorpusRowV1` or the mood-arc roadmap's item 2 (training).

## Acceptance check

```bash
python scripts/check_inner_state_registry.py
```
Registry entry added for `chat_stance_disposition` with `composition_status=REHEARSAL`, `schema=None` (it is a field group on `ChatTurnStateV1`, not a standalone schema — matches the precedent set by `phi_heuristic.valence`'s formula-not-schema entry), `cognition_consumers=()`, referencing this document.
