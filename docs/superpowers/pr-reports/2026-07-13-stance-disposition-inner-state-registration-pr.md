# PR: register chat_stance_disposition as REHEARSAL, document rejected composition route

Branch: `docs/stance-disposition-inner-state-registration` → `main`

## Summary

Follow-up to a direct question: "is wiring `stance_disposition` into `SelfStateV1` the most robust path, and given we're training phi encoders off this data, could it create a dangling artifact?" Traced the actual code (not the schema shape) and found the answer was no on both counts — the obvious route was going to be inert for the trained model while still corrupting an existing tracked dimension's provenance. This PR documents that finding and registers the signal honestly instead of silently leaving it untracked.

## Outcome moved

`stance_disposition`/`stance_disposition_reasons`/`stance_boundary_register` (added to `ChatTurnStateV1` by `feat/unified-turn-grammar-trace`, merged earlier today) are now traceable in `orion/self_state/inner_state_registry.py` instead of being an undocumented signal someone would eventually rediscover by grep-archaeology — the exact failure mode that registry exists to stop, per its own docstring ("five independent times in one session..."). No runtime behavior changed.

## Current architecture

`orion/self_state/inner_state_registry.py` tracks every inner-state-shaped signal in the repo with a `CompositionStatus` (`COMPOSED` / `SHADOW` / `DUPLICATE` / `REHEARSAL`), gated by `scripts/check_inner_state_registry.py`. `mood_arc_corpus.v1` (merged same day, `docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md`) is the most recent precedent: a real, computed signal deliberately left `REHEARSAL` (no cognition consumer) pending accumulation of real training data.

## Investigation (this is the actual content of the PR)

Traced where `stance_disposition` goes today, by reading code, not inferring from the schema:
- `orion/substrate/chat_loop/grammar_extract.py::compute_chat_pressure_hints` never reads it — so it never enters the `pressure_hints` dict.
- `services/orion-field-digester/app/ingest/state_deltas.py::delta_to_perturbations`'s `chat_turn` case reads exclusively from `pressure_hints` — so the signal never reaches `substrate_field_state`, `SelfStateV1`, or anything downstream.
- No HTTP read surface exists either (`services/orion-substrate-runtime/app/main.py` only exposes `/projections/execution_trajectory`).

Considered the obvious fix — map a new `boundary_pressure` channel to `SelfStateV1.social_pressure`, reusing the exact pipeline `repair_pressure`/`conversation_load` already use. Rejected after checking `services/orion-spark-introspector/app/inner_state.py`:
```python
SEEDV4_THEATER_FELT: frozenset[str] = frozenset({"coherence", "continuity_pressure", "social_pressure"})
```
`social_pressure` is already excluded from φ's live `seed-v4` trainable feature set — a dated, deliberate design decision (`docs/superpowers/specs/2026-07-09-phi-seedv4-feature-set-design.md`), not an oversight. Composing into it would (a) never reach the deployed encoder, while (b) still mutating the `infra`-provenance recording of a dimension that decision explicitly evaluated and excluded — with no record of why the numbers moved — on the same day `mood_arc_corpus.v1` started real corpus collection for a future training run.

## Files changed

- `docs/superpowers/specs/2026-07-13-stance-disposition-inner-state-path.md` (new) — full trace, the rejected route with evidence, three candidate paths forward (own `SelfStateV1` dimension / feed `mood_arc_corpus.v1` directly / defer to a seed-v5 redesign), none chosen.
- `orion/self_state/inner_state_registry.py` — new `chat_stance_disposition` entry, `schema=None` (field group on `ChatTurnStateV1`, not a standalone schema — same distinction as the existing `phi_heuristic.valence` entry), `composition_status=REHEARSAL`, `cognition_consumers=()`, referencing the doc above.
- `orion/self_state/README.md` — REHEARSAL section updated from "two current entries" to three, with the new one summarized and cross-referenced.
- `services/orion-hub/README.md` — the existing unified-turn chat grammar trace section extended with where the stance fields terminate today and why, cross-referencing the same doc. (Matches a review finding from the `mood-arc-corpus-collector` PR: intro/summary sections need updating when a registry entry lands, not just the registry file itself — applied the same discipline here proactively rather than waiting to be caught.)

## Schema / bus / API changes

None. No new schema, no new channel, no new consumer. Pure documentation + registry bookkeeping.

## Env/config changes

None.

## Tests run

```
/tmp/orion-test-venv/bin/python scripts/check_inner_state_registry.py
→ inner_state_registry gate OK (11 entries checked)

/tmp/orion-test-venv/bin/python -m pytest tests/test_inner_state_registry_gate.py -q
→ 8 passed
```

## Evals run

Not applicable — no runtime code changed.

## Docker/build/smoke checks

Not applicable — no service touched.

## Review findings fixed

None from an external pass; self-applied one finding from a different, recent PR in this repo (`mood-arc-corpus-collector`, review angle B: "intro paragraph listing tracked signal types wasn't updated to mention the new entry") proactively, rather than shipping the registry entry without the corresponding README cross-references and waiting for a reviewer to catch the same gap again.

## Restart required

None. Docs and a registry constant only — no running service reads `inner_state_registry.py` at runtime (it's a dev-time/gate-time module).

## Risks / concerns

None identified. This PR is deliberately inert by design — that was the entire point of choosing `REHEARSAL` + a doc over silently wiring the signal into `SelfStateV1`.

## PR link

Push and open via: `git push -u origin docs/stance-disposition-inner-state-registration` (already pushed), then open the compare URL GitHub printed (no `gh` auth in this environment).
