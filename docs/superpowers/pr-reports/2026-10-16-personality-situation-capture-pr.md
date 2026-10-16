## Summary

- Every new piece of personality evidence now records the situation it happened in: `meta.situation = {arousal, day_phase, company}`, read at the row's own time.
- This is the "when" half of the CAPS if-then signature ("when X, Orion tends to Y"). It is recorded starting today, so that situation subnodes can later be built from real data. **Nothing reads it back yet.**
- Inputs that are already stored:
  - `arousal`: the latest Temporal Self `arousal_transition` at or before the row's time. It reads `unknown` when there is none, never a guessed `idle`.
  - `day_phase`: the phase of Orion's local day.
  - `company`: whether a Temporal Self `company` arc (someone at the room camera) was open.
- Attention is left out on purpose: it nearly always crowns a prediction-error node, so it would not tell situations apart.

## Outcome moved

The personality graph (#2738, live since 2026-10-16 22:37 UTC) knows *what* Orion keeps returning to, but not *when*. From the moment this deploys, each evidence row carries its situation, so the data for if-then subnodes builds up while the graph matures.

## Current architecture

The durable-runs personality reducer reads Orion's own outputs by cursor, embeds them, folds them with `orion/personality/graph.py`, and stores them in `personality_evidence.meta` (jsonb). Temporal Self already stores arousal transitions (`temporal_self_event`) and company arcs (`temporal_self_arc`).

## Architecture touched

- orion-durable-runs only: `PersonalityStore.load_situation_inputs` (read-only SQL) and `PersonalityDriver._with_situation`. The stamping runs after accept and before fold, and fails open.
- New pure module `orion/personality/situation.py`.

## Files changed

- `orion/personality/situation.py`, `orion/personality/tests/test_situation.py`: the pure situation lookup
- `services/orion-durable-runs/app/personality_store.py`: `load_situation_inputs`
- `services/orion-durable-runs/app/personality_driver.py`: stamps new evidence
- `services/orion-durable-runs/tests/test_personality_loop.py`: tests for stamping at the row's own time and for a failed lookup still folding
- `services/orion-durable-runs/README.md`: new section
- `docs/superpowers/pr-reports/2026-10-16-personality-situation-capture-pr.md`: this report

## Schema / bus / API changes

- Added: `meta.situation` on new `personality_evidence` rows (jsonb; no migration)
- Removed: none
- Renamed: none
- Behavior changed: none for weights or matching (`meta` is never read by the fold)
- Compatibility notes: `choice_style_from_rows` reads only `meta.choice_id` and `meta.chosen_best_rank`. The Hub panel does not read `meta`.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Metric quality gate: situation fields

1. **Provenance.**
   - Arousal: `regulation_store.py:166-170` writes `label = new.arousal_level` on `arousal_transition` rows.
   - Company: Temporal Self `company` arcs from camera presence transitions (#2665).
   - Day phase: `orion.temporal_self.day.day_phase_for`.
2. **Independence.** Arousal `engaged` and `company` can co-occur when Juniper is in the room and chatting, but each can happen without the other. Day phase is a clock. None of them comes from the evidence text.
3. **Theory anchor.** CAPS (Mischel & Shoda) holds that personality is a set of situation-behavior if-then signatures. Recording the "if" is the precondition for estimating them.
4. **Live sanity.** A read-only run of the real query over the last 6 hours found 7 transitions and 11 company arcs. Arousal moved between strained and idle, company turned true in the last hour, and the day phase followed MDT. The data is not degenerate.
5. **Existing mechanism.** No personality or evidence row recorded situation before this. Temporal Self stores the raw signals, and this reads them.
6. **Reversibility.** It is an annotation inside `meta` jsonb. It changes no weights and creates no schema, so reverting the commit is enough.

## Tests run

```text
orion/personality/tests -> 110 passed (also 110 passed in a venv with only pytest + pydantic, the CI install)
services/orion-durable-runs: pytest tests (throwaway local Postgres 16) -> 527 passed, 1 skipped
services/orion-durable-runs: tests/test_personality_loop.py -> 26 passed (after the review fix)
```

## Evals run

```text
Read-only live check of load_situation_inputs + situation_at over the last 6 h (see metric gate 4).
```

## Docker/build/smoke checks

```text
Not deployed. Live stamping UNVERIFIED until durable-runs is rebuilt.
```

## Review findings fixed

- Finding: `LIMIT 500` on arousal transitions would silently give every row after the 500th transition that row's level during a backlog more than 3 weeks wide.
  - Fix: no row cap on either query. Volume is tiny: 56 transitions in all, about 20 a day.
  - Evidence: `personality_store.py` docstring; tests pass.
- Finding: test import placement and blank lines (style).
  - Fix: tidied.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- Severity: low. Concern: `company` means "someone at the room camera", with no identity. Company arcs carry no confirmed label today. Mitigation: documented in the module. Identity can be added when arcs carry it.
- Severity: low. Concern: the situation is recorded but nothing reads it back. Mitigation: intended. Situation subnodes are the next phase, once data has built up.

## PR link

(set after creation)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
