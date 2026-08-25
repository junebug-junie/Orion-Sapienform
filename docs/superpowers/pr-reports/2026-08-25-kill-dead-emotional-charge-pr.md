# Kill dead emotional_charge regex in attention scoring

## Summary

- Found while investigating a suspected competing architecture for reading Juniper's affect: `orion/substrate/attention/scoring.py` has had its own independent, regex-based Juniper-emotion detector (`_EMOTION_RE`, 11 words) since 2026-05-16, stamping `emotional_charge` onto every `OpenLoopV1` on every chat turn.
- Confirmed dead since 2026-07-31 (`b1d567787`, `feat(attention)!: kill ha[nd-tuned salience]`): that's the exact commit that deleted the only formula that ever read it. The Borda-v2 salience formula never references it. No scoring, persistence, or UI/debug consumer reads it either -- pure empty-shell cognition since that date.
- Removed `_EMOTION_RE` and `emotional_charge` (field, computation, and the docstring's stale claim that a sibling field was still live -- it wasn't, see review findings below).
- Code review caught a real, material gap in the first cut: `OpenLoopV1` keeps `extra="forbid"` while its shape changed, and it round-trips through Postgres JSONB history. Fixed with a narrow backward-compat validator.

## Outcome moved

One fewer disconnected, non-functional "reads Juniper's affect" mechanism in the codebase -- closes the gap flagged in the "what's next on the affect arc" conversation, alongside the two real signals (`JuniperAffectiveStateV1`/orion-cocreation-signals, `JuniperMultimodalAffectV1`/AffectGPT, PR #1865/#1871).

## Current architecture

Before this patch, three unconnected systems claimed to read "how is Juniper doing":

| Signal | Added | Method | Status |
|---|---|---|---|
| `_EMOTION_RE`/`emotional_charge` | 2026-05-16 | 11-word regex on raw chat text | Dead since 2026-07-31 |
| `JuniperAffectiveStateV1` (swear_frequency) | 2026-07-30 | Real word-frequency scoring | Published + persisted, no cognition consumer (Juniper's explicit "leave it running" call) |
| `JuniperMultimodalAffectV1` (AffectGPT) | 2026-08-22 | Real GPU vision+audio inference | Published + persisted + wired into chat-turn context |

## Architecture touched

- `orion/substrate/attention/scoring.py`: removed the dead regex/field computation.
- `orion/schemas/attention_frame.py`: removed the `OpenLoopV1.emotional_charge` field; added a `model_validator(mode="before")` backward-compat carve-out.

## Files changed

- `orion/substrate/attention/scoring.py`: removed `_EMOTION_RE`, the `emotional = ...` local, and the `emotional_charge=` kwarg.
- `orion/schemas/attention_frame.py`: removed the field; added `_REMOVED_LEGACY_FIELDS`/`_drop_removed_legacy_fields()`; corrected the docstring's inaccurate claim about `novelty`.
- `orion/substrate/tests/test_salience_schema.py`: 3 new tests for the backward-compat carve-out.

## Schema / bus / API changes

- Removed: `OpenLoopV1.emotional_charge` (never had a bus channel of its own -- it's a field inside `AttentionBroadcastProjectionV1`/`AttentionFrameV1`, both already-registered schemas whose shape changed, not a new registration).
- Compatibility notes: `OpenLoopV1` still rejects genuinely unknown fields (`extra="forbid"` unchanged) -- only the specific, named, historically-real `emotional_charge` key is stripped before validation, via an explicit `_REMOVED_LEGACY_FIELDS` set meant to grow (never shrink) the next time a field is removed from this model.

## Env/config changes

None.

## Tests run

```text
PYTHONPATH=. .venv/bin/python -m pytest -q \
  orion/substrate/tests/test_salience_schema.py \
  orion/substrate/tests/test_salience_discrimination_eval.py \
  orion/substrate/tests/test_attention_verdict_exclusion.py \
  orion/substrate/tests/test_scoring_salience_wiring.py \
  orion/substrate/tests/test_attention_broadcast_dwell.py \
  services/orion-cortex-exec/tests/test_attention_frame_integration.py \
  services/orion-cortex-exec/tests/test_attention_frame.py \
  services/orion-cortex-exec/tests/test_chat_attention_salience_trace.py \
  orion/hub/tests/test_association_read_fail_closed.py \
  tests/test_top_down.py \
  scripts/analysis/tests/test_measure_ast_hot_reducer.py
104 passed

# services/orion-thought (separate PYTHONPATH, cwd-relative imports)
125 passed

# services/orion-hub (separate PYTHONPATH, run from repo root)
20 passed (test_attention_loops_api.py, test_attention_loops_ui_smoke.py, test_attention_loops_reader.py)
```

229 total, zero failures.

## Evals run

None -- no eval harness covers this scoring path directly; no behavior change intended for the live formula (emotional_charge was already unread).

## Docker/build/smoke checks

Not run -- pure code/schema removal, no runtime config, port, or dependency change. Flagging for Juniper: this DOES change the shape `AttentionBroadcastProjectionV1` rows get written in going forward, so a restart is needed to pick it up (below), but no other Docker action is required.

## Review findings fixed

- Finding: `OpenLoopV1` keeps `extra="forbid"` while the field is removed, so `model_validate()` on any pre-deploy row from `substrate_attention_broadcast_log` (168h append-only history) or `substrate_attention_broadcast_projection` (live singleton) raises. `scripts/analysis/measure_ast_hot_reducer.py`'s replay tool would silently count every one as a skip, not a schema break -- real, under-the-radar historical data loss.
  - Fix: `model_validator(mode="before")` on `OpenLoopV1` strips a small, explicit, append-only `_REMOVED_LEGACY_FIELDS` set before strict validation runs.
  - Evidence: new test `test_a_stored_row_still_carrying_the_removed_field_still_parses` constructs the exact pre-removal row shape and asserts it still parses; `test_a_genuinely_unknown_field_still_raises` confirms the carve-out is narrow, not a blanket `extra="ignore"`.
- Finding: during a rolling deploy where the producer (old code) writes an `emotional_charge`-bearing row after the consumer (new code) is live, `orion/hub/association.py` / `services/orion-thought/app/broadcast_reader.py` would silently degrade to stale/no-coalition for the deploy window.
  - Fix: the same backward-compat validator above -- old-shaped rows now parse fine regardless of which side of the schema change is deployed first.
  - Evidence: same test coverage; no sequencing/ordering requirement needed for this deploy.
- Finding: the commit's own disclosed dead-field audit misclassified `OpenLoopV1.novelty` as "confirmed still live" when precise re-grep found zero readers of `loop.novelty` anywhere.
  - Fix: corrected the docstring to group `novelty` with `continuity_relevance`/`relational_relevance` as similarly unread, not touched in this patch; kept the accurate claims (`predictive_value`/`concept_value`/`autonomy_value` are genuinely read by `policy.py`/`top_down.py`).
  - Evidence: `grep -n "loop\.novelty"` across the live repo returns exactly one hit, a comment describing the already-deleted pre-v2 formula.

## Restart required

```bash
# cortex-exec / hub / substrate-runtime / thought all import orion.schemas.attention_frame
# and orion.substrate.attention.scoring -- redeploy whichever of these are running
# from a checked-out copy of this code rather than a fresh pull.
docker compose \
  --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml \
  up -d --build
```

If Hub/orion-thought/orion-substrate-runtime run from baked images rather than a live mount, redeploy those too on the same schedule -- there is no hard ordering requirement (the backward-compat validator absorbs either direction of skew), but keeping them together avoids running mixed code longer than necessary.

## Risks / concerns

- Severity: low
- Concern: `continuity_relevance`/`relational_relevance` (and now-corrected `novelty`) appear equally dead by the same grep methodology used here, but were not removed -- a future reader could assume this patch was a complete legacy-field cleanup when it wasn't.
- Mitigation: explicitly disclosed as a follow-up, not removed here, in both the field comment and this report -- was outside the scope Juniper approved for this specific patch (the emotional_charge/Juniper-affect duplication question, not a general legacy-field audit).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1875
