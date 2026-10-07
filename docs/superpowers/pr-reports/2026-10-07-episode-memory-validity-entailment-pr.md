## Summary

- **End dates on any memory.** The episode distiller now keeps an end date on any memory, not just follow-ups, when Juniper's own words name the period ("Will be here till Wednesday"). Before this, "in Chicago until Wednesday" was stored with no end date.
- **Imported names get asked about.** A memory that names a known person, place, project or service Juniper never said in that conversation is kept, but goes to her as a confirmation question instead of being stored as settled fact. This catches the live confabulation "Juniper corrected me that she lives in Ogden, Utah, not Chicago": "Chicago" came from Orion's own reply.
- **No word lists.** The names come from referent keys Orion already uses.
  - In Juniper's voice, only her own prompts ground a name.
  - In Orion's voice, Orion's quoted reply also counts.
- **Distiller prompt v5.** It asks for `expires_at` plus `until_quote` on any purpose, and names only from this episode.
- **Repair script.** Re-checks stored rows with the same function. It runs as a dry run by default. Live dry run: 38 rows checked, **1 flagged (the Ogden row)**, 0 skipped.

## Outcome moved

Prerequisite for the Situation Graph (spec in #2526, section 7). The situation state will carry episode-memory facts into every chat turn, so the facts must have real end dates, and a confabulated "correction" must not be presented as settled.
- **Live, before:** 3 of 40 rows had end dates, all of them follow-ups, and the Ogden row was stored as settled.
- **After:** time-bounded facts get end dates, and the Ogden-style row becomes a confirmation card.

## Current architecture

`orion-durable-runs` runs `memory.episode_distill` when an episode closes. The LLM proposes memories, `orion/memory/episode/validate.py` checks quotes, voice and stakes, and `store.py` persists.
- The validator dropped `expires_at` unless `purpose=follow_up`.
- Nothing checked whether a statement's names were supported. Only whether its quotes existed.

## Architecture touched

- Validator (shared, `orion/memory/episode`).
- Distill prompt.
- The durable-runs distill graph (passes `candidate_referents` to validation).
- Confirmation card wording (hub).
- The episode report's event list.

No bus, channel, or env changes.

## Files changed

- `orion/memory/episode/validate.py`:
  - `ungrounded_names()` and the escalation (`stakes_reason="ungrounded_name"`);
  - validity windows gated on `until_quote`;
  - a value without an offset is read in Juniper's timezone (`DEFAULT_TZ`, moved here from `distill.py`).
- `orion/schemas/memory_episode.py`: `DistilledMemoryV1.until_quote` (LLM output model, `extra="ignore"`); prompt version constant bumped to v5.
- `orion/cognition/prompts/memory_episode_distill.j2`: v5 instructions and an example.
- `orion/memory/episode/confirmation.py`: `WHY_BY_VALIDATOR_LABEL["ungrounded_name"]` card line.
- `orion/memory/episode/report.py`: lists `ungrounded_name` and `validity_dropped` events.
- `orion/memory/episode/distill.py`: imports `DEFAULT_TZ` from the validator.
- `services/orion-durable-runs/app/episode_distill_graph.py`: `known_referents=candidate_referents`.
- `scripts/repair_episode_memory_ungrounded_names.py`: backfill-protocol repair (dry run, snapshot, progress log).
- Tests:
  - `orion/memory/episode/tests/test_validate_grounded_names_and_validity.py` (new, 17 tests);
  - updates in `test_confirmation.py`, `test_distill_prompt.py`;
  - `services/orion-durable-runs/tests/test_episode_distill_graph.py` (end-to-end that candidate keys reach validation).

## Schema / bus / API changes

- Added: `until_quote` on the LLM-output model only. No DB column: `expires_at` already exists.
- Behavior changed:
  - `expires_at` is now stored on non-follow_up rows when grounded;
  - new `stakes_reason` value `ungrounded_name`;
  - new event ops `ungrounded_name` and `validity_dropped`.
- Compatibility:
  - The confirmation picker selects on stakes/state, not on the reason.
  - Nothing yet reads `expires_at` on non-follow_up rows. This is groundwork for the Situation Graph's expire step (#2526, section 4).

## Env/config changes

- None. No `.env_example` changed, so no sync was needed.

## Tests run

```text
PYTHONPATH=<wt> pytest orion/memory -q                                        2350 passed, 28 skipped
services/orion-durable-runs: pytest tests -q                                  273 passed, 72 skipped
services/orion-memory-consolidation: pytest tests -q                          260 passed, 11 skipped
```

## Evals run

```text
Live measurement of the rule over all 40 stored episode memories (read-only):
  quotes-only grounding            6/40 flagged (too strict: flags normal cross-turn names like Hecate)
  Juniper-said-it-in-episode       2/40
  + only real-name kinds           1/40 = exactly c0dc86c8 (the target)
Repair dry run on live conjourney: checked 38, flagged 1, skipped 0, written 0.
```

The episode distill eval (`services/orion-memory-consolidation/evals/run_episode_distill_eval.py`) was not re-run against the model. It renders no candidate list, so its name check sees only keys the answer emits. Follow-up: re-run it on v5 after deploy.

## Docker/build/smoke checks

```text
Not run in this branch. Deploy from the primary checkout on main after merge (see Restart required).
```

## Review findings fixed

- Finding: quoting Orion's own reply grounded a name in Juniper's voice. That is the live incident with one extra quote.
  - Fix: in Juniper's voice, only her prompts ground; Orion's reply grounds only Orion's voice. The repair script follows the same rule.
  - Evidence: `test_quoting_orions_own_reply_does_not_ground_a_name_in_juniper_voice`, `test_orions_own_memory_may_name_what_orion_said`.
- Finding: a datetime without an offset was read as UTC, six hours early in MDT.
  - Fix: `_parse_dt` localizes to `America/Denver`.
  - Evidence: `test_end_date_without_offset_is_read_in_juniper_timezone`.
- Finding: the repair would flag every name in a row whose episode text didn't load.
  - Fix: such rows are skipped and counted.
  - Evidence: the dry-run report line "skipped, episode text did not load: 0".
- Finding: a too-short `until_quote` was reported with the wrong reason.
  - Fix: new reason `until_quote_too_short`.
  - Evidence: `test_too_short_until_quote_says_so`.
- Finding: `progress.log` lacked ETA and rate.
  - Fix: added.
- Finding: the eval uses a narrower name list than production.
  - Not changed: the eval renders no candidate list, so passing none matches what the model saw. Noted above.

## Restart required

After merge, from the primary checkout on main:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && for s in orion-durable-runs orion-hub orion-memory-consolidation; do docker compose --env-file .env --env-file services/$s/.env -f services/$s/docker-compose.yml up -d --build; done
```

Then the one-off repair, which writes to production and needs Juniper's OK:

```bash
cd /mnt/scripts/Orion-Sapienform && .venv/bin/python scripts/repair_episode_memory_ungrounded_names.py --dsn postgresql://postgres:postgres@localhost:55432/conjourney --apply
```

## Risks / concerns

- **Medium: more confirmation cards.**
  - Concern: a statement that names a known place Juniper didn't say becomes a card. The confirmation loop caps cards at 3 a day and 5 open.
  - Mitigation: the live measurement flags 1 of 40.
- **Low: missed names.**
  - Concern: names are matched on slugs only (for example "ogden-utah" won't match "Ogden, Utah"), so the check can miss names but never invents them.
  - Mitigation: aliases can be added later from the referent alias store.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
