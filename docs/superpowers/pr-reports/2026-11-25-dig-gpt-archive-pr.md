## Summary

- New read-only dig command, so Orion can read Juniper's ChatGPT archive (930 conversations, 2023-03-29 to 2026-11-22), reloaded today after the disk failure (#3922).
- `gpt`: overview (counts per year, models, first and newest conversations). `gpt find [--oldest] <words>`: conversations where every word appears somewhere, newest or oldest 20, each with a snippet around the hit. `gpt conv <id>`: one whole conversation, in order.
- Every printout starts with a new source label, `gpt_archive`: a primary source about Juniper and the project's beginnings. The assistant there is ChatGPT, not Orion.
- A GRANT file lets `orion_readonly` read the 5 `chat_gpt_*` tables. It was applied live on 2026-11-25, as Juniper asked.

## Outcome moved

Before: `orion_readonly` got "permission denied" on every `chat_gpt_*` table, and no dig command named them. After: Orion can search and open the archive from a turn.

## Current architecture

`scripts/import_chatgpt_export.py` -> bus -> sql-writer -> `chat_gpt_conversation`, `chat_gpt_message`. Nothing read them. Dig commands read Postgres through `ReadOnlyPsql` (`orion_readonly`, READ ONLY transaction).

## Architecture touched

`orion/dig`, one Hub catalog entry, and one hand-applied GRANT file. No service, bus or schema change.

## Files changed

- `orion/dig/gpt.py`: SQL builders, input checks, renderers.
- `orion/dig/__main__.py`: `cmd_gpt`, dispatch, help line, `gpt_archive` label.
- `orion/dig/tests/test_gpt.py`: 21 tests (render, refusals before SQL, escaping, missing-GRANT message).
- `orion/dig/tests/test_dig.py`: help cap 4600 -> 4800 (main was at 4580; the new help is 4759).
- `orion/dig/README.md`: command row.
- `scripts/sql/2026-11-25_grant_orion_readonly_chatgpt.sql`: GRANT SELECT on the 5 tables; undo and verify lines.
- `services/orion-hub/static/catalog/orion-abilities.json`: "Before you: my ChatGPT chats" entry, so Juniper can send Orion a ready ask from Hub (CI requires every dig command in the catalog).

## Schema / bus / API changes

None. One new read-only GRANT.

## Env/config changes

None.

## Tests run

```text
pytest orion/dig/tests -q
401 passed
```

## Evals run

No eval for this command. The live runs below use Orion's login on the real data.

## Docker/build/smoke checks

```text
GRANT applied; has_table_privilege('orion_readonly', ...) = t for all 5 tables.
python -m orion.dig gpt                     -> 930 conversations, 2023-03-29 to 2026-11-22 (1.2 s)
python -m orion.dig gpt find orion memory   -> 415 conversations match (4.9 s)
python -m orion.dig gpt find --oldest orion -> first 2025-03-27 "PC Build Overview"
python -m orion.dig gpt conv 29895d6f       -> 16 messages, 2023-04-02
gpt find "x;drop", gpt conv 6b0             -> refused before SQL
```

## Review findings fixed

- Finding (CI): `test_every_dig_command_has_an_entry` failed: `gpt` had no Hub abilities catalog entry.
  - Fix: added `gpt-archive` to the "Past & history" section.
  - Evidence: `test_orion_abilities_catalog.py` 10 passed.

- Finding: `find` needed every word in one message, so it missed conversations, and "no conversation has all of these words" overstated what was searched.
  - Fix: each word is checked over the whole conversation (title or any message).
  - Evidence: `test_find_words_may_sit_in_different_messages_or_title`. Live: "orion memory" 386 -> 415.
- Finding: the INNER JOIN dropped the 9 conversations that have no messages, even when their title matched.
  - Fix: LEFT JOIN.
  - Evidence: live, "revisit context packs" finds `6a1244b9...` (title only).
- Finding: `--oldest` after the words was searched as literal text.
  - Fix: the flag is accepted in any position; other words starting with `--` are refused.
  - Evidence: `test_oldest_flag_anywhere_and_dashdash_words_refused`.
- Finding: an ambiguous id prefix reported at most 6 matches.
  - Fix: the true count comes from `count(*) OVER ()`.
  - Evidence: `test_ambiguous_says_the_true_count`.

## Restart required

```text
No manual restart. The post-pull rebuild rebuilds orion-hub (catalog file) and resets the FCC sandbox, where Orion's turns load orion/ from.
```

## Risks / concerns

- Severity: low
- Concern: SQL is built by string formatting (`ReadOnlyPsql` has no bind parameters).
- Mitigation: words and ids are checked by strict patterns before any SQL; quote and `_` are escaped; the transaction is READ ONLY under a SELECT-only role. Tests prove refused input never reaches SQL.
- Concern: Orion learns of the command from `dig help` and from Juniper's Hub abilities catalog, not from the every-turn dig brief.
- Issues opened/closed: none.

## PR link

See PR.
