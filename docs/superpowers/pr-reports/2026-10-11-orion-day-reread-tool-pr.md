## Summary

- New Orion tool `mcp__orion-introspect__orion_day`: Orion rereads their own Orion's Day letters. This is patch 2 of `docs/superpowers/specs/2026-10-11-orion-day-letter-reread-design.md`.
- Ways to call it:
  - **No arguments:** an outline of the latest letter (¶ paragraphs, carry items, section counts).
  - **`part=note|carry_forward` + `index`:** the exact words of that part, with evidence:
    - note paragraphs carry a claim check: each concrete token, and the records it was found in;
    - carry items carry their citations, resolved to record excerpts.
  - **`part=section`:** that day's records for the section.
  - **`query`:** search letter parts by meaning. Same `semantic_index` mechanism as `curiosity`, in its own collection, `orion_day_letter_parts`.
- New Hub responder (`orion_day_introspect_listener.py`) on `orion:introspect:orion_day:request`:
  - read-only (READ ONLY transactions);
  - same error-vs-empty contract as the other introspect tools.
- Guide line, telling Orion: when Juniper points at a letter, call `orion_day` first, quote what was written, and separate what the records support from what they do not.

## Outcome moved

Before, Orion had no way to read a past letter, so talk about one was reconstruction. After, the real 2026-10-09 letter (read live, read-only) gives:
- 24 numbered paragraphs, 10 carry items, and 21 of 21 citations resolved, matching the email's numbering;
- ¶2's `06:35:27Z` and `260.6` traced to `curiosity:e7d03d2ecdbb`;
- an unknown date, and ¶30, both coming back as not-found errors.

## Current architecture

- `orion-introspect` served `reading_results`, `dreams` and `curiosity`, each answered by a Hub listener.
- `orion_day_letter` was readable only by the email path.
- Patch 1 (#2606) added the shared splitters and claim check.

## Architecture touched

- Contract: `OrionDayArguments`, and `orion_day` added to the operation literals.
- Registry, bus channel, introspect tools and brief.
- Hub: a new responder plus startup and shutdown wiring.
- `orion/orion_day`: letter_parts helpers and read-only store queries.
- CI workflow paths.

## Files changed

- `orion/schemas/introspect.py`, `orion/schemas/registry.py`, `orion/introspect/transport.py`, `orion/bus/channels.yaml`: the contract and channel.
- `orion/introspect/tools.py`, `orion/introspect/brief.py`: the tool spec and description, plus the exact-name guide line.
- `orion/orion_day/letter_parts.py`: `parse_ref`, `SECTION_PREFIXES`, `section_records`, `record_text`, `record_excerpt`.
- `orion/orion_day/store.py`: `fetch_latest_letter`, `fetch_letter_texts`, `LETTER_CREATED_SINCE_SQL` (read-only).
- `services/orion-hub/scripts/orion_day_introspect.py` (pure item builders) and `services/orion-hub/scripts/orion_day_introspect_listener.py` (bus responder plus search index loop).
- `services/orion-hub/scripts/main.py`: start/stop.
- `services/orion-hub/README.md`, `services/orion-harness-governor/README.md`: docs.
- Tests: `orion/introspect/tests/test_orion_day_tool.py`, `services/orion-hub/tests/test_orion_day_introspect_listener.py`, and additions to `test_letter_parts.py` and `fixtures.py`.
- Eval: `services/orion-hub/evals/run_orion_day_reread_eval.py` (+ pytest wrapper).
- `.github/workflows/orion-day-letter-hub-tests.yml`: the new tests, the eval and the paths.

## Schema / bus / API changes

- Added:
  - `OrionDayArguments` and `OrionDaySection`;
  - channel `orion:introspect:orion_day:request`;
  - MCP tool `orion_day`;
  - item kinds `orion_day_letter_part` (unsettled) and `orion_day_record`.
- Removed: none
- Renamed: none
- Behavior changed: none for the existing tools.
- Compatibility notes: additive. The harness MCP server lists `orion_day` once the governor runs this code.

## Env/config changes

- Added keys: none. Search reuses `HUB_CURIOSITY_SEARCH_*`.
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
orion/introspect/tests + orion/orion_day + orion/fcc/tests + test_harness_prefix:
  390 passed, 3 skipped, 1 failed (test_no_undiscovered_root_callers_of_claude_permission_argv,
  pre-existing on main)
services/orion-hub: orion_day listener, curiosity_introspect (+listener), orion_day_letter,
  reread eval, email eval: 106 passed, 3 skipped
```

## Evals run

```text
run_orion_day_reread_eval.py (10 planted references: 6 supported, 4 not):
  quoted exactly 10/10, flagged correctly 10/10, supported claims name the planted record 6/6,
  unknown letter is a not-found error: yes. PASS.
  Mutation check: forcing claim_check to report everything found fails it with 7 failures.
```

## Docker/build/smoke checks

```text
Live read-only responder run against the real 2026-10-09 letter: numbers as above.
Static gates: check_bus_reply_channels, check_settings_defaults orion-hub --example-drift,
check_scripts_dir_no_stdlib_shadow, check_async_routes_not_blocking, check_chat_route_poachers,
check_service_hostname_refs all OK.
check_single_consumer_channels.py needs a live bus and was not run. Its static test already fails on
main for 6 unrelated channels; the new channel is annotated.
```

## Review findings fixed

- Finding: an unresolved-citation list had no cap, so one carry item with 400 refs serialized to 24,670 characters.
  - Fix: capped at 10, with the full count beside it.
  - Evidence: a 400-ref test, which fails when the cap is removed.
- Finding: a date-narrowed search for a missing letter reported "search unavailable".
  - Fix: the letter is resolved first, so the answer is "not found".
  - Evidence: test.
- Finding: a date-narrowed empty search could say "index behind" because of an unrelated newer letter.
  - Fix: the check is scoped to that one letter.
  - Evidence: test.
- Finding: the "no letter yet" error text did not match the README.
  - Fix: it is now "unknown letter … (not found)".

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build && docker compose --env-file .env --env-file services/orion-harness-governor/.env -f services/orion-harness-governor/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: medium
  - Concern: acceptance check 5 (in a live chat, Orion calls the tool before answering about "Oct 9 ¶N") is UNVERIFIED until deploy.
  - Mitigation: ask Orion about Oct 9 ¶18 after deploy and check the trace for the `orion_day` call.
- Severity: low
  - Concern: the 0.65 search similarity floor was calibrated on curiosity write-ups and is UNVERIFIED for letter paragraphs.
  - Mitigation: searches are re-read from the table, so a wrong floor costs recall, not truth.
- Severity: low
  - Concern: the search index backfills about 10 docs per 5-minute pass (about 34 docs per letter). Until the index is confirmed complete, an empty search reports "unknown", not "none".
  - Mitigation: none needed; it says so honestly.
- Severity: low
  - Concern: acceptance check 1 (the real letter's numbers) is confirmed live but not pinned in a test. No copy of the real letter is committed.
  - Mitigation: the synthetic fixture tests cover the same mechanics.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2613

🤖 Generated with [Claude Code](https://claude.com/claude-code)
