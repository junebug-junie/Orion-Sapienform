## Summary

- Orion's background outreach choice point now shows two more kinds of topic, both about the world instead of Orion's machinery: **people** Juniper named for the cameras, seen in a new way, and **chat threads** that went quiet after Orion asked Juniper something.
- People: back after 4+ days away, first match in the 60-day window, or at a time of day none of their 6+ earlier visits came near. Only people Juniper named (typed-name links), never unnamed clusters; not Juniper or Orion; the office camera is left out (office visits keep their own door). A camera match is a cluster match, not proof of who it was, so a glimpse (under 3 frames and 5 s) or a moment when another named person matched on the same camera within 2 minutes is dropped, and the text says "a camera match".
- Chat threads: a turn of Juniper's in the last 7 days where Orion's reply ended on a question and her next message came 2+ hours later (or not yet). Plumbing-shaped threads (code span, URL, path, `#123`, `snake_case`, `qwen38`-style words) are left out.
- Every new topic cites the row it stands on (`vision_individual_sighting:<id>`, `chat_history_log:<id>`): stored on the decision row (`choice.shown[].evidence_ref`) and in the published choice's situation (`evidence`).
- Fix: the 7-day subject ledger query left out `no_pick` / `invalid_pick` rows, so a reply with no readable pick did not keep its topics out as `fold_said_subjects` documents. It now reads every reason the fold reads.
- Eval readout: shown / picked per topic kind, and `--candidates` counts the person and chat-thread topics that would exist right now.

Design: personality rev 4 "Move 1" (`docs/superpowers/specs/2026-10-16-orion-dynamic-personality-graph-design.md`). Juniper approved implementation 2026-11-15.

## Outcome moved

Before: the choice point had two kinds (newest daydream, reads waiting to be told). Live 11-13..14: 3 picks, 2 daydreams. Juniper said the choices should be about people, readings, how Orion thinks about themself, not plumbing.

After: two more kinds reach the same compose call. Live read-only count at 2026-11-15 ~02:00Z: **person 0, chat_thread 2** (both real, non-plumbing: "is it the gameplay or the nostalgia?" about A Link to the Past, and "So what's the plan for 44?"). A 30-day replay of the people rule over live sightings gives 9 distinct visits (~0.3 per day) after the glimpse and ambiguity filters, so the person count of 0 right now is a quiet moment, not a dead source.

## Current architecture

`EndogenousOutreach._choice_topics` builds the daydream topic and appends whatever each `topic_sources` callable returns (only the readings door's `outreach_topics` before this). `filter_unsaid` drops subjects already sent/declined in 7 days at the same version; `rank_topics` orders freshest first and caps at `choice_max_options`. `outreach_object_text` puts the shown topics' text on the binding bid (#3336).

## Architecture touched

- Hub only. New topic sources plug into the existing `topic_sources` seam; no new model call, no new bus channel, no schema change.
- Reads (SELECT only, Hub's `memory_pg_pool`): `person`, `person_link`, `vision_individual`, `vision_individual_sighting`, `vision_office_session`, `chat_history_log`.

## Files changed

- `services/orion-hub/scripts/outreach_world_topics.py`: new. Pure `person_topics` / `chat_thread_topics`, SQL, and two never-raising async sources.
- `services/orion-hub/scripts/outreach_choice.py`: `OutreachTopic.evidence_ref`; `choice_record` stores it per shown topic; `evidence_line` for the situation; docstring documents kinds and the freshness order rule.
- `services/orion-hub/scripts/endogenous_outreach.py`: situation carries `evidence`; docstring.
- `services/orion-hub/scripts/endogenous_outreach_decisions.py`: said-ledger query reads every reason the fold reads.
- `services/orion-hub/scripts/main.py`: registers the two sources when the choice point and their flags are on.
- `services/orion-hub/app/settings.py`, `services/orion-hub/.env_example`: two flags, ON.
- `services/orion-hub/README.md`: the two kinds, rules, flags, readout.
- `services/orion-hub/evals/run_outreach_choice_readout.py` (+ eval test): per-kind shown/picked; `--candidates`.
- `services/orion-hub/tests/test_outreach_world_topics.py`: new. `services/orion-hub/tests/test_outreach_choice.py`: tick integration + ledger regression.

## Schema / bus / API changes

- Added: none (`evidence_ref` is a key inside the existing free-form `result_json.choice.shown[]` and a key in the existing flat `PersonalityChoiceV1.situation` dict; both already accept new keys).
- Removed / Renamed: none.
- Behavior changed: new topic kinds `person`, `chat_thread`; subjects `person:<person_id>`, `chat_thread:<correlation_id>`.
- Compatibility notes: the personality reducer embeds option text whatever the kind.

## Env/config changes

- Added keys: `HUB_ENDOGENOUS_OUTREACH_PEOPLE_TOPICS_ENABLED=true`, `HUB_ENDOGENOUS_OUTREACH_CHAT_THREAD_TOPICS_ENABLED=true`.
- Removed / renamed: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (both keys present in the primary checkout's `services/orion-hub/.env`).
- skipped keys requiring operator action: none.

## Tests run

```text
pytest test_outreach_world_topics.py test_outreach_choice.py + readout eval test                             -> 96 passed
pytest services/orion-hub/tests -k "outreach or reading_finding or settings or env"                          -> 710 passed, 3 skipped
pytest services/orion-hub/tests (full Hub suite, CI's command)                                                -> all pass except 1 FAILED + 4 ERROR that fail identically on main
   (test_fcc_model_labels_api: PermissionError on ~/.fcc/.env on this host; test_consult_belief_store_sql:
   local Postgres CREATE INDEX CONCURRENTLY in a transaction). Host environment, not this change.
python scripts/check_env_template_parity.py                                                                  -> PASS
```

## Evals run

```text
pytest services/orion-hub/evals/test_outreach_choice_readout_eval.py -> 3 passed
run_outreach_choice_readout.py --days 7 (live, read-only):
  3 choices, 3 published; daydream shown 3 picked 2; reading shown 6 picked 1
run_outreach_choice_readout.py --candidates (live, read-only):
  person: 0 candidate(s)
  chat_thread: 2 candidate(s)
    chat_thread:3d2cf3a0-... "...is it the gameplay, which still holds up thirty-odd years on, or the nostalgia?"; the chat went quiet for 13 hours after.
    chat_thread:a1e54ae9-... "So what's the plan for 44?"; the chat went quiet for 2 hours after.
30-day replay of person_topics over live sightings (scratch script): 9 distinct visits, e.g.
  "Exterminator (a camera match for a name Juniper gave) at the front door, ...: the first camera match for Exterminator in 10 days."
  "Teddy (a camera match for a name Juniper gave) at the walkway, ...: a time of day Teddy had not matched at in 14 earlier visits over 5 days."
fetch_said_outreach_subjects against live Postgres (read-only): returns the 7-day ledger with the new reasons param.
```

## Docker/build/smoke checks

```text
No Docker build run: no dependency, Dockerfile or compose change (new keys arrive through env_file).
Live tick path UNVERIFIED until Hub is rebuilt.
```

## Review findings fixed

Review ran in a subagent (read-only, with live SELECTs).

- Finding (must-fix): person text claimed identity the data cannot prove; live 2026-11-14 the doorbell matched Addy, Exterminator and Emerald inside 81 s (almost surely one visitor).
  - Fix: glimpse floor (3 frames or 5 s dwell), drop a visit when another named person matched on the same camera within 2 min, wording "a camera match for a name Juniper gave" / "first camera match", prompt says a match can be wrong.
  - Evidence: `test_a_glimpse_is_not_a_topic`, `test_two_named_people_matching_at_once_is_ambiguous`; replay 36 -> 9 visits.
- Finding: "since <date>" could clamp to the person's own visit ("since Nov 15").
  - Fix: date from OTHER named people's records only; no earlier record -> "on record", no date.
  - Evidence: `test_first_sighting_with_no_other_records_has_no_since_date`.
- Finding: machinery filter dropped prose ("10pm", "2nd", "90s", "and/or", "24/7").
  - Fix: ordinals/times/decades/units exempt; a path needs two slashes or a file extension.
  - Evidence: `test_machinery_shape` cases.
- Finding: "she has not written since" overclaims (only Hub chat is read).
  - Fix: "the chat has been quiet since".
- Finding: a future-dated sighting hid the person's real newest visit.
  - Fix: rows more than 5 min in the future are dropped first. Evidence: `test_a_future_dated_sighting_does_not_hide_the_real_visit`.
- Finding (nit): quotes inside quoted text; "e.g." split. Fixed (double quotes become single; lowercase tail joined).
- Finding (nit, not taken): require the question to be the reply's very last sentence. Tried it: it dropped BOTH live threads, because Orion adds a closing line after the question. Kept the last-paragraph rule and corrected the docstring.
- Finding (noted, not changed): role names ("Mail carrier") are in scope as Juniper named them (README says so); two-level merge chains drop the first link's sightings (none live); a vision split can re-version a visit; `evidence_line` in the situation clips at 200 chars (the decision row keeps every ref whole).

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

(from the clean primary `main` checkout after merge)

## Risks / concerns

- Severity: low. Concern: a person topic is only as right as the camera cluster Juniper named; clusters have been impure before (Teddy/Juniper, 10-27). Mitigation: only typed-name links, the text says only "seen at <place>, <time>", and the prompt tells Orion not to invent what the person was doing.
- Severity: low. Concern: "the chat went quiet after" is a proxy; Juniper may have answered later in other words. Mitigation: the text says exactly the proxy, and the prompt tells Orion not to say she ignored it.
- Severity: medium. Concern: the tick still fires only on a body spark or a novel daydream, but once it fires, open chat threads make it talkable far more often (before, a spark with no new daydream or read recorded `tension_without_content` and composed nothing). So compose calls per day will rise. Every gate (daily cap, cooldown, quiet hours, bid) still holds. Mitigation: after deploy compare decision reasons per day (`tension_without_content` down, compose rows up) with the readout; flags turn each kind off.
- Issues opened/closed: none.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
