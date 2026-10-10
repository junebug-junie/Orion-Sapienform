## Summary

Since about 2026-09-28 every help request Orion opened was refused. The real cause was that Cursor had hit its monthly usage limit, which resets 2026-10-14. The brief Orion read only said `claude_budget_unobserved` (the second link), so Orion built a "regime break" theory around a billing cap.

- The brief's `refusal_reason` now spells out the whole chain in plain sentences first. Cursor's cause comes first: usage limit / rate limit / auth / program missing / unknown, plus the reset date when Cursor states one (otherwise "reset date unknown"). Then the Claude outcome (refused, failed, or not wired). Then `[codes]`, then Cursor's own words (first 400 characters).
- Orion can now actually see it. Before this, the kickoff "COULD NOT HIRE" section (`format_soft_nudge`) never rendered `refusal_reason` at all. The role-teach progress line said "Cursor budget is spent" for every refusal. Both now show the plain cause.
- Fail-closed behaviour is unchanged: Claude is never called when its meter can't be read. The non-token `cursor_other` path is also unchanged.

## Outcome moved

When Orion asks for help and it is refused, Orion reads why. For the live case that reads: "Cursor unavailable: it hit its usage limit (resets 2026-10-14). Claude fallback refused: this service cannot see Claude's usage meter, so it refuses rather than spend blind."

## Current architecture

`services/orion-curiosity-peer/app/worker.py` works through budget gate → Cursor → Claude-once → persist `PeerBriefV1`. Briefs reach Hub through the graph (`REFUSED_OR_FAILED_RECENT_CYPHER`). From there they go into the kickoff prompt via `format_soft_nudge`, and into the role-teach line via `role_teach_peer_brief` → `turn_orchestrator` → `hire_progress` → `format_budget_spent_progress`. The Atlas and run-story operator surfaces already passed `refusal_reason` through verbatim.

## Architecture touched

- Cursor-cause wording is built deterministically in the peer service.
- Rendering changed in the two Orion-facing formatters.
- One extra key in the in-process Hub payload dict.
- No bus, schema, or registry change: `refusal_reason` is already a free `Optional[str]`, clipped at 4000 characters.

## Files changed

- `services/orion-curiosity-peer/app/cursor_errors.py`: cause classification, reset-date parse (only an explicit `reset(s) ... on M/D/YYYY`), plain sentences, chained reason builder.
- `services/orion-curiosity-peer/app/worker.py`: the three post-Cursor-token paths use the chained reason. The Claude exception is clipped to 200 characters.
- `orion/curiosity/peer_briefs.py`: the nudge renders `Why: <reason>`. Chained reasons are clipped at 700 characters; any other reason at 300.
- `orion/curiosity/role_teach_disclosure.py`, `orion/curiosity/hire_progress.py`, `orion/hub/turn_orchestrator.py`, `services/orion-hub/scripts/curiosity_investigation.py`: thread the reason into the role-teach line. Bare codes keep the old sentence.
- `services/orion-curiosity-peer/tests/test_refusal_reason_chain.py`: new regression tests, using the verbatim live Cursor error.
- `services/orion-hub/tests/test_turn_orchestrator_role_teach_disclosure.py`: payload → line test.
- `services/orion-curiosity-peer/evals/run_contractor_peer_eval.py`: new `usage_limit_chain` case, checked against the rendered nudge.
- `.github/workflows/agency-ask-episodes.yml`: runs the role-teach tests and adds the `requests` dependency they need.
- `services/orion-curiosity-peer/README.md`: "What the refusal says" section.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behaviour changed: the `refusal_reason` text on Cursor-token paths. Nothing parses its prefix; existing tests use substring checks, and `claude_budget_unobserved` is still present.
- Compatibility: briefs written before this deploy still hold bare codes. The role-teach line falls back to "Cursor budget is spent." for those.

## Env/config changes

None. No `.env_example` change; env sync was not needed.

## Tests run

```text
PYTHONPATH=services/orion-curiosity-peer:. python -m pytest services/orion-curiosity-peer/tests -q   -> 77 passed
pytest tests/test_role_teach_disclosure.py services/orion-hub/tests/test_turn_orchestrator_role_teach_disclosure.py tests/test_curiosity_peer_briefs.py tests/test_curiosity_peer_patch0_acceptance.py tests/test_curiosity_peer_kickoff.py -> 45 passed
PYTHONPATH=. pytest services/orion-hub/tests/test_curiosity_investigation.py test_curiosity_self_inquiry.py test_curiosity_introspect.py test_curiosity_routes_runs.py test_turn_orchestrator_ws_frames.py -> 270 passed, 3 skipped
pytest services/orion-hub/tests/test_curiosity_urgent_start.py -> 28 passed
pytest tests/test_curiosity_urgent_prompt.py tests/test_curiosity_incident_report.py tests/test_curiosity_peer_brief_persist.py tests/test_curiosity_atlas_peer_briefs.py ... -> 53 passed
Fresh venv with only the agency-ask-episodes CI deps: new role-teach step 30 passed, peer suite 70 passed
```

## Evals run

```text
PYTHONPATH=services/orion-curiosity-peer:. python services/orion-curiosity-peer/evals/run_contractor_peer_eval.py -> 4/4 passed (incl. new usage_limit_chain)
python services/orion-curiosity-peer/evals/run_agency_episode_eval.py -> 8 passed
```

## Docker/build/smoke checks

```text
No image build; the change is pure Python, with no dependency or compose change.
Live evidence for the fixture: docker logs orion-athena-curiosity-peer, 2026-10-09 10:04:15:
  kind=token_unavailable err=cursor agent exited 1: S: You've hit your usage limit ... Your usage limits will reset when your monthly cycle ends on 10/14/2026.
Post-deploy live brief: UNVERIFIED until the next help request after restart.
```

## Review findings fixed

- Finding: "rate limit" and bare "insufficient" were reported as the monthly usage limit.
  - Fix: a separate `rate_limit` cause ("usually temporary"); "insufficient" is narrowed to funds, credit, or balance.
  - Evidence: `test_rate_limit_is_not_reported_as_the_monthly_cap`, `test_insufficient_permissions_is_not_a_usage_limit`.
- Finding: bare "401"/"403" substrings (e.g. "line 4012") were reported as an auth failure.
  - Fix: the cause step matches `\b40[13]\b`.
  - Evidence: `test_bare_401_substring_is_not_auth`.
- Finding: an uncapped Claude exception could push the codes and Cursor's text past the 700-character nudge clip.
  - Fix: the exception is clipped to 200 characters, and a doubled trailing period is removed.
  - Evidence: `test_long_claude_exception_keeps_codes_inside_nudge_clip`.
- Finding: bare codes (the pre-Cursor gate and old briefs) rendered as "Peer hire refused: budget_limited Do not...".
  - Fix: only the chained format supplies a plain part; otherwise the line keeps "Cursor budget is spent."
  - Evidence: `test_role_teach_bare_code_keeps_plain_sentence`.
- Finding: the nudge now shows raw `cursor_other` stderr, which is third-party text.
  - Fix: reasons not in the chained format are clipped to 300 characters.
  - Evidence: `test_nudge_clips_non_chained_reasons_short`.
- Finding (nit): the date regex missed "resets".
  - Fix: changed to `\bresets?\b`.
  - Evidence: `test_resets_spelling_parses`.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker compose --env-file .env --env-file services/orion-curiosity-peer/.env -f services/orion-curiosity-peer/docker-compose.yml up -d --build
cd /mnt/scripts/Orion-Sapienform && docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low.
  - Concern: the nudge now includes `cursor_other` error text (up to 300 characters) that Orion previously never saw.
  - Mitigation: it is clipped and framed as "Why:" under "COULD NOT HIRE".
- Severity: low.
  - Concern: a pre-Cursor budget-gate refusal (`budget_unobserved` from the Cursor meter) still writes a bare code.
  - Mitigation: the role-teach line falls back to the old sentence, and the nudge shows the code. A follow-up could make it plain too, but existing tests pin that exact string.
- Severity: low.
  - Concern: the role-teach line can pick up a brief with a null `run_id` from another run (a filter that predates this change).
  - Mitigation: none here; noted.

## PR link

(filled in after push)
