## Summary

Live-caught follow-up to #1817. Checking the redeployed panel found `bus` had reappeared with a genuinely new trace row — the single-bare-word floor shipped in #1817 didn't catch it this time because the LLM classified `"bus"` as `"place"` instead of `"concept"`. Same word, different (wrong) type. The floor trusted the model's type field alone; that field isn't reliably consistent call-to-call for the identical word (confirmed twice now: `"concept"` on 08-21, `"place"` on 08-22).

## Outcome moved

`parse_current_turn_llm_signals()` now requires capitalization in addition to type for a bare single-token phrase to survive the person/place carve-out. `"Sarah"`/`"Paris"` (real names) stay capitalized by ordinary convention; `"bus"`/`"glad"` don't.

## Current architecture

`current_turn_llm_signals.py`'s structural floor (from #1817) dropped any single-token phrase not typed `person`/`place`, trusting the model's own classification as the sole signal.

## Architecture touched

`services/orion-cortex-exec/app/current_turn_llm_signals.py`, its tests, its eval fixture, its README section.

## Files changed

- `app/current_turn_llm_signals.py`: capitalization check (`not phrase[:1].islower()`) added on top of the type check; INFO-level log on every bare-word-name acceptance (audit hook for the disclosed interjection gap below).
- `tests/test_current_turn_llm_signals.py`: +3 tests (mistyped-lowercase-still-dropped, uncased-script-survives, INFO-log-fires).
- `evals/run_current_turn_signal_eval.py`: +2 fixtures (bus/place, CJK/place).
- `README.md`: documents both the fix and the disclosed residual gap.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```
services/orion-cortex-exec/tests/test_current_turn_llm_signals.py    25 passed
```

## Evals run

```
python services/orion-cortex-exec/evals/run_current_turn_signal_eval.py
12/12 fixtures correct
```

## Docker/build/smoke checks

N/A — pure Python logic change, no runtime/compose surface touched.

## Review findings fixed

Two full review passes on this branch before push:

**Pass 1** (on the initial capitalization fix, caught before it ever shipped):
- Finding: `phrase[:1].isupper()` is False for every uncased script (CJK, Arabic, Hebrew, Thai) — would wrongly drop a real bare name in one of those, a regression the pre-fix code never had.
  - Fix: `not phrase[:1].islower()` instead — correctly treats "no case signal available" as non-disqualifying while still catching an affirmatively-lowercase Latin word.
  - Evidence: `a03cc2d15`, new test `test_single_bare_word_in_an_uncased_script_survives` + eval fixture `real_uncased_script_place`.
- Finding: the capitalization check doesn't stop a capitalized interjection ("Heck", "Glad") from sailing through if the model ever mistypes it person/place — reproducing the deleted regex detector's exact known failure mode one layer up.
  - Response: disclosed, not fixed — not yet observed live, and CLAUDE.md's metric-quality-gate calls for a live-data sanity check before wiring a new mechanism (a static denylist) in, not before. Made the highest-risk path auditable instead: every bare-word person/place acceptance now logs at INFO.
  - Evidence: README section + inline comment + `test_bare_word_name_acceptance_is_logged_at_info_for_auditability`.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-cortex-exec/.env -f services/orion-cortex-exec/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low, disclosed. The interjection-mistyped-as-person/place gap above is real but unobserved; monitored via the new INFO log rather than closed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1825
