## Summary

- The repo is public. Docs, PR reports, specs, evals and test fixtures were quoting Juniper's private life: names of family and coworkers, her home city and its coordinates, health details about people close to her, and direct quotes of her chat. This patch removes them from the current tree.
- Docs get role placeholders (`[spouse]`, `[child A]`, `[boss]`, `[team member 1]`, `[home city]`, `[quote from chat, redacted]`) so each sentence still says what it meant.
- Tests and fixtures get obviously fake names and a fake town, so they still exercise the same code paths.
- Things Orion actually reads at runtime keep their real values, moved to gitignored local files (the `.env` pattern), so nothing Orion sees changes.
- New CI gate: every tracked file is checked against a list of SHA-256 hashes of the terms. The list holds hashes only, never the words.
- History is NOT rewritten (out of scope; would need a force push to main). See "Risks".

## Outcome moved

A public reader of the current tree no longer finds the named people, the home city, health details or private chat quotes. A new commit that reintroduces one of the hashed terms fails CI.

## Current architecture

Real personal details lived in committed files: `.env_example` values for the home place and weather point, a hardcoded name in Mind's relational-cue regex, the AI-town human card and its patch, two journal-row SQL dumps, and many docs/tests that quoted live data.

## Architecture touched

- Runtime inputs, real values now local only:
  - `services/orion-cortex-exec/.env_example`, `services/orion-hub/.env_example`: home locality, home location, weather lat/lon blanked. Local `.env` already holds the real values (sync reported them as "diverged", unchanged).
  - `services/orion-mind/app/engine.py`: relational name cues read from `ORION_MIND_RELATIONAL_NAMES` (local `.env`) instead of a literal in code. Local `.env` set to the previous value, so behaviour after restart is identical.
  - `services/orion-ai-town`: generator reads `cards/town_cards.local.yaml` first (gitignored, real), else the committed redacted `town_cards.yaml`. Outputs follow the source (`generated/*.local.txt` for real cards). `patches/orion-human-juniper.patch` carries a token in place of the join description; `scripts/apply_upstream_patches.sh` renders it from the cards. `services/orion-embodiment` bootstrap prefers `orion_town_card.local.txt`.
- Gate: `scripts/check_personal_terms.py`, `config/privacy/personal_terms.sha256`, `tests/test_no_personal_terms.py`, step in `.github/workflows/orion-static-gates.yml`.

## Files changed

73 files: 22 docs/PR reports/specs/README, 1 eval fixture, 30 tests/fixtures, 2 SQL dumps (row data removed, schema kept), 2 `.env_example`, AI-town generator/patch/cards/README, embodiment bootstrap, Mind engine, gate script + test + hash list + workflow, `scripts/check_settings_defaults.py` (two exemptions no longer needed).

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: none at runtime on this host (verified below). A fresh host without the local files gets blank place/weather and the redacted AI-town card.

## Env/config changes

- Added keys: `ORION_MIND_RELATIONAL_NAMES` (orion-mind; blank in template).
- Changed template values: `ORION_SITUATION_LOCALITY`, `ORION_SITUATION_HOME_LOCATION`, `ORION_SITUATION_WEATHER_LAT`, `ORION_SITUATION_WEATHER_LON` (cortex-exec, hub) now blank.
- local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-mind orion-hub orion-cortex-exec`: yes; real values kept; `ORION_MIND_RELATIONAL_NAMES` set by hand in the primary checkout's `services/orion-mind/.env`.
- skipped keys requiring operator action: none.

## Runtime equivalence (what Orion sees, before vs after)

- Situation block: local `.env` values unchanged (sync left them in place).
- Mind relational cue: same name list, now from env (`tests/test_relational_names_env.py`).
- AI town: rendered human patch from the local cards is byte-identical to the old committed patch; NPC `Descriptions`, Juniper's join blurb and Orion's town card are identical between the old cards and `town_cards.local.yaml`; the live `upstream/convex/world.ts` already contains that same blurb.
- One deliberate text change: the format example inside Mind's semantic-synthesis system prompt (`services/orion-mind/app/synthesis.py`) used a real name; it now uses a fake one. It is an output-shape example, not a fact Orion is meant to know.

## Tests run

```text
tests/test_no_personal_terms.py                      7 passed
orion/memory/episode, orion/situational, model_replay 541 passed, 22 skipped
cortex-exec 23, cortex-orch 26, durable-runs 33, hub 247 (7 skipped), mind 120, ai-town 136 (1 skipped): passed
recall: 36 passed, 1 failed; embodiment: 80 passed, 14 failed -- same failures on a clean checkout of main (pre-existing, unrelated)
check_settings_defaults orion-hub --example-drift: OK; check_env_template_parity: PASS
```

## Evals run

```text
No eval harness for a privacy scrub. The gate itself runs over the whole tree in CI (~10s).
```

## Docker/build/smoke checks

```text
No image rebuild needed. AI-town apply script run against a copy of the live upstream: the human patch behaves the same as before (the live upstream had already drifted for earlier patches; unchanged by this PR).
```

## Review findings fixed

- Finding: AI-town drift test always skipped on the real host (compared the live upstream to the redacted example card).
  - Fix: build the expected blurb from `gen.load_cards()` (local first).
  - Evidence: `tests/test_generate_descriptions.py` passes; skip now only fires on a vanilla upstream.
- Finding: the gate missed names inside camelCase, accented spellings, longer email local parts, `%40` emails.
  - Fix: case-aware tokenizer (`nameMemory` -> name, memory), accent folding, dotted-suffix email variants. Plurals/suffixes are a documented limit.
  - Evidence: `test_line_hits_catches_camel_case_accents_and_email_variants`.
- Finding: the apply script leaked a temp file holding the rendered description and gave a misleading error without local cards.
  - Fix: `trap` cleanup, explicit render-failure message, note when `town_cards.local.yaml` is missing.
  - Evidence: `bash -n`; rendered patch still byte-identical.
- Finding: a few remaining private paraphrases in touched docs (a family/feeling paraphrase, a hotel chain).
  - Fix: replaced with placeholders.
  - Evidence: gate clean.
- Finding: history still holds everything (HIGH). Not fixable without a force push; recorded under Risks.

## Restart required

```bash
./scripts/safe_docker_build.sh orion-mind up -d --build
```

Only orion-mind, after merge and pull, so the env-driven name list replaces the hardcoded one. Situation and AI-town are unchanged at runtime.

## Risks / concerns

- Severity: high. Concern: git history still holds every detail removed here (docs, tests, `.env_example` coordinates, SQL dumps, AI-town card). Mitigation: only a history rewrite of main plus a force push removes it; Juniper's call. Commit list is in the hand-off report, not here.
- Severity: medium. Concern: the hashes of common first names can be recovered by hashing a name list. Mitigation: they stop casual reading and grep; the plaintext list is not in the repo.
- Severity: low. Concern: Juniper's surname is kept: it is the AI-town player name and thread key at runtime, and the author name on thousands of commits. Her email was removed from README/pyproject but remains in commit metadata.
- Severity: low. Concern: a fresh host needs `town_cards.local.yaml` and the `.env` values copied by hand.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
