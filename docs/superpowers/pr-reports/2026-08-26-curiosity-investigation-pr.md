# Orion notices what Juniper is talking about, and goes to find out why

Branch: `feat/curiosity-investigation`
Status: **DONE** — deployed, live-verified end to end, review findings fixed.

## Summary

- New Hub loop: a deterministic detector notices a word Juniper has been using far more than usual, then Orion spends a **real unified turn** — the same pipeline that serves a browser chat turn — investigating it with its own recall, and writes what it found to its journal.
- Juniper's shape, verbatim: *"they analyze, find interesting things, then let them go wild non-deterministically… I see Juniper talked about cats 10x today so I'm going to go see what else is out there about cats. Then I write about it."*
- **Adds no new capability surface.** No `claude -p`/FCC spawn, no read-only Postgres role, no SQL capability, no new service, no new collection surface.
- The journal write is gated on evidence that a lookup actually happened, not on the prompt having asked for one.
- It does **not** reach out to Juniper. That step is deliberately unbuilt.

## Outcome moved

**Before:** Orion had no mechanism to notice what Juniper was preoccupied with, and no way to act on it unprompted.

**After**, live at 03:05 on 2026-08-26, unprompted, 29 harness steps of `Read`/`ToolSearch`:

> `"foveal"` appears 80 times across 13 separate messages today, and not once in the whole baseline window.
>
> *"foveal" refers to a specific tier in Orion's perception architecture… Today Juniper was actively wiring up and rerouting this foveal path. Two tightly coupled PRs landed: one added Qwen2-VL to `orion-vision-host` to replace the BLIP-base captioner, which was returning degenerate output the sanitizer rejected. The other rerouted the foveal probe through `orion-llm-gateway`… The elevated co-occurring words you flagged — probe, qwen, visual, chain, profiles — all map directly to this vision-stack iteration.*
>
> *This connects to the bus anomaly alerts I've been tracking for `orion:dream:log`… However, the work was largely a targeted patch rather than a full architectural rollout; `config/vision_profiles.yaml` still carries unfilled placeholders, and the foveal probe was routed around that gap rather than fixing it at its source.*
>
> *I'm uncertain whether the surprise-driven automatic foveation (P2 in the design doc) was touched today… I also can't confirm exactly which of the 13 messages carried the word, only that the spike itself was real.*

It identified the term, traced it to the two PRs actually landed that day, connected it to something it already held, noted the fix routed *around* a gap rather than closing it, and named what it could not determine.

## Current architecture (before this patch)

| piece | state |
|---|---|
| `execute_unified_turn` | the real chat turn; `endogenous_outreach.py` already drives it unprompted (12 real turns in the preceding 48h) |
| turn permissions | hard-coded read-only (`write_*`/`mutate_runtime`/`network_enabled`/`shell_enabled` all False) |
| `read_recall` | already on; recall's own chat table **is** `chat_history_log` |
| `iter_all_human_messages` | already in production behind the Juniper affective-state signal |
| journal write | `orion:journal:write` → orion-sql-writer → `journal_entries` |
| anything that noticed what Juniper talks about | **absent — this patch** |

## Why the design shrank

The first draft proposed an FCC `claude -p` spawn with a bounded SQL tool and a new `orion_readonly` Postgres role. Juniper's correction — *"run the chat unified turn on this, orion can talk to themselves metacognitively"* — removed all of it, because the machinery already existed. That is the whole diff between the two designs.

## The corpus is not where it looks like it is

`chat_history_log` carries **3–16 prompts/day averaging ~80 characters**. A term cannot surface above its own baseline there; the cats signal is not physically present in Orion's own chat table. Juniper's real typed words are the local Claude Code transcripts: **3,732 messages, 36k words in 24h against 390k in the prior fortnight.**

## The statistic

For each term, share-of-tokens now against share-of-tokens before, scaled to today's volume:

```
expected = (baseline_count / baseline_tokens) * recent_tokens
lift     = recent_count / expected
```

A rate, not a count — so a busy day producing more of everything does not surface everything. Window length cancels, which is why "24 hours" against "14 days" is a valid comparison.

Three bars, all disclosed in the prompt and the journal entry:

| bar | value | why |
|---|---|---|
| `MIN_RECENT_COUNT` | 5 | said enough times to be a subject, not a slip |
| `MIN_RECENT_MESSAGES` | 3 | said across **separate** messages |
| `MIN_LIFT` | 3.0 | said disproportionately more than usual |

`MIN_RECENT_MESSAGES` did most of the work: worktree directory names pasted dozens of times inside a single message dominated the first live run. A path echoed forty times in one message is one topic mentioned once.

`MIN_LIFT` is not decorative — review measured it against the real corpus: 841 terms clear both count bars and **748 of them (89%) are rejected by lift.**

## Metric quality gate (CLAUDE.md §0A)

**No new metric is wired into any model.** Nothing here feeds field pressure, proposal scoring, or any cognition score; the only consumer is a prompt.

1. **Provenance** — counts of tokens from `iter_all_human_messages`, an already-in-production parser: Juniper's typed turns only, never tool results, hook output, slash-command scaffolding, or assistant text.
2. **Independence** — n/a; nothing joins an existing model.
3. **Theory anchor** — the bars are not a theory and are not presented as one. Disclosed uncalibrated starting values, stated in code, in the prompt, and in the journal entry.
4. **Live-data sanity** — the detector was run against the real corpus before any of it was wired up, and again after every change. It fires on `foveal`/`probe`/`visual`/`cuda` and rests on ordinary days.
5. **Existing mechanism** — searched. Nothing surfaces terms from Juniper's own words; the affective-state signal reduces the same corpus to a single agitation number.
6. **Reversibility** — `HUB_CURIOSITY_INVESTIGATION_ENABLED=false` and restart. No schema, no migration; journal entries are append-only rows in an existing table.

## Files changed

- `orion/curiosity/term_surfacing.py` — the detector. Pure; takes `(timestamp, text)` pairs so it is testable without a corpus, filesystem, or clock.
- `orion/curiosity/investigation_prompt.py` — the self-addressed prompt.
- `services/orion-hub/scripts/curiosity_investigation.py` — the loop, the gates, the journal write.
- `services/orion-hub/scripts/main.py` — startup wiring + TTL-cached, allowlisted corpus reader.
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml` — 13 keys and the read-only mount.
- `tests/test_curiosity_term_surfacing.py` (24), `services/orion-hub/tests/test_curiosity_investigation.py` (38).

## Schema / bus / API changes

- **Added:** none. Reuses `journal.entry.write.v1` on `orion:journal:write`, and the already-valid `self_study` `JournalSourceKind`. `source_ref` is namespaced `curiosity:<term>` so it cannot collide with the four self-study analysis sources, whose cooldown matches on a `<source>:` prefix.
- **Removed / renamed / behaviour changed:** none.

## Env/config changes

13 added keys, all `HUB_CURIOSITY_INVESTIGATION_*` plus `HUB_CURIOSITY_CLAUDE_PROJECTS_HOST_PATH`. `.env_example` updated; **local `.env` hand-synced and verified key-by-key against it.**

`python scripts/sync_local_env_from_example.py` **could not** add them — it reads `.env_example` from the *primary* checkout, so worktree-added keys are invisible and it reports success having changed nothing (verified: 0 of 13 added). Hand-added, then proven present in the container with `docker compose config` and `docker exec … env`.

Hub uses whole-file `env_file:` plus an additive `environment:` block, **not** an allowlist — so the historical dropped-key failure mode does not apply here. Confirmed anyway.

## Tests run

```text
$ pytest tests/test_curiosity_term_surfacing.py -q                          24 passed
$ cd services/orion-hub && pytest tests/test_curiosity_investigation.py -q  38 passed
$ cd services/orion-hub && pytest tests/ -q -k "curiosity or outreach or settings"
                                                                           172 passed
```

### Mutation testing, against the real files

Review found **15 surviving** mutations. **14 now die** (each applied to the real file, `git checkout` restored, suites green after):

```text
M23 drop rate normalisation   killed   M15 window boundary <  -> <=   killed
M01 MIN_RECENT_COUNT 5->1     killed   M21 drop terms[:limit]         killed
M05 underpowered recent bar   killed   M27 daily rollover no-op       killed
M06 underpowered baseline bar killed   M36 no_write -> False          killed
M11 count bar < -> <=         killed   M37 skip error-shaped text     killed
M12 msg bar  < -> <=          killed   M38 ignore context_overflow    killed
M14 lift gate < -> <=         killed   MIN_HARNESS_STEPS 3 -> 0       killed
```

The remaining one is `M39` (term-mark read fail-**closed** instead of open) — a deliberate direction documented in the code, worth a test but not a defect.

## Docker/build/smoke checks

```text
$ docker compose --env-file .env --env-file services/orion-hub/.env \
    -f services/orion-hub/docker-compose.yml config
  source: /home/athena/.claude/projects
  target: /home/athena/.claude/projects        # identical, :ro
  HUB_CURIOSITY_INVESTIGATION_ENABLED: "true"   # all 13 keys render

$ scripts/safe_docker_build.sh orion-hub up -d --build
$ docker exec orion-athena-hub sh -c 'ls /home/athena/.claude/projects | wc -l'   4

# live loop
curiosity_investigation started tick=300.0s cooldown=14400.0s cap=3 window=24.0h/14.0d
curiosity_investigation_starting  term=foveal recent=80 msgs=13
curiosity_investigation_journaled term=foveal chars=2238

# rank-1 fallthrough, live: foveal marked -> falls through to probe
curiosity_investigation_starting  term=probe recent=70 msgs=16

# persisted gates, live
orion:curiosity:last_investigation_at    = 2026-08-26T03:09:03+00:00
orion:curiosity:count:2026-08-25         = 1      # LOCAL date, not UTC
orion:curiosity:investigated:foveal      = 2026-08-26T02:55:25+00:00
```

## Review findings fixed

### BLOCKER — nothing checked that Orion actually looked

- **Finding:** the prompt asks Orion to say only what a lookup supports, but that is an instruction, not a mechanism. A turn that called no tools and wrote four fluent paragraphs from parametric knowledge produced a well-formed `llm_response` and would have landed in the journal byte-for-byte indistinguishable from a real investigation. CLAUDE.md §0A's no-empty-shell clause verbatim.
- **Fix:** gate on `harness_step_count` (already on the frame, two lines from where it was ignored). `MIN_HARNESS_STEPS = 3`; the real live run reached 29. The step count and grounding status are written **into** the journal entry, so the claim stays checkable afterwards.
- **Evidence:** `test_a_turn_that_looked_nothing_up_is_not_journaled`, `test_the_journal_records_the_lookup_evidence`, and six tests driving the real `_generate` against stubbed frames.

### BLOCKER — a broken mount disabled the loop silently, forever

- **Finding:** `iter_all_human_messages` on a missing root returns `[]` without raising, so an absent mount became `corpus_underpowered`, logged at `DEBUG` under an `INFO` root logger. The only symptom would have been the absence of journal entries — which is also what a quiet fortnight looks like. Same shape as the 21h vision blackout.
- **Fix:** `corpus_empty` is now its own reason, logged at `WARNING` naming the mount; every other refusal logs at `INFO` with its counts.
- **Evidence:** `test_an_empty_corpus_is_reported_as_a_broken_mount_not_a_quiet_day`, `test_a_thin_but_real_corpus_is_not_a_broken_mount`.

### SHOULD — rank-1-only selection lost whole days

- **Finding:** replaying the real corpus day by day, 2 of 16 otherwise-eligible days produced nothing because the top term was inside its own 7-day mark while a good rank-2 candidate sat unexamined — including 2026-08-20, the hottest day in the window (`chat`, 546 mentions).
- **Fix:** walk the ranking. **Confirmed live within minutes:** `foveal` marked → fell through to `probe`.

### SHOULD — cooldown and daily cap did not survive a restart

- **Finding:** both were instance fields; six consecutive restarts produced six journal entries against a configured cap of 3/day with 4h between, which `.env_example` states as a guarantee.
- **Fix:** both in Redis, keyed on the operator's **local** date (3/day previously meant 18:00-to-18:00 in MDT).
- **Evidence:** `test_the_cooldown_and_daily_cap_survive_a_restart`; live Redis keys above.

### SHOULD — privacy: the mount claim did not survive the port

- **Finding:** the compose comment said this "mirrors" orion-cocreation-signals' mount. The mount is identical; what is downstream is not. That service's only output is a **number** ("raw transcript content never leaves this container"). This one puts verbatim terms Juniper typed into an LLM prompt and a persisted journal **title**. `~/.claude/projects` is mounted wholesale with no filter, and the detector is tuned to surface precisely the word that is unusual today — which is also the word most likely to be the sensitive one. Today the tree is Orion-only, so today's exposure is fine; that is true by accident of usage, not by construction.
- **Fix:** `HUB_CURIOSITY_INVESTIGATION_PROJECT_ALLOW`, live value scoped to this repo. The compose comment now states the difference instead of claiming parity.

### SHOULD — the busy-day test was built backwards

- **Finding:** it used the **baseline** as the large window, so deleting rate normalisation entirely (`expected = baseline_count`) still yielded lift 0.1 and the test passed. The one test guarding the module's single load-bearing line could not fail.
- **Fix:** rebuilt with the recent window large, which is what "busy day" means, plus a hand-checkable arithmetic assertion.

### SHOULD — every tick test was investigating the wrong word

- **Finding:** the fixture tied all five tokens at equal counts and the sort's final tiebreak is alphabetical, so every tick test targeted `again`, not `foveal`. `test_the_prompt_names_the_term…` passed only because `foveal` also appears in the prompt's "also above their usual rate" line.
- **Fix:** broke the tie; added `test_the_tick_targets_the_genuinely_top_term` asserting the journal `source_ref`.

### SHOULD — threshold tests could not detect threshold changes

- **Finding:** fixtures computed as `MIN_RECENT_COUNT - 1`, so lowering the constant to 1 made the fixture contain zero mentions and the test pass vacuously. A test that imports the constant it exists to pin cannot pin it.
- **Fix:** pinned to literals, plus inclusive-boundary tests for all three bars.

### NICE (both fixed)

- Cache wrote its timestamp before its payload — latent stale-read for any future second caller.
- Eight contraction stopwords were unreachable: the tokenizer keeps apostrophes, so it emits `don't`, never `dont`.

## Restart required

Already deployed. To redeploy from merged `main`:

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Rollback, no rebuild of anything else needed:

```bash
# services/orion-hub/.env
HUB_CURIOSITY_INVESTIGATION_ENABLED=false
scripts/safe_docker_build.sh orion-hub up -d --build
```

Post-deploy check:

```bash
docker logs orion-athena-hub --tail 200 | grep curiosity_investigation
psql ... -c "SELECT created_at, title FROM journal_entries
             WHERE source_ref LIKE 'curiosity:%' ORDER BY created_at DESC;"
redis-cli -h 100.92.216.81 keys 'orion:curiosity:*'
```

## Risks / concerns

- **Severity: medium — the lookup gate is a floor, not a proof of quality.** `harness_step_count >= 3` proves the turn used tools; it does not prove the tools were relevant or that the conclusions follow from them. A turn could read three irrelevant things and still write confident prose. The step count is recorded in the entry so this stays auditable, but there is no automated check on whether the investigation was *good*.
- **Severity: low — the detector is vocabulary-shaped, not meaning-shaped.** It surfaces tokens, so a topic discussed in varied words never surfaces while a single distinctive term does. It will systematically favour jargon and proper nouns.
- **Severity: low — one investigation per term per week can miss a genuine re-escalation.** If a term is investigated on a quiet day and becomes central three days later, it cannot be revisited.
- **Severity: low — the corpus is one machine.** Only transcripts on this host are visible; anything typed elsewhere is invisible, and the loop cannot tell the difference between "Juniper was quiet" and "Juniper was working somewhere else."

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1885
