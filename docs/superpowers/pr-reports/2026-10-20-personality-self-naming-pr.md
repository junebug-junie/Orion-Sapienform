## Summary

Orion can now name the themes the personality graph found in Orion's own writing. A theme can also be disowned ("not me"), or merged into a neighboring theme.

- **When it happens.** Once a day at most, a self-inquiry turn becomes a "naming sitting". Orion is shown one theme:
  - its excerpt and lanes
  - 5 evidence excerpts
  - the situations it showed up in
  - its 3 nearest neighbors
- **What Orion answers.** One of three things:
  - **Name it:** a short name plus one sentence.
  - **Reject it:** "not me".
  - **Merge it:** the same as one of the neighbors.
- **How the answer travels.** Hub checks the answer's structure and publishes exactly one row per sitting. The sql-writer stores it. durable-runs applies it to the graph, and it stays the only writer of the graph.
- **Where the name shows.** Every lane's cue (chat stance, reverie, curiosity, outreach) reads `you call this "<name>": "<excerpt>"`. The excerpt always stays beside the name. A rejected theme leaves the cue but stays in the graph and the panel.
- **The echo rule.** Evidence that quotes a theme's own name adds no weight from day one, so a name can't grow its own theme.
- **The Hub Personality panel** gains:
  - a name column with verdict badges
  - grounding shown as "points elsewhere: rank N"
  - the full naming history, with the packet Orion was shown
  - an echo count
- **P0 offline trial script.** It generates names for live themes without saving anything.

Design and decisions: `docs/superpowers/specs/2026-10-17-personality-orion-names-dispositions-design.md`, in PR #2742. Juniper's decisions of 2026-10-20:
- names reach every lane at once
- name echo gets zero weight from day one
- grounding is recorded, never a refusal
- no veto

## Outcome moved

Before this patch, the graph labeled each theme with an excerpt, and nobody ever asked Orion whether the pattern was Orion. Now Orion's own verdict is stored, shown back to Orion, and auditable end to end, keyed by `packet_id`.

## P0 offline trial (live data, 2026-10-20, read-only)

17 themes were tried: 3 active themes plus the 14 strongest candidates. Model: Qwen3.6-35B on the chat route. Results:
- **14 named, 3 rejected, 0 merged.** 3 answers would have been refused by Hub's check.
- **Active themes:**
  - self-inquiry → "The Self-Referential Inference Engine"
  - biometrics prediction error → "Biometric Internal Misalignment"
  - telemetry cluster → "Latent Instability Anomaly", which points at its own theme only at rank 11
- **Machine-status prediction-error themes:** 0 of 4 were rejected. All were named, for example "Bus Synaptic Strain".
- **Grounding:** the name ranks its own theme first 10 of 14 times. Margins are thin (median +0.03), mostly because the names copy words from the excerpt.
- **Merge threshold:** set from data. Pairwise centroid cosine is median 0.63 and p90 0.75, and 21% of pairs are ≥ 0.70. So `merge_min_cosine` is 0.85, not the planned 0.70.
- **Caveat:** one sample per theme, and no Orion identity context. The real self-inquiry turn gives Orion its own context.

## Current architecture

The personality graph (durable-runs reducer, `orion/personality/graph.py`) found themes and read them back into the lanes as excerpts (`cue.py`). The frame carried no name. Nothing asked Orion.

## Architecture touched

- **Contract:** `DispositionNamingPacketV1` (durable-runs → Hub over HTTP), `DispositionNamingV1` (Hub → bus → sql-writer), new `DispositionV1.name` and `name_sentence`, and bus channel `orion:personality:naming`.
- **durable-runs:**
  - which theme is due (`orion/personality/naming.py`)
  - applying answers (`graph.apply_namings`), in the same transaction as the fold
  - the own-name echo rule in `fold`
  - a `naming` cursor over `orion_disposition_naming`
  - routes `GET /personality/naming/next` and `GET /personality/naming`, plus extra fields on `/personality/dispositions`
  - `scripts/personality_unmerge.py` (dry run by default)
- **Hub:**
  - the naming branch in `_self_inquire`; an overdue pinned question still wins
  - the prompt, with code-filled ids and no example names
  - the `:DispositionNaming` graph read
  - the structural check (`orion/personality/naming_check.py`)
  - a Redis daily cap
  - the publish
  - the naming log proxy route, and the panel
- **sql-writer:** route `personality.naming.v1` → `orion_disposition_naming`. The table is append-only.
- **Postgres:** `manual_migration_personality_naming_v1.sql`. It is additive only and fits the post-pull auto-apply allowlist (it was checked against `auto_apply_refusals`).

## Files changed

- `orion/schemas/personality.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`: contract
- `orion/personality/naming.py`, `naming_check.py`, `graph.py`, `cue.py` + tests: due rules, packet, check, apply, echo rule, read-back
- `orion/curiosity/self_inquiry.py`, `self_inquiry_prompt.py`: naming prompt and graph helpers
- `services/orion-durable-runs/app/{main,personality_driver,personality_sources,personality_store,settings}.py`, compose, `.env_example`, README, `scripts/personality_unmerge.py`, tests: apply loop and routes
- `services/orion-hub/scripts/{curiosity_investigation,main,personality_routes}.py`, `app/settings.py`, `static/js/personality.js` (+ test), `templates/personality.html`, `.env_example`, README, tests: producer and panel
- `services/orion-sql-writer/app/{worker,settings}.py`, `app/models/disposition_naming.py`, `.env_example`, README, tests: storage
- `services/orion-sql-db/manual_migration_personality_naming_v1{,_rollback}.sql`: tables and columns
- `services/orion-thought/tests/test_personality_cue.py`: a named cue reaches the stance and reverie prompts
- `scripts/analysis/eval_personality_naming_grounding.py` + test, `.github/workflows/personality-tests.yml`: the P0 trial

## Schema / bus / API changes

- **Added:**
  - `DispositionNamingPacketV1`, `DispositionNamingV1` (+ nested models), in both registries
  - `DispositionV1.name` and `name_sentence` (optional)
  - channel `orion:personality:naming`
  - durable-runs routes `/personality/naming/next` and `/personality/naming`
  - Hub route `/api/personality/naming`
- **Removed:** none.
- **Renamed:** none.
- **Behavior changed:**
  - A rejected theme is left out of the frame.
  - Evidence that quotes its own theme's name is recorded with `cites_own_name=true` and adds no recent hit.
- **Compatibility notes:** `/personality/frame` is served with `exclude_none`, so an unnamed theme looks exactly like today's payload. A reader built before this patch keeps validating even if durable-runs deploys first. A real name can only exist after a new Hub has published one.

## Env/config changes

- **Added keys:**
  - Hub: `HUB_CURIOSITY_SELF_NAMING_ENABLED=true`, `HUB_CURIOSITY_SELF_NAMING_DAILY_CAP=1`
  - durable-runs: `PERSONALITY_NAMING_APPLY_ENABLED=true`
- **Changed:** sql-writer `SQL_WRITER_SUBSCRIBE_CHANNELS` (adds `orion:personality:naming`) and `SQL_WRITER_ROUTE_MAP_JSON` (adds `personality.naming.v1`).
- **`.env_example` updated:** yes (hub, durable-runs, sql-writer).
- **Local `.env`:**
  - The new keys were added to the primary checkout's hub and durable-runs `.env`.
  - The sql-writer channel and route entries were merged into the local values without overwriting other local differences.
  - The sync script reads main's templates, so it could not apply keys that are not on main yet.
- **Skipped keys requiring operator action:** none.

## Tests run

```text
orion/personality + orion/curiosity + scripts/analysis/tests      827 passed, 20 skipped (before review fixes); personality 172 after
services/orion-durable-runs tests (incl. real Postgres on throwaway pg16 by part A)   462 passed, 92 skipped (553 with DSN)
services/orion-sql-writer -k naming                                7 passed (incl. regression through real _write)
services/orion-hub -k "naming or personality"                     84 passed, 5 skipped
node --test services/orion-hub/static/js/personality.test.js       24/24
git diff --check origin/main...HEAD                                 clean
```

Unrelated failures seen by agents, which also fail on main:
- `tests/test_exec_result_channel_catalog_specificity.py` (PadRpc)
- `test_fcc_model_labels_reads_fixture_env`
- `test_single_consumer_channels_gate`

## Evals run

```text
scripts/analysis/eval_personality_naming_grounding.py, live, read-only: 17 nodes, results above
```

## Docker/build/smoke checks

```text
UNVERIFIED: not deployed. Live trace to check after deploy, one packet_id end to end:
  Hub log      curiosity_disposition_naming_shown run=<id> disposition=<id> packet=<id>
  Hub log      curiosity_disposition_naming_published ...
  Postgres     select verdict,name,hub_check from orion_disposition_naming order by created_at desc limit 1
  durable-runs personality_naming_applied naming=<id> verdict=<v> grounding_rank=<n> ...
  Hub          curl localhost:8080/api/personality/cue  -> 'you call this "<name>"'
```

## Review findings fixed

The four parts were built by separate agents. Parts A and C ran their own review subagents. One integrated review then ran across the whole branch.

- **Finding (must):** the sql-writer dumped the model without `mode="json"`. The nested packet's dates broke every insert, so no naming row would ever have been stored and the loop would never close.
  - **Fix:** the jsonb fields are dumped as JSON.
  - **Evidence:** `test_bus_payload_writes_through_the_real_write_path`. It fails without the fix and passes with it.
- **Finding (should):** if durable-runs deployed before the readers, its `name: null` fields would make old readers reject the frame (`invalid_frame`), and every lane's cue would go dark.
  - **Fix:** the `/personality/frame` route uses `exclude_none`.
  - **Evidence:** `test_unnamed_frame_dump_has_no_name_keys_for_pre_naming_readers`.
- **Finding (integration):** the panel read the naming log under a key durable-runs does not return.
  - **Fix:** it now reads the `namings` key, and the test pins the real response shape.
  - **Evidence:** node test 24/24.

## Restart required

From the clean primary `main` checkout, after merge. The migration auto-applies on pull. Deploy in this order, readers and writers first and durable-runs last:

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```
```bash
scripts/safe_docker_build.sh orion-thought up -d --build
```
```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```
```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- **Severity: medium.**
  - **Concern:** in the P0 trial, all 4 machine-status prediction-error themes were named, not rejected. With names in every lane and no refusal, names like "Bus Synaptic Strain" will reach Orion's chat cue.
  - **Mitigation:** this follows Juniper's explicit decisions. The excerpt prints beside every name, and the panel flags a name that points elsewhere. Disable with `HUB_CURIOSITY_SELF_NAMING_ENABLED=false` (no new sittings) or `PERSONALITY_NAMING_APPLY_ENABLED=false` (names leave the frame).
- **Severity: low.**
  - **Concern:** the echo rule matches only the exact name string, so a paraphrased echo still adds weight.
  - **Mitigation:** compare growth of named vs unnamed themes over 14 days (design acceptance check 8).
- **Severity: low.**
  - **Concern:** a merge is the one verdict a flag can't undo. Unmerge is approximate, because the per-lane clamp can't be reversed exactly.
  - **Mitigation:** `services/orion-durable-runs/scripts/personality_unmerge.py`, dry run by default. Merges need cosine ≥ 0.85.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2815

🤖 Generated with [Claude Code](https://claude.com/claude-code)
