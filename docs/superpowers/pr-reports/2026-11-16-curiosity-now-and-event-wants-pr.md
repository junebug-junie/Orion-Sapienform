## Summary

- Curiosity now reads Orion's now. At the start of every investigation and self-inquiry run, Hub asks orion-durable-runs what Orion's other organs lived since the last curiosity run looked (`GET /temporal-self/now?organ=curiosity`, the endpoint's own per-organ cursor; see Review findings for why there is no explicit `since`). The kickoff shows those items, each with an `[experience:<id>]` tag, so a painting or a prediction surprise can become what curiosity looks into.
- At the end of the turn, Hub sends up to 3 distinct `[experience:<id>]` cites from Orion's text (shown ids first) to `POST /temporal-self/now/use`. Only ids that were shown are stored (durable-runs checks them; invented ones are counted). A failure never touches the turn.
- The `shown_id` and shown ids are kept with the run on its decision row (`curiosity_offer_decisions.constants.now_shown_id` / `now_event_ids`), so a durable turn that runs in another process attempt, or after a Hub restart, still records its cites. The self-inquiry naming branch and urgent runs show no now (out of scope).
- Curiosity event wants, step 3: a new philosophical finding whose novelty is in the top tenth of the last 30 days of philosophical findings posts one shadow event want (family `hub.curiosity_finding`, verb `infer`, `event_ref curiosity_run:<run_id>`). Prime's menu labels it with the finding's own words from the run's finish row.
- Organ registry: curiosity now names its `reads_now` module (exemption removed) and owns the new family.

## Outcome moved

- Integration: curiosity is the first organ that reads the now and records checked cites through the shared contract (spec Part A, patch 3, curiosity slice). Before: every organ was exempt from reading.
- Attention: curiosity findings can compete in the want market. Before: only reads, camera asks, office visits and the cooling act posted event wants.

## Current architecture

Patches 1 (#3453: `orion/temporal_self/now.py`, the two endpoints) and 2 (#3454: `config/integration/organs.yaml`, `scripts/check_integration_seam.py`) were merged; no organ read the now yet. The finding register (#3437) wrote one label + novelty row per finished curiosity turn, but nothing read it.

## Architecture touched

- orion-hub: curiosity kickoff (both lines), turn end (all three places a curiosity turn ends: durable turn request, in-process investigation, in-process self-inquiry), a new event-want loop started in Hub main.
- orion-durable-runs: Prime's event-wants menu gains a `curiosity_run:` label read from its own `substrate_durable_run_state` finish row.
- Config: `config/physiology/families.yaml`, `config/physiology/contract.yaml`, `config/integration/organs.yaml` (curiosity entry only).

## Files changed

- `orion/curiosity/now_section.py` (new): pure kickoff section and cite parser.
- `orion/curiosity/kickoff_prompt.py`, `orion/curiosity/self_inquiry_prompt.py`: `now=` parameter; nothing rendered when None.
- `services/orion-hub/scripts/curiosity_now.py` (new): the now read, shown_id persistence, cite post. Curiosity's `reads_now`.
- `services/orion-hub/scripts/curiosity_investigation.py`: wiring (read at kickoff, cites at every turn end).
- `services/orion-hub/scripts/curiosity_event_wants.py` (new): the producer.
- `services/orion-hub/scripts/main.py`, `app/settings.py`, `.env_example`, `README.md`: flags and start/stop.
- `services/orion-durable-runs/app/event_wants_menu.py`: `curiosity_run:` prefix and `finding_label`.
- `config/physiology/families.yaml`: `hub.curiosity_finding` (tier 2, 12 h window, no lanes; reasoning in the entry).
- `config/physiology/contract.yaml`: the producer registered as a display (shadow) reader.
- `config/integration/organs.yaml`: curiosity `reads_now`, new family; `exempt_read` removed.
- `tests/test_check_integration_seam.py`: `planted(repo_root=...)` copies every organ's real reader into the planted root, so the read-back test keeps judging one organ at a time now that a real reader exists.
- Tests: `services/orion-hub/tests/test_curiosity_now.py`, `test_curiosity_event_wants.py`, `services/orion-durable-runs/tests/test_event_wants_menu.py`.
- Eval: `services/orion-hub/evals/run_curiosity_event_wants_replay.py`.

## Schema / bus / API changes

- Added: none (uses the existing `TemporalNowV1` / `ExperienceCitesV1` contracts and `WantV1`).
- Behavior changed: `curiosity_offer_decisions.constants` may now carry `now_shown_id` and `now_event_ids` (no reader of `constants` existed).
- New event-want family `hub.curiosity_finding`; new menu `event_ref` prefix `curiosity_run:`.
- No bus channel, no migration.

## Env/config changes

- Added keys (orion-hub): `HUB_CURIOSITY_NOW_ENABLED=true`, `HUB_CURIOSITY_EVENT_WANTS_ENABLED=true`, `HUB_CURIOSITY_EVENT_WANTS_TICK_SEC=300`, `HUB_CURIOSITY_EVENT_WANTS_LOOKBACK_SEC=43200`.
- `.env_example` updated: yes.
- local `.env` synced with `python3 scripts/sync_local_env_from_example.py`: yes (4 keys present in `services/orion-hub/.env` of the primary checkout).
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-hub: pytest tests -k "curiosity or reading_event or self_inquiry or kickoff or settings or env"
  -> 824 passed, 4 skipped (includes the new test_curiosity_now.py 13 and test_curiosity_event_wants.py 9)
services/orion-durable-runs: pytest tests -k "event_want or physiology or market or families or temporal_self_now or menu"
  -> 158 passed, 6 skipped; plus test_experience_now/test_run_bids/test_session_roundup_graph/test_run_value 37 passed
orion/curiosity orion/physiology orion/temporal_self orion/integration tests/test_check_integration_seam.py -> all passed
orion/prime/tests, orion/dig/tests/test_look.py, orion/body_atlas/tests/test_claims.py -> 318 passed
services/orion-mind tests/test_prime_loop.py tests/test_prime_regulate_loop.py -> 61 passed
scripts/check_integration_seam.py -> OK (16 organs; 44 families)
scripts/check_body_economy_contract.py -> OK (109 state-reading modules, all registered)
scripts/check_verb_physiology.py -> OK
```

## Evals run

```text
evals/run_curiosity_event_wants_replay.py (fixture, CI via test): 60 planted findings over 30 days ->
  no_history 20, below_band 36, would_post 4 (band share 0.100) -> PASS
evals/run_curiosity_event_wants_replay.py --live (read-only), 2026-11-16:
  14 register rows, 6 philosophical with novelty, all 6 no_history -> 0 would have posted.
  At ~6 philosophical findings a day the source reaches 20 past findings in about 3 days; after that the band
  posts about 1 finding in 10 (~0.6 a day).
263 reference runs: NOT replayed. They carry labels but no novelty (the register started 2026-11-15), and novelty
  needs the graph corpus plus an embedding per answer -- not cheap; skipped on purpose.
Ablation eval for cites (spec, weekly): not built in this patch.
```

## Docker/build/smoke checks

```text
No image built. Hub and durable-runs deploy only from main (CLAUDE.md section 8).
Live read of /temporal-self/now from this branch: not run (every call writes an experience_shown row in production).
```

## Review findings fixed

Code review ran in a subagent (orion-repo-agent) over the full diff. It confirmed the ordering (the decision row exists before the shown_id is written, both lines), never-raise paths, SQL param styles and flag parity.

- Finding (material): an explicit `since=<last run decided_at>` made every real curiosity read an "inspection" read, so the endpoint never moved curiosity's cursor, and a run whose read failed still advanced the window past unseen items.
  - Fix: no `since`; the endpoint's own cursor is "when curiosity last looked" (the brief's "last curiosity run", measured by the read itself). `LAST_RUN_SQL` removed.
  - Evidence: `test_kickoff_read_uses_the_server_cursor_and_persists_the_shown_row_on_the_decision_row` asserts the params are exactly `organ`, `run_ref`.
- Finding (material): the prompt's own example `[experience:<id>]` parsed as a cite, and an invented cite could push a real one out of the 3-cite cap.
  - Fix: ids never contain `<>`; the shown ids are persisted with the run and real cites go first, invented ones fill any room left (still counted by the endpoint).
  - Evidence: `test_the_prompts_own_placeholder_is_not_a_cite_and_shown_cites_go_first`.
- Finding: the kickoff now read had no overall time bound on its Postgres write.
  - Fix: `asyncio.wait_for` around the whole read, and a timeout on the write.
- Finding: the self-inquiry now read ran serially.
  - Fix: started as a task after the decision row, awaited just before the prompt.
- Finding: menu label read cancelled runs' finish rows.
  - Fix: `AND status = 'completed'` (as `orion/orion_day/gather.py`).
- Finding: the memory guard (copied from reading) dropped unanswered findings from retry.
  - Fix: keep only final decisions. Evidence: `test_the_memory_guard_keeps_retrying_unanswered_findings`.
- Finding: ordering was not pinned by tests; naming clash `_now` vs `now` locals; eval test env lacked PATH.
  - Fix: AST ordering test; renamed to `_now_reader`; `{**os.environ, ...}`.
- Not fixed (cosmetic): cite tasks are not drained at Hub shutdown (same as the register's tasks; they never raise).

## Metric quality gate (salience of a curiosity finding)

1. Provenance: `curiosity_finding_register.novelty`, written by `services/orion-hub/scripts/curiosity_finding_register.py` `finding_novelty` (1 - nearest cosine of the finding against everything Orion wrote before the run).
2. Independence: the only other curiosity magnitude in the system is `curiosity_moved_formed` (priors moved + formed, the now's salience for `curiosity_run`). Different source (graph diff vs text embedding); no shared computation.
3. Theory anchor: novelty-as-distance from prior writing is the standard novelty measure in information retrieval (nearest-neighbour novelty detection); the percentile is the existing event-want rule (`source_salience`).
4. Live data: 6 philosophical rows with novelty 0.107..0.162 (not flat, not null, not saturated). The rest point: the percentile can read 0 (lowest novelty yet); the band exists because a continuous magnitude rarely ties at the bottom.
5. Existing mechanism: reuses `source_salience`, `event_want`, the reading producer's shape.
6. Reversibility: a flag, one family entry, one menu prefix. No schema, no training default.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- Severity: medium. Concern: the register judge (prompt v2, Flash-Next) did not pass the dressing eval (#3435: 17.6% of operational findings flipped to philosophical when rewritten reflectively). A dressed plumbing finding can win a place in the band and become a Prime concern. Juniper chose to ship (2026-11-16) instead of tuning first. Mitigation: shadow only (nothing obeys the decision); the band admits ~1 in 10; #3435 stays open.
- Severity: low. Concern: no event want will post for about 3 days (live replay: 6 of 6 no_history). Expected, not a bug.
- Severity: low. Concern: when the decision row is missing (spend log off), the shown_id lives only in this process; a durable turn in another process then records no cites (logged `reason=no_shown_id`).
- UNVERIFIED (live): an `experience_use` row with `citing_organ = curiosity` (spec acceptance: within 48 h of deploy), and a `curiosity_event_want` log line with `outcome=`.
- Issues: references #3435 (not fixed). Opened: #3502 (verify-live).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3503

🤖 Generated with [Claude Code](https://claude.com/claude-code)
