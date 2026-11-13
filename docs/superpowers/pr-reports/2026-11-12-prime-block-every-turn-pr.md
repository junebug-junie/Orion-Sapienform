## Summary

Every unified turn now starts with a short block that shows what Prime holds right now. Prime is Orion's slow loop between turns. Chat, curiosity, consult and self-sense turns all get the block. This is Part 1 of the living Prime design (PR #3224).

- **The renderer.** A new pure renderer, `orion/prime/block.py::render_prime_block`, builds the block from Prime's own Redis state. It shows the concerns Prime holds (with their age and arc), the last thought (its reading, its time, and its move or "no move"), any open questions to curiosity and regulate signals with how they stand, and one line on the body. Every line ends with a ref. The block is at most 600 characters.
- **Stale state.** If Prime last ticked more than 10 minutes ago, the block shows only "Prime has not ticked since <time>". If Prime's state cannot be read, the block shows a labelled "absent" line. Neither case can break a turn.
- **Hub.** Hub reads Prime's state once per turn and sends the block on a new optional field, `HarnessRunRequestV1.prime_block`. The governor places the block right after the Situation block. Hub assembles its own copy of the prompt for the cockpit (`cockpit_turn_sighting` `motor_boot`), and that copy includes the block too.
- **Finalize judge.** The judge now sees the block as well. Without it, the judge would read a correct "I'm holding X [prime_concern:..]" as ungrounded.
- **Citation count.** After each reply, `check_prime_citations` counts the reply's Prime refs against the refs the block showed. It logs one line, `prime_block_citations`.
- **Kill switch.** The kill switch is `HUB_PRIME_BLOCK_ENABLED`. It ships ON.

## Outcome moved

Before this patch, no unified turn read Prime's state. A search of Hub, orion-thought, `orion/harness` and `orion/fcc` found only one reader of `orion:prime`: Hub's curiosity focus listener. Now every turn sees Prime's present, with cited ids. Orion can answer "what are you holding right now?" from the block.

## Current architecture

- Prime (`services/orion-mind/app/prime_loop.py`) ticks every 15 s. On every tick it saves `orion:prime:state` (`PrimeStateV1`) and `orion:prime:concerns` (`PrimeConcernsViewV1`, whose `as_of` is the tick time).
- Unified turns run through Hub (`orion/hub/turn_orchestrator.py::execute_unified_turn`), then the harness-governor runner (`orion/harness/runner.py`), then `compile_harness_prefix` (`orion/harness/prefix.py`). Curiosity and consult do not build their own prefixes: durable-runs `harness_step` sends them through the Hub turn RPC into the same function.
- The Situation block uses the same pattern this patch follows: Hub resolves it, sends it on the request, and the prefix renders it verbatim.

## Architecture touched

- `orion/prime/block.py` (new, pure except one fail-open Redis read).
- Hub turn orchestrator: reads Prime's state alongside the Situation build, sets the request field and the cockpit `motor_boot` prompt, and logs the citation check.
- `HarnessRunRequestV1`: one new optional field.
- Governor: the runner threads the block into the prompt, and `bus_listener` passes it to the finalize judge.
- Prefix: places the block and adds a one-line turn-rule brief.

## Files changed

- `orion/prime/block.py`: renderer, Redis parse/fetch, citation check.
- `orion/prime/tests/test_prime_block.py`: stale, cap, water-fill, citations (bracketed and unbracketed), absent, fetch failure, half-ref cut.
- `orion/prime/evals/run_prime_block_eval.py` + `fixtures/prime_block_live_2026-11-12.json`: eval on live state captured today; `--live` reads Redis and checks the concern ids.
- `orion/harness/prefix.py`, `orion/harness/runner.py`: place and thread the block.
- `orion/harness/tests/test_prime_block_prefix.py`: prefix order, unchanged-when-absent, runner prompt and motor_boot step, an old request without the field.
- `orion/hub/turn_orchestrator.py`: `_build_prime_block`, `_log_prime_citations`, wiring.
- `services/orion-hub/tests/test_turn_orchestrator_cockpit_hops.py`: end to end through `execute_unified_turn`, covering the request, the motor_boot hop, the logs, the kill switch and Redis being down.
- `orion/schemas/harness_finalize.py`: `HarnessRunRequestV1.prime_block`.
- `services/orion-harness-governor/app/bus_listener.py` + `tests/test_finalize_reply_budget.py`: the finalize judge gets the block ahead of the Situation notes.
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: flags and docs.
- `.github/workflows/orion-prime-tests.yml`: runs the new eval.

## Schema / bus / API changes

- Added: `HarnessRunRequestV1.prime_block: str | None = None`.
- Removed: none.
- Renamed: none.
- Behavior changed: the motor prompt and the finalize notes include the Prime block when the field is set.
- Compatibility notes: the model ignores unknown keys. An older governor drops the field, and an older Hub never sends it. No channel or registry change. **Deploy the governor first.** If Hub goes first, the sighting's `motor_boot` shows a block the motor did not get until the governor is rebuilt.

## Env/config changes

- Added keys: `HUB_PRIME_BLOCK_ENABLED=true`, `HUB_PRIME_BLOCK_STALE_MIN=10` (orion-hub).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes (orion-hub), with compose and `settings.py` defaults to match.
- Local `.env` synced: yes. `python scripts/sync_local_env_from_example.py orion-hub --all-keys` added both keys. The default run adds only "feature keys", so it did not add them.
- Skipped keys requiring operator action: none.

## Metric quality gate: `prime_block_citations`

1. **Provenance.** `check_prime_citations` (`orion/prime/block.py`) produces the line, and `_log_prime_citations` (`orion/hub/turn_orchestrator.py`) logs it once per turn. It reads the reply's `[kind:id]` tokens for Prime ref kinds and compares them with the refs this turn's block actually rendered.
2. **Independence.** This is a new counter on a new input. It does not use the arc gate (`orion/temporal_self/citations.py`, which judges arc refs on stance and letter text), and arc refs are excluded here.
3. **Theory anchor.** A ref that the block did not show cannot come from the block. This is the same rule the arc gate uses for unknown arc ids.
4. **Rest state.** `no_prime_claims` is the normal state: the reply cites nothing from Prime. A stale or absent block shows no refs, so any Prime ref cited then reads `UNVERIFIED`. It is a count only: it is not wired into any model or gate, and it never blocks or rewrites a reply. **UNVERIFIED live:** no deployed turn has produced the line yet.
5. **Existing mechanism.** The only citation gate on the reply side is the arc gate, and it does not run on motor replies.
6. **Reversibility.** It is one log line and one pure function, with no schema and no storage.

## Tests run

```text
PYTHONPATH=. pytest orion/prime orion/harness/tests/test_prime_block_prefix.py      -> 99 passed
PYTHONPATH=. pytest orion/harness/tests/test_stance_scope_hierarchy.py orion/harness/tests/test_harness_prefix.py orion/harness/tests/test_working_notes.py orion/hub/tests -> 121 passed
PYTHONPATH=. pytest orion/prime orion/harness/tests                                  -> 898 passed (before review fixes)
pytest services/orion-hub/tests (full suite)  -> 4688 passed, 147 skipped, 1 failed:
   test_fcc_model_labels_api.py::test_fcc_model_labels_disabled_by_default -- PermissionError reading
   /home/athena/.fcc/.env on this host; unrelated to this patch.
services/orion-harness-governor tests          -> 101 passed, 1 failed: test_fcc_warm_pool_wiring
   (same host /home/athena/.fcc/.env PermissionError; unrelated).
services/orion-durable-runs harness/consult/turn tests -> 40 passed, 4 skipped
orion/dig/tests/test_refs_contract.py          -> passed
Static: check_settings_defaults orion-hub --example-drift OK; check_env_key_single_source OK;
  check_env_template_parity PASS; metric lineage gate PASS; async-route gate OK; schema registry tests passed.
```

## Evals run

```text
python orion/prime/evals/run_prime_block_eval.py --live   (ORION_BUS_URL=bus Redis)
PASS quiet_grounded, quiet_bounded, busy_grounded, busy_bounded, busy_shows_concerns (3 of 3),
PASS stale_is_absent, missing_is_labelled, live_concern_ids_match_redis, live_bounded
python orion/prime/evals/run_prime_frame_eval.py  -> PASS
```

Block rendered from live Prime state, 2026-11-12 04:22Z:

```text
PRIME (my slow loop between turns), ticked under a minute ago:
- Holding: no concerns now. [prime_tick:04:22Z]
- Last thought (55 min ago; no move): The system is quiet with no strain and zero heat. ... [thought:#235]
- Asks: none open. [prime_tick:04:22Z]
- Body: nothing binding. [prime_tick:04:22Z]
```

## Docker/build/smoke checks

```text
Not run. No dependency or Dockerfile change. Live chat-turn sighting check is UNVERIFIED until deploy:
after deploy, one chat turn's cockpit_turn_sighting motor_boot prompt should contain "PRIME (" with
concern ids matching `redis-cli get orion:prime:state`.
```

## Review findings fixed

- Finding: `_fit` shared the room in line order, so a long first concern was cut while room went unused.
  - Fix: water-fill. The shortest bodies take only what they need, and the rest goes to the longer ones.
  - Evidence: `test_short_bodies_leave_their_room_to_the_long_first_concern`.
- Finding: the citation regex kept trailing punctuation, so "thought:#5." was rejected.
  - Fix: the id must end on a letter, digit, `#` or `_`.
  - Evidence: `test_citation_gate_reads_unbracketed_refs_before_punctuation`.
- Finding: the finalize judge never saw the block, so correct Prime claims looked ungrounded.
  - Fix: `bus_listener._finalize_notes` puts the block ahead of the Situation notes, inside the judge's 6,000-char clip.
  - Evidence: `test_finalize_gets_the_prime_block_the_motor_saw_ahead_of_the_situation`.
- Finding: refs invented on stale or absent turns were not counted.
  - Fix: the check always runs. With a stale or absent block it uses an empty shown set, and the log line carries the block status.
  - Evidence: `test_stale_block_shows_no_refs_so_any_cited_prime_ref_is_rejected`.
- Finding: the hard cut could leave half a ref in the prompt.
  - Fix: it cuts back to a line boundary.
  - Evidence: `test_hard_cut_never_leaves_a_half_ref`.
- Finding: the Prime read ran after the Situation build, adding up to 1 s.
  - Fix: it now runs as a task alongside that build.
- Finding: the docstring gave the wrong drop order, the deploy order was documented only in a field description, and the thought-id match had no explanation.
  - Fix: corrected the docstring, added the deploy order to the README and this report, and added a comment that points at `prime_loop` `_conclude`.

## Restart required

Deploy the governor first, then Hub. From the clean primary `main` checkout after merge:

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- **Severity: low.**
  - Concern: the last thought's id is not stored in Prime's state. When no move from that thought is open, the ref is Prime's own thought count (`thought:#235`), not a thought id that dig can open. `prime_concern:` and `thought:` are not `orion/refs.py` kinds, so the gate can check them, but dig cannot open them.
  - Mitigation: Part 2 (the parallel agent, which owns the Prime state models) can add `last_thought_id`. The renderer would then use it.
- **Severity: low.**
  - Concern: Prime's concerns now steer every turn (the design's named failure mode).
  - Mitigation: the stale rule, refs on every line, the 600-char cap, and the kill switch.
- **Severity: low.**
  - Concern: the block uses some prompt room every turn (≤ 600 chars).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3229

🤖 Generated with [Claude Code](https://claude.com/claude-code)
