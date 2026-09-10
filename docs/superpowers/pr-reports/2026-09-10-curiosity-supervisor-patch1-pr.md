# Curiosity supervisor, Patch 1: a read-only reader that grades Orion's own investigation notes

## Summary

Orion investigates things, writes notes as it goes (`Hop` nodes in its own
`orion_worldview` graph), and until now nothing ever read them back to check
whether a run was actually getting anywhere. This ships the first, smallest
slice of `docs/superpowers/specs/2026-09-09-curiosity-supervisor-design.md`'s
own "Recommended next patch": a program that reads every note Orion has
written, asks an LLM to say which claim each one was testing and whether it
moved that claim, and reports — per claim — whether the last few tests in a
row changed anything or just went in circles. It does not touch Orion's
graph, does not publish anything Orion or Hub would react to, and does not
change what happens next. It only produces a report a person can read.

- New module (`orion/curiosity/supervisor.py`) that turns a run's worth of
  investigation notes into a structured judgment per note, via one LLM call
  per run.
- Two new bulk reads on Orion's own graph (`orion/curiosity/worldview.py`)
  that didn't exist before — everything else in that file reads one run at a
  time or "what's still open"; this patch needed "everything, ever."
  A dial (`is_circling`) that says whether the last few tests of one belief
  moved anything, computed from what the LLM read — three time in a row where
  nothing moved is stuck, not merely unlucky once.
- A command-line report (`scripts/report_curiosity_supervisor_readings.py`)
  a person runs by hand; it writes its findings to two files under `/tmp` and
  prints a table.
- Two live services (`orion-cortex-orch`, `orion-cortex-exec`) needed a
  rebuild to recognize the new kind of request this makes — see "Env/config
  changes" and "Docker/build/smoke checks" below for exactly what changed and
  why.

## Outcome moved

Before this patch, the only thing that ever read a `Hop` note back was Orion
itself, one run later, out of its own kickoff prompt. Nothing external ever
asked "did this actually work?" Now there's a report that answers that
question for every investigation note Orion has ever written, and flags,
per open question, whether Orion is closing in on an answer or running in
place. Run live against the real graph tonight: 68 notes, 40 investigation
runs, 67 graded. One of the beliefs it flagged as "running in place"
(`atlas_prediction_error_territory`) is the exact one the design doc named
as the known example of Orion going in circles — the report caught it.

## Current architecture

Before this patch: `orion/curiosity/worldview.py` held every read Hub does
against Orion's own graph, and every one of them served the NEXT kickoff
prompt — "what's still open," "what did I just settle," "what happened last
run." Nothing read the full history of what Orion had tried at any point in
the past. `orion/schemas/curiosity_supervisor.py` and
`orion/curiosity/supervisor.py` did not exist.

## Architecture touched

- `orion/curiosity/worldview.py` — added two read-only queries and their
  Python readers (`read_all_hops`, `read_all_priors`), plus a `HopRecord`
  dataclass, following the file's existing conventions exactly (tolerant row
  parsing, "unreadable vs. empty" distinction, no query parameters beyond a
  validated `run_id`).
- `orion/curiosity/supervisor.py` (new) — prompt construction, the LLM call
  (one per investigation run, via the same `cortex.orch.request` bus RPC
  shape `orion/memory_graph/suggest_runner.py` already uses), response
  parsing (tolerant — a malformed reading is dropped and logged, not
  raised), and `is_circling` (a pure function over already-produced
  readings).
- `orion/schemas/curiosity_supervisor.py` (new) — the `HopReadingV1` /
  `HopReadingBatchV1` contract from the spec, verbatim.
- `orion/cognition/verbs/curiosity_hop_reading.yaml` +
  `orion/cognition/prompts/curiosity_hop_reading_prompt.j2` (new) — this
  patch's own verb, so `orion-cortex-orch`'s brain-mode request gate accepts
  the call. Minimal, single-step, no framing of its own (this module's own
  prompt is already complete) — same shape as `memory_graph_suggest`'s verb.
- `scripts/report_curiosity_supervisor_readings.py` (new) — the CLI a person
  runs by hand.
- `tests/test_curiosity_supervisor.py` (new) — 162 tests total in the two
  touched test files, all against fakes, no live dependency.

## Files changed

- `orion/curiosity/worldview.py`: two new bulk reads (`read_all_hops`,
  `read_all_priors`) and their Cypher, plus `HopRecord`.
- `orion/schemas/curiosity_supervisor.py`: new — `HopReadingV1` /
  `HopReadingBatchV1`.
- `orion/curiosity/supervisor.py`: new — the reading pipeline and
  `is_circling`.
- `orion/cognition/verbs/curiosity_hop_reading.yaml`,
  `orion/cognition/prompts/curiosity_hop_reading_prompt.j2`: new — verb
  registration so the live request gate accepts this call.
- `scripts/report_curiosity_supervisor_readings.py`: new — CLI entry point.
- `tests/test_curiosity_supervisor.py`: new — unit coverage.
- `docs/superpowers/specs/2026-09-09-curiosity-supervisor-design.md`: status
  line updated, "Patch 1 results" section appended with the live-run
  findings.

## Schema / bus / API changes

- Added: `HopReadingV1` / `HopReadingBatchV1` (`orion/schemas/curiosity_
  supervisor.py`) — a Pydantic contract, **not registered** in
  `orion/schemas/registry.py` or `orion/bus/channels.yaml`. Nothing
  publishes it on the bus in this patch; registration is the next patch's
  job once/if a channel is armed.
- Added: verb `curiosity_hop_reading` (`orion/cognition/verbs/
  curiosity_hop_reading.yaml`) — discovered automatically by `orion/
  cognition/verb_activation.py` (its `active.yaml` default allow-list is
  empty, meaning "every discovered verb not explicitly denied" — no
  additional registration step needed beyond adding the file).
- Removed: none.
- Renamed: none.
- Behavior changed: none to any existing verb, schema, or channel.
- Compatibility notes: fully additive. No existing consumer, producer,
  schema, or channel is touched.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no (no new env keys anywhere in this patch).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`:
  not applicable — no `.env_example` changed.
- skipped keys requiring operator action: none.

## Tests run

```text
cd /mnt/scripts/Orion-Sapienform-curiosity-supervisor-design && \
/mnt/scripts/Orion-Sapienform/.venv/bin/python3 -m pytest \
  tests/test_curiosity_supervisor.py tests/test_curiosity_worldview.py -q
162 passed
```

## Evals run

No eval harness exists for `orion/curiosity/`. This patch's own live
verification (below) serves as the eval the spec's own acceptance checks
call for — run once against real data, findings recorded in the spec doc's
"Patch 1 results" section, not just asserted.

## Docker/build/smoke checks

Runtime behavior changed for two live services — a new verb needed to be
recognized by both, and neither bind-mounts `orion/` (build-baked only), so
a code/config-only addition still needed an image rebuild to take effect.
Done with Juniper's explicit sign-off before each step, given the live blast
radius (both services carry Orion's actual live chat/background/spark
traffic, not just this patch's own calls):

```bash
cp .env-and-two-service-envs-from-main-checkout   # worktrees don't inherit .env
./scripts/safe_docker_build.sh orion-cortex-orch up -d --build
./scripts/safe_docker_build.sh orion-cortex-exec up -d --build   # 4 containers: base, chat, background, spark
```

Both came back healthy; confirmed via a live RPC round-trip through each
before proceeding. `python3 scripts/report_curiosity_supervisor_readings.py`
(no flags beyond `ORION_BUS_URL`) run to completion end-to-end against the
real graph and the real bus — see the spec doc's "Patch 1 results" for
output.

## Review findings fixed

- Finding: `hop_run_id` in a model's structured-output response was
  hallucinated (`"n=1"`) rather than left absent, even though nothing in the
  prompt asks for it — because the JSON schema marks it required.
  - Fix: `parse_reading_batch` now always overwrites `hop_run_id` with the
    caller's own known `run_id`, never trusts the model's value.
  - Evidence: caught live during Patch 1's own verification run; regression
    test added (`test_parse_reading_batch_overrides_model_hallucinated_hop_
    run_id`).
- Finding: the top-level cortex RPC response is a `CortexClientResult`
  (`ok`/`mode`/`verb`/`final_text`/`steps`/...), not the structured-output
  dict itself — an earlier version of this code tried to read `readings`
  directly off it and silently got nothing every time.
  - Fix: `_extract_cortex_result_text` pulls the model's JSON out of
    `final_text` (falling back to `text`/`content`/`steps`), matching the
    field-priority convention `orion/memory_graph/cortex_suggest_extract.py`
    already uses for the same problem one domain over.
  - Evidence: live-verified; the same fix turned a 0-reading run into a
    correct 1-reading one, reproduced with a debug script before landing the
    fix.
- Finding: the same request could fail `orion-cortex-orch`'s downstream
  verb-activation gate and succeed on a bare retry seconds later, for
  reasons internal to that service this patch does not touch.
  - Fix: `generate_readings_for_run` retries up to 3 times with a short
    backoff on ANY RPC/decode failure or a `CortexClientResult` with
    `ok: false`.
  - Evidence: live-verified; regression test
    (`test_generate_readings_for_run_retries_a_not_ok_result`) fakes the
    exact observed failure/success sequence.
- Finding: `is_circling` reads a prior's last 3 readings as "most recent,"
  but nothing sorted them — an LLM's multi-hop batch response isn't
  guaranteed to come back in `hop_n` order, so a reordered response could
  silently flip which readings count as "recent" and therefore flip the
  verdict for no chronological reason.
  - Fix: `parse_reading_batch` now sorts its output by `hop_n` before
    returning.
  - Evidence: `test_parse_reading_batch_sorts_by_hop_n_regardless_of_model_order`.
- Finding: `build_run_order` repurposes `RECENT_RUNS_CYPHER` (capped at 200
  rows, sized for "show Orion its recent thread") to order ALL of history,
  with no warning if that bound is ever hit — inconsistent with the two
  sibling full-scan queries this patch added, which both warn on truncation.
  - Fix: named the bound (`RECENT_RUNS_LIMIT`), added the same warning.
  - Evidence: `test_build_run_order_warns_when_the_recent_runs_bound_is_hit`.
- Finding: the CLI report's headline number ("Total hops read") was actually
  counting successfully-produced readings, not hops read from the graph —
  exactly the two numbers that diverge when a batch's JSON response is
  dropped for being malformed, which this patch's own live run hit 3 times.
  - Fix: reports both numbers separately (`Hops read from the graph: N.
    Readings produced: M`).
  - Evidence: live-verified; manual run against the current graph.
- Finding: unreachable trailing `raise` after the retry loop (dead code,
  misleading — implied a wrapped `RuntimeError` a caller would never
  actually see) and an unused `dataclasses.asdict` import.
  - Fix: added an explicit `max_attempts < 1` guard (making the trailing
    raise a real, if rare, defensive path instead of silently dead code) and
    removed the unused import.
  - Evidence: `test_generate_readings_for_run_rejects_max_attempts_below_one`.
- Finding (noted, not changed): `_extract_cortex_result_text` is a thinner,
  independent reimplementation of text-extraction logic that already exists,
  more thoroughly, in `orion/memory_graph/cortex_suggest_extract.py`.
  - Disposition: kept separate — the existing version is memory-graph-coupled
    (checks a domain-specific `"draft"` key) and its real search logic is
    private to that module, so importing it wholesale for a narrower need
    was judged the wrong trade. Documented in code as a conscious, not
    accidental, duplication.

## Restart required

Already done as part of this patch's own live verification (see
"Docker/build/smoke checks"). No further restart needed for this patch to
work; the two rebuilt services are already running the new image.

## Risks / concerns

- Severity: low. Concern: 3 of 40 live runs returned truncated JSON from the
  LLM route for reasons not yet diagnosed (ruled out `max_tokens` as the
  cause, did not find the real one). Mitigation: degrades safely today
  (dropped and logged, rest of the sweep unaffected); flagged in the spec
  doc's "Patch 1 results" for whoever picks up the next patch, with the
  ruled-out hypothesis recorded so it isn't re-tried.
- Severity: low. Concern: `Hop` still carries no timestamp, so run ordering
  (and therefore "most recent" in `is_circling`) is a best-effort proxy via
  `TurnOutcome.written_at`, not a real clock. Mitigation: documented in code
  and in the spec doc; missing question 2 (mid-turn vs. between-runs seam)
  stays explicitly open rather than answered on shaky evidence.
- Severity: low. Concern: the new verb's rebuild touched two live,
  actively-serving services. Mitigation: purely additive (no existing code
  path changed), done only after explicit sign-off, health-checked
  immediately after each rebuild before proceeding.

## PR link

<filled in after push>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
