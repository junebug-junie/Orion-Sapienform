# One event makes Orion look up, then it fades

## Summary

Some of Orion's body signals are written once per event and then carried forward unchanged until the next event. Codebase change is one of them. After the world-first attention patch (#2583), one such event could hold attention for as long as the value sat there. Live on 2026-10-10: one codebase event (the deploy's git pull, 0.988 at 05:51:03) won every field frame for 18 minutes, until the next poll wrote 0. This patch makes those signals behave like an orienting response. They compete at full strength for the first minute, fade over the next four, and stop competing at five minutes even though the carried value has not changed. A new event re-arms them.

- **Which signals fade is read from the semantic layer, not a list.** A node fades when its glossary semantics say it is written per event: `sparsity: event_gated`, or `sparsity: designed_sparse` with `absent_means` starting "only written". The second form is where `orion/metrics/semantics.py` says the write cadence goes when designed_sparse takes precedence.
- **Event age** is how long ago the stored reading was written (`unusualness.age_sec`). It is not the frame time.
- **Inside the window** the source is judged as before. After a 60 s grace, its rank and salience fade linearly to 0 at 300 s. At 300 s it is ineligible, with the reason "event Ns old: past the 300s orienting window (value carried forward since)".
- **Unchanged:** continuous level signals (biometrics, bus_synaptic), camera surprise, the cabinet trigger and the chat-rate world source.
- **Kill switch:** `ATTENTION_EVENT_DECAY_ENABLED`, which ships on. With `ATTENTION_WORLD_FIRST_ENABLED=false` the ranking is still exactly the old one; the golden test passes untouched.

## Outcome moved

Read-only replay (`services/orion-attention-runtime/evals/replay_world_first.py`). The 3-day run uses a 1-minute grid from 2026-10-07 07:00 to 2026-10-10 07:00 UTC (4,321 ticks). The 05:45–06:15 window uses a 10 s grid.

| | World-first, no decay | World-first + event decay |
|---|---|---|
| **05:51–06:09 codebase hold** (longest unbroken win) | **1,100 s** (61% of the window) | **300 s**, then bus_synaptic / biometrics / no winner |
| Codebase longest hold, 3 days | 1,080 s | 300 s |
| Body-chat (`node:substrate.chat`) longest hold | 1,800 s | 300 s |
| Route longest hold | 1,800 s | 420 s (re-armed by back-to-back route events) |
| Codebase / route / chat share of all ticks | 5.3% / 7.3% / 2.6% | 1.5% / 3.8% / 1.1% |
| No-winner share | 46.0% | **51.9%** |
| Calm body winners (raw < 0.05) | 0.56% | 0.19% |
| **Execution spikes ≥ 0.9 won** | 60 / 86 | **65 / 86** |
| **Biometrics surges ≥ 0.25 won** | 106 / 110 | **106 / 110** |
| **Synthetic 5 h execution storm** (1.0 every 135 s) | eligible 60/60, won 85% | eligible **60/60**, won **87%** |

Real body spikes still win, and slightly more often: stale carried events no longer take the near-ties. The continuous storm does not decay away, because every write re-arms it.

The old stored frames for the same window (123,678 frames) had 35.3% calm winners.

## Semantic-layer trace (standing rule)

Sources: `config/field/field_channel_glossary.v1.yaml` (node-qualified `prediction_error` entries, PR #2579 `semantics:`), and `check_metric_lineage.py --metric node:substrate.<x>.prediction_error`, run for all nine nodes. The lineage cards repeat the same fields.

| Node | value_kind | sparsity | absent_means (first words) | Fades? |
|---|---|---|---|---|
| execution | level | designed_sparse | "only written when a batch carried execution events; … carried forward" | **yes** |
| chat (body PE) | level | designed_sparse | "only written when chat turns were touched; carried forward" | **yes** |
| codebase | level | designed_sparse | "only written on a codebase delta event; carried forward" | **yes** |
| route | count | designed_sparse | "only written when routing runs were touched; carried forward" | **yes** |
| harness_closure | placeholder | event_gated | "the node keeps its last 0.65" | selected, but a placeholder is never eligible anyway |
| cabinet | trigger | designed_sparse | "when the thermal reading is 'unknown' … the tick writes nothing" (rewritten every 30 s otherwise) | no |
| perception | level | designed_sparse | "written as 0.0 every tick rather than dropped" | no |
| biometrics | level | per_tick | "carried forward in node_vectors between biometrics samples" | no |
| bus_synaptic | level | per_tick | "carried forward in node_vectors …" | no |

- **"Carried forward" alone is not the test.** The per-tick biometrics and bus_synaptic say it too. A test pins this.
- **World sources:**
  - `world:chat` is Juniper's turn count over the last 15 minutes. It is recomputed every tick from turn times, so its value is current, not carried, and it does not fade. One message keeps it eligible for up to 15 minutes by its own definition. Its longest hold in the replay is 480 s.
  - Camera surprise (`perception`) is written every tick and does not fade.
- **`tests/test_attention_event_decay.py::test_event_written_set_comes_from_glossary_semantics`** pins this exact classification against the real glossary. A new node-qualified PE entry fails the test until someone classifies it on purpose.

## Horizon derivation (300 s window, 60 s grace)

Data: 7 days of `substrate_node_prediction_error_history`. An "isolated onset" is a nonzero reading in the node's own top decile, with no onset on the same node in the previous 30 minutes. There are 225 such onsets across execution, chat, codebase and route.

**How long one event's consequence lasts in the body.** The change in mean percentile, versus the 30 minutes before the event, ± 95% CI:

| after onset | biometrics | bus_synaptic |
|---|---|---|
| 0–60 s | +0.087 ± 0.034 | +0.067 ± 0.038 |
| 60–120 s | +0.067 ± 0.033 | +0.029 ± 0.037 |
| 120–180 s | +0.040 ± 0.032 | −0.004 |
| 180–240 s | +0.054 ± 0.032 | −0.040 |
| 240–300 s | +0.029 ± 0.037 (not distinguishable from 0) | −0.020 |
| 300–420 s | −0.006 | −0.012 |

**How long it lasts in the source's own readings.** Readings return to their pre-event level within 60–180 s:
- execution: 0.823 → 0.203 (0.210 before);
- chat: 0.944 → 0.046;
- route: 0.880 → 0.161 by 180–300 s.

So one event's consequence is gone by about 300 s. That is the window. It is tighter than, and inside, the existing 1800 s PE staleness horizon (`PressureConfig().prediction_error_decay_horizon_seconds`). That horizon stays the outer bound for everything. It does not fit as the event window: it is 6× longer than any measured consequence, and it would have let the codebase event hold for the full 18 minutes.

**A storm still re-arms well inside the window.** After an elevated execution reading, the next write lands at p50 68 s and p90 160 s.

**Grace (60 s): why the fade does not start at 0.**
- A per-tick level rival's reading is itself up to one write interval old, and it is never faded. Over the same 7 days the p90 gap between writes is: biometrics 62.6 s, bus_synaptic 60.7 s, perception 60.5 s, cabinet 60.6 s.
- With no grace, a 30 s-old execution spike lost near-ties it used to win: 47 / 86 spikes won, against 60 / 86 without decay, and the storm won 77%. That is the real-alarm-suppressed failure.
- The grace comes from the rivals' write cadence, not from the replay score.

**Other grace values, measured for disclosure only.** Neither was picked:
- grace 120 s: execution 65/86, storm 97%;
- full-strength plateau to 300 s (no fade): execution 62/86, storm 93%.

**Choice of mechanism.** I used onset recency (a hard window) plus a rank fade, and did not decay the percentile toward rest. Against a mostly-zero week, a decayed percentile crosses the 0.9 band cut at a time set by the onset percentile and the band cut. That turns the window into a hidden function of two other knobs. The hard window keeps "was the event unusual" (band, unchanged) separate from "is it still news" (age).

**These are knobs, not findings.** The window and grace are measured on 7 days of data and should be re-graded on live frames after deploy.

## Current architecture

- World-first (#2583) ranks every candidate by its own 7-day percentile.
- Body nodes are eligible at band high or unusual while the reading is less than 1800 s old (`INTERNAL_MAX_AGE_SEC`).
- Event-written nodes keep their last value between events, so a single top-percentile event stays eligible until the next write or 30 minutes, whichever comes first.

## Architecture touched / files changed

- `orion/attention/world_first.py`:
  - `prediction_error_is_event_written`, the semantic predicate;
  - `node_prediction_error_event_written`, a glossary read where only successes are cached;
  - `EVENT_ORIENTING_WINDOW_SEC` / `EVENT_FADE_GRACE_SEC` with their derivations;
  - `event_fade`;
  - the age clause in `judge_candidate`;
  - `CandidateVerdict.event_decay` / `event_age_sec` / `salience`, plus the trace fields;
  - the `node_candidate(event_decay=)` parameter.
- `orion/schemas/attention_candidate.py`: optional `event_window_sec` (default None). The model is registered and not on the bus; it only travels in frame traces.
- `orion/attention/field_attention/selectors.py`: target salience is the faded strength.
- `orion/substrate/attention_broadcast.py`: `event_decay` threaded through `build_substrate_attention_frame` → `world_first_signals`. The loop score falls back to the faded salience.
- `services/orion-attention-runtime/app/{settings,worker}.py` and `services/orion-substrate-runtime/app/{settings,worker}.py`: the flag. Each service's `.env_example` and `docker-compose.yml` exposes it.
- `services/orion-attention-runtime/evals/replay_world_first.py`: adds `--no-event-decay`, `--start/--end`, and a longest-hold metric per source.
- Tests:
  - `tests/test_attention_event_decay.py` (new, 20 tests);
  - `services/orion-attention-runtime/tests/test_world_first_worker.py` (+1 flag-wiring test).

Temporal Self (#2369) habituation code is not touched.

## Schema / bus / API changes

- **Added:**
  - `AttentionCandidateV1.event_window_sec: float | None`, optional, default None.
  - Trace keys `event_window_sec`, `event_age_sec`, `event_decay` in `frame.debug["world_first"].candidates[]`.
- **Removed / renamed:** none.
- **Behavior changed:** with world-first on, an event-written node is ineligible once its event is ≥ 300 s old, and its rank and salience fade from 60 s on.
- **Compatibility:**
  - No bus payload changed.
  - The candidate is never persisted standalone, so deploy order is free.
  - The field target `salience_score` and the broadcast `loop.salience` for an event-written node now report the faded strength.

## Env/config changes

- **Added keys:** `ATTENTION_EVENT_DECAY_ENABLED=true` in orion-attention-runtime and orion-substrate-runtime, across `.env_example`, settings and compose.
- **Removed / renamed keys:** none.
- **`.env_example` updated:** yes, both services.
- **Local `.env` synced:** yes. `python scripts/sync_local_env_from_example.py --all-keys orion-attention-runtime orion-substrate-runtime` was run from the worktree and wrote the primary-checkout `.env` files (`+ATTENTION_EVENT_DECAY_ENABLED='true'` in both).
- **Skipped keys:** none from this patch. orion-attention-runtime's `.env` is still missing `ORION_BUS_URL`, a protected NEVER_SYNC key that predates this patch.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest -q -p no:cacheprovider -W ignore ...
tests/test_attention_*.py tests/test_voluntary_attention_wiring.py orion/substrate/tests/test_attention_broadcast*.py
  services/orion-attention-runtime/evals orion/reverie/tests/test_proposal_world_winner.py      PASS (incl. flag-off golden parity, unchanged)
services/orion-attention-runtime/tests + evals + tests/test_attention_event_decay.py          77 passed, 5 skipped
orion/substrate/tests/test_attention_self_model.py test_endogenous_curiosity.py, tests/test_proposal_*.py   161 passed
services/orion-substrate-runtime/tests      17 failed + 1 collection error: the identical pre-existing set on origin/main (cursor/quarantine/reducer/self-model-tick)
Static gates: check_definition_drift --gate PASS (0 changed, no re-lock), check_metric_lineage --gate PASS,
  --prompt-semantics PASS, check_inner_state_registry OK, check_env_template_parity PASS,
  check_service_env_compose_parity: attention-runtime OK; substrate-runtime same 17 pre-existing gaps
```

**Mutation check: 15 of 15 killed.** Each mutant was applied to the real file, its tests were run, and the file was restored.

| Group | Mutants killed |
|---|---|
| Window | age window ignored; `>` instead of `>=`; age taken from the frame time instead of the event |
| Fade | rank not faded; no grace; fade never reaches 0 |
| Semantic predicate | "carried forward" text match; any designed_sparse; event_gated ignored |
| Glossary | a glossary failure gets cached |
| Kill switch | `node_candidate` ignores it; the broadcast drops it; the worker ignores the setting |
| Wiring | field salience unfaded; decay applied to level signals |

The first run left two mutants alive: the "carried forward" text match, and unfaded field salience. Two tests were added to kill them.

## Evals run

```text
replay_world_first.py --days 3 --end 2026-10-10T07:00Z --step-sec 60                 (decay)    -> table above
replay_world_first.py --days 3 --end 2026-10-10T07:00Z --step-sec 60 --no-event-decay            -> table above
replay_world_first.py --start 2026-10-10T05:45Z --end 2026-10-10T06:15Z --step-sec 10 [--no-event-decay]  -> codebase hold 1100 s -> 300 s
grace variants (spike + storm checks): 0 s 47/86, 77%; 60 s 65/86, 87%; 120 s 65/86, 97%; plateau 62/86, 93%
pytest services/orion-attention-runtime/evals -> pass
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-attention-runtime build -> Built
scripts/safe_docker_build.sh orion-substrate-runtime build -> Built
(temporary .env symlinks to the primary checkout, removed right after; nothing deployed)
In-image: both images resolve window 300.0, grace 60.0, event-written = [chat, codebase, execution, harness_closure, route]
```

The deployed path has not been checked (**UNVERIFIED**). After deploy, look for:
- `frame_json->'debug'->'world_first'->'candidates'` entries with a non-null `event_window_sec` and a reason containing "orienting window";
- no field frame won by `node:substrate.codebase` more than 300 s after its `observed_at`.

## Review findings fixed

REVIEW_PLACEHOLDER

## Proposal-mode items (CLAUDE.md §0A)

Juniper approved this on 2026-10-10: "let event type signals decay".

- **Capability change:** event-written body signals can hold attention for at most 5 minutes per event.
- **Data touched:** read-only, as in #2583. Writes are unchanged; the trace gains three keys.
- **Privacy:** unchanged.
- **Trace:** every candidate carries `event_window_sec`, `event_age_sec` and `event_decay`, and its reason names the window.
- **Dangerous failure mode:** a real, sustained execution problem that writes less often than every 300 s would flicker in and out of eligibility. Execution's p99 gap between writes over 3 days is 729 s (measured), so this can happen. Mitigation: each new write re-arms it, and the 7-day band is unchanged.
- **Disable:** set `ATTENTION_EVENT_DECAY_ENABLED=false` in either service and restart. That leaves world-first on.

## Restart required

No forbid-model consumer is involved, so deploy order is free. Run from the PRIMARY checkout on main, after merge:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && for s in orion-substrate-runtime orion-attention-runtime; do docker compose --env-file .env --env-file services/$s/.env -f services/$s/docker-compose.yml up -d --build; done
```

## Risks / concerns

- **Severity: medium. Continuous level signals still hold for a long time.**
  - Longest holds in the replay: bus_synaptic 1,860 s, biometrics 1,200 s.
  - These are per-tick readings that really are in their top decile, so they are out of scope here: the request was carried-forward single events.
  - If they need limiting, habituation is the tool, and that belongs to Temporal Self (#2369). Not touched here.
- **Severity: low. The window and grace are knobs, measured on 7 days of history.** Next step: re-grade them on 48 h of live frames.
- **Severity: low. The predicate reads prose.** It checks `absent_means` for "only written", which is where `orion/metrics/semantics.py` puts the write cadence. A structured `write_cadence` field would be cleaner, but it is a definition change across the lock. The pinned classification test fails if any node-qualified PE entry is added or reclassified.
- **Severity: low. Chat-rate holds up to 15 minutes per message.** This is by its own windowed definition, and it is not carried forward.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
