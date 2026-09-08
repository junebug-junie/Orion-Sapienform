# Orion initiating with Claude: the dry run, and a budget that was already built

## Summary

- Gives `orion/dev_economics/rate_limit_events.py` its first consumer. It reads
  the one Claude budget signal that is real -- the constraint's own first-party
  message, including the reset time it carries -- and had **zero consumers**
  from the day it shipped. New channel `orion:substrate:claude_limit`.
- Adds `orion/autonomy/ask_claude_trigger.py`: a read-only decision answering
  "would Orion ask Claude something right now, and about what". **Nothing is
  sent and nothing is spent** -- no `claude` subprocess, no publish to
  `orion:room:claude:request`.
- Adds `scripts/report_ask_claude_dry_run.py` plus a 30-minute systemd timer,
  so the would-have-saids accumulate into a log a human can read over days.
- Records, in code, why the two obvious triggers are unusable -- both checked
  against live data, both refuted.
- Fixes a real env-sync blind spot found by this patch: `orion-cocreation-signals`
  was invisible to `sync_local_env_from_example.py` in both required ways.

## Outcome moved

Before: Claude only ever spoke *after* Orion spoke, inside a live Hub websocket
turn, which requires Juniper in the browser. **5 Claude turns in 6 days of
uptime.** There was no path from Orion's own state to a Claude conversation,
and no way to see how much Claude budget was left.

After: the budget is on the bus and readable, and the question "would Orion
have reached out, about what, and what stopped it" produces a dated,
inspectable answer every 30 minutes without spending anything.

## Current architecture

`orion:room:claude:request` -> `orion-room-companion` -> `orion:room:claude:utterance`
-> Hub relay -> live websockets + chat history. Two producers of the request,
both requiring an in-flight chat turn:

- `api_routes.py:2396` -- Juniper clicks "Ask Claude" (`invited_by=Juniper`)
- `websocket_handler.py:2266` -- fires after every Orion reply, `trigger="auto"`,
  `invited_by="Oríon"`. Live (`HUB_ROOM_CLAUDE_AUTO_RESPOND=true`).

The second is labelled as Orion inviting Claude, and `room_claude_relay.py`'s
own docstring concedes it "shipped without that enforcement landing first". It
is still reactive: Claude reacts to the room, and the room only exists while a
human is typing.

`orion/schemas/room_claude.py:69` states the v1 contract plainly: "`invited_by`
is always a human participant id. When Orion gains the ability to invite (v2),
that is the field that changes meaning -- and the field a budget policy would
key on." This patch does not change that field. It builds the budget policy
and the trigger that would key on it, in dry run.

## Architecture touched

- **New contract**: `orion:substrate:claude_limit` / `ClaudeLimitObservationV1`,
  produced by `orion-cocreation-signals`, consumed by `orion-hub` (declared;
  the Hub-side reader is the follow-up patch, see Concerns).
- **Producer placement is a safety decision, not convenience.**
  `orion-cocreation-signals` already mounts `~/.claude/projects` read-only for
  two existing producers. Hub must not: `services/orion-hub/docker-compose.yml`
  runs it as root with `/var/run/docker.sock`, and Orion's own FCC turns execute
  in that container with Bash. Mounting Juniper's whole transcript tree there
  would be a real expansion of what a Hub-resident agent can read.
- **No new service, no new taxonomy.** One producer loop in an existing service,
  one pure decision module, one report script.

## Files changed

- `orion/schemas/claude_limit.py`: new. Wire shape for `LimitObservation`,
  field-for-field so no consumer re-derives a property the producer already
  computed correctly.
- `orion/bus/channels.yaml`, `orion/schemas/registry.py`: register it.
- `services/orion-cocreation-signals/app/producers/claude_limit.py`: new. One
  event per configured trailing window per tick.
- `services/orion-cocreation-signals/app/settings.py`: three keys plus a shared
  `_parse_window_hours` used by both the validator and the property, so the two
  can never disagree about what a string means.
- `services/orion-cocreation-signals/app/main.py`, `.env_example`,
  `docker-compose.yml`: wiring.
- `orion/autonomy/ask_claude_trigger.py`: new. The decision.
- `scripts/report_ask_claude_dry_run.py`: new. What Juniper reads.
- `deploy/systemd/orion-ask-claude-dry-run.{service,timer}`: new. The cadence.
- `scripts/sync_local_env_from_example.py`: closes the blind spot below.
- `config/metrics/metric_definitions.lock.json`: re-locked for the new channel.
- Tests: `orion/autonomy/tests/test_ask_claude_trigger.py`,
  `services/orion-cocreation-signals/tests/test_claude_limit_producer.py`.

## The trigger, and why the two obvious candidates are dead

Both checked against live data 2026-09-08. Recorded in the module docstring so
this is not re-derived later.

**`scripts.tension_outreach_trigger.current_run()`** -- the trigger that already
makes Orion message Juniper. Every Borda winner over the trailing 7 days is
infrastructure:

| day | winner | ticks |
| --- | --- | --- |
| 09-07 | `node:athena` | 11,576 |
| 09-07 | `node:circe` | 5,941 |
| 09-07 | `node:substrate.bus_synaptic` | 1,344 |

Reusing it means Orion opens a conversation with Claude about host load. Right
mechanism, wrong subject.

**Curiosity candidates naming `sub-concept-seed-claude`** -- present in **1,382
of 1,385** candidate sets in 24h, every one an `ontology_sparse_region` in
`world_ontology`, and `signal_strength` had **one distinct value across all
1,382: exactly 1.0**. A pinned constant, not a signal. Keying on it means "fire
every tick", which carries no information about whether Orion wants anything.
This was initially misread as strong evidence *for* this option; the count is
what refuted it.

**What is left, and it is the right one.** `orion_worldview` is a FalkorDB graph
Orion writes itself, in-turn, with real Cypher; Hub holds `GRAPH.RO_QUERY` only.
Its priors carry real verdicts -- 4 `refuted`, 4 `supported`, 3 `revised`. Orion's
own kickoff prompt (`orion/curiosity/kickoff_prompt.py:595`) already names the
state this trigger looks for, in Orion's own words:

> "Inconclusive is a real answer: bump times_tested, leave confidence where it
> was... **Three of those and the claim is probably not answerable with what you
> can reach.**"

A claim Orion has tested repeatedly without settling *is* "not answerable with
what I can reach" -- the one condition where a second mind is the missing input
rather than a nicety, and it is Orion's assessment, not ours.

**A correction worth recording**: "zero priors are open" was wrong.
`CLOSED_STATUSES` is `(refuted, retired_unresolvable)` only -- `supported` and
`revised` stay live, and `worldview.py:80` documents that exact misreading as a
real accumulation outage on 2026-08-27. Seven priors are live.

## Selectivity, measured rather than asserted

`MIN_TIMES_TESTED = 3` and `MAX_SETTLED_CONFIDENCE = 0.7` are **knobs, not
findings**. They are not calibrated against anything. The report prints every
live prior with its numbers, selected or not, so the population is visible:

```
  tested   conf status     stuck  claim
      10   0.30 revised    YES    The LIVE prediction-error nodes (codebase=0.998, ...
       4   0.90 supported  -      The dominant_shift signal from turn_change_appraisal ...
       4   0.92 revised    -      The 9 node:substrate.* concepts are isolated in the ...
       2   0.95 supported  -      The stance crystallization gate does not algorithmically ...
       2   0.85 supported  -      An automated formation policy gate runs before manual ...
       2   0.95 supported  -      The auto-activate path for semantic/episode kinds ...
       1   0.80 revised    -      The degree-0 isolation of the live node:substrate.* ...
```

1 of 7. A test pins this exact population, and a second test asserts the count
is neither 0 nor 7 -- the two degenerate outcomes that would mean the knobs
carry no information. A retune has to face what it does to real data, not just
to synthetic cases built to make it pass.

## Schema / bus / API changes

- **Added**: `ClaudeLimitObservationV1`; channel `orion:substrate:claude_limit`
  (`substrate.claude_limit.v1`).
- **Removed / renamed**: none.
- **Behavior changed**: none. No existing producer or consumer path is touched.
- **Compatibility**: additive. The channel has one producer, off by default in
  code (`COCREATION_SIGNALS_CLAUDE_LIMIT_ENABLED` defaults `False`).
- **Deliberately not carried**: the individual `RateLimitEvent`s. They hold
  message text, and the dev-economics privacy boundary is token counts and
  timestamps only.
- **`observed` rides as its own field** rather than being inferred from
  `event_count == 0`, because that number is identical for "the pool is full"
  and "nobody looked". A budget that reads full during a mount outage is the
  defect CLAUDE.md §0A's prediction-error incidents exist to force out.

## Env/config changes

- **Added keys** (`services/orion-cocreation-signals/.env_example`):
  `COCREATION_SIGNALS_CLAUDE_LIMIT_ENABLED=true`,
  `COCREATION_SIGNALS_CLAUDE_LIMIT_POLL_INTERVAL_SEC=300.0`,
  `COCREATION_SIGNALS_CLAUDE_LIMIT_WINDOW_HOURS=5,168`,
  `CHANNEL_CLAUDE_LIMIT=orion:substrate:claude_limit`
- **Removed / renamed**: none.
- **local `.env` synced**: yes, all four keys, verified present.
- **Skipped keys requiring operator action**: none.

**The sync did not work until it was fixed, and that is the finding.**
`orion-cocreation-signals` was absent from `DEFAULT_SERVICES` **and** no
`SYNC_PREFIXES` entry matched any `COCREATION_SIGNALS_*` key. Both halves are
required -- the service list decides which `.env` files are visited, the prefix
decides which keys are considered once there -- so the first run considered
**zero** of the four new keys while printing a clean report. Identical in shape
to the `orion-whisper-tts` hole from PR #1956. Both halves added; the re-run
then added all four for real.

## Tests run

```text
$ pytest orion/autonomy/tests services/orion-cocreation-signals/tests -q
291 passed in 6.16s

$ pytest orion/autonomy/tests/test_ask_claude_trigger.py -q
17 passed

$ pytest services/orion-cocreation-signals/tests/test_claude_limit_producer.py -q
16 passed
```

CI static gates, all 12 from `.github/workflows/orion-static-gates.yml`:

```text
check_metric_lineage.py --gate                PASS
check_definition_drift.py --gate              PASS   (after --update re-lock)
check_inner_state_registry.py                 PASS
check_scripts_dir_no_stdlib_shadow.py         PASS
check_service_hostname_refs.py                PASS
check_compose_no_relative_mounts.py           PASS
check_compose_no_host_claude_json_mount.py    PASS
check_journal_dispatch_registry.py            PASS
check_daily_schedule_collisions.py            PASS
check_system_health_producers.py              PASS
check_control_surface_store_parity.py         PASS
check_async_routes_not_blocking.py            PASS
```

## Evals run

`orion/autonomy/evals/` already has a kill-criterion harness pattern
(`run_attention_bound_proposal_eval.py`: state a falsifiable criterion, read a
real window, report PASS / FAIL / insufficient-data). This feature gets one in
that shape: `orion/autonomy/evals/run_ask_claude_trigger_eval.py`.

The criterion is stated before the data is read. The prior side PASSes only if
the stuck-count is **neither always 0 nor always N** -- both extremes mean the
knobs carry no information, and "always N" is precisely what the pinned
`sub-concept-seed-claude` signal was rejected for. The budget side is
**reported, never failed on**: a week of `clear` is a fact about how contended
the pool is, not a defect in this trigger, and a week of `budget_unobserved`
is a broken mount rather than a dead trigger.

```text
$ python orion/autonomy/evals/run_ask_claude_trigger_eval.py
runs observed: 1 (need >= 48)
RESULT: insufficient data
```

That is the honest current verdict -- the log is one run old. **An eval that
can only ever say "insufficient data" is a permanent green light**, so all
three verdicts were exercised against synthetic logs and then pinned by tests:

```text
$ pytest orion/autonomy/tests/test_ask_claude_trigger_eval.py -q
8 passed

vary   RESULT: PASS     (stuck count moves between runs)
none   RESULT: FAIL     (knobs never select -- trigger cannot fire)
all    RESULT: FAIL     (knobs always select all -- no discrimination)
```

Two of those tests exist for failure modes that would otherwise blame the
trigger for someone else's outage: an unreachable worldview reports
insufficient-data rather than FAIL, and a week of budget refusals does not fail
the prior criterion.

`services/orion-cocreation-signals/` has no `evals/` directory and none was
created -- there is no harness there to extend. Noted in Follow-ups.

## Docker/build/smoke checks

```text
$ python -c "yaml.safe_load(open('services/orion-cocreation-signals/docker-compose.yml'))"
yaml ok; new env keys wired: ['CHANNEL_CLAUDE_LIMIT',
  'COCREATION_SIGNALS_CLAUDE_LIMIT_ENABLED',
  'COCREATION_SIGNALS_CLAUDE_LIMIT_POLL_INTERVAL_SEC',
  'COCREATION_SIGNALS_CLAUDE_LIMIT_WINDOW_HOURS']

$ python -c "yaml.safe_load(open('orion/bus/channels.yaml'))"
parsed ok, 289 channels, no duplicates, orion:substrate:claude_limit present
```

`docker compose config` could not be run from the worktree: the service's
compose file carries `env_file: - .env`, resolved relative to the compose file,
and `.env` is gitignored so it exists only in the primary checkout.
Pre-existing worktree limitation, not introduced here. `env template parity`
and `check_service_hostname_refs` both PASS via `safe_docker_build.sh` before
it reaches that point.

**Live end-to-end run of the report** (real FalkorDB, real transcripts):

```text
=== Claude budget (orion/dev_economics/rate_limit_events.py) ===
  state             : clear
  observed          : True
  times limit bound : 0        (35 over the trailing 168h window)
  staleness         : 0.79s (refuse above 900.0s)
  files scanned     : 38

=== Decision (DRY RUN -- nothing was sent, nothing was spent) ===
  would ask Claude : YES
  about prior      : atlas_prediction_error_territory
```

The exact `ExecStart` line was run as-is and appends valid JSON to
`~/.orion/ask-claude-dry-run.jsonl`.

## Review findings fixed

- Finding: *(pending -- `/code-review --effort high` was dispatched against
  `origin/main..feat/ask-claude-dry-run`; findings and fixes appended before
  merge.)*

Two defects were found and fixed during development, both in the new test file
rather than the code:

- Finding: `_obs(latest_activity=None)` silently substituted the default.
  - Fix: sentinel object instead of `x if x is not None else default`.
  - Evidence: `test_an_unobserved_window_publishes_unknown_rather_than_nothing`
    failed with `assert 5.0 is None`; passes after. The unobserved case is the
    one this suite exists to construct, so a helper that cannot express it was
    testing nothing.
- Finding: `RateLimitEvent` was constructed without its required `source`.
  - Fix: pass it.
  - Evidence: `TypeError` on collection; passes after.

## Restart required

Merge, then from the **primary checkout**:

```bash
cd /mnt/scripts/Orion-Sapienform
git pull --ff-only
python scripts/sync_local_env_from_example.py
scripts/safe_docker_build.sh orion-cocreation-signals up -d --build
docker compose -f services/orion-cocreation-signals/docker-compose.yml logs --tail=50 | grep claude_limit
```

Expect `cocreation_claude_limit_published window_hours=5.0 state=clear ...`
within 300s. `cocreation_claude_limit_disabled` means the env key did not land.

For the dry-run log (needs root, so **not run here** -- these are for Juniper):

```bash
sudo cp deploy/systemd/orion-ask-claude-dry-run.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now orion-ask-claude-dry-run.timer
systemctl list-timers orion-ask-claude-dry-run.timer
tail -f ~/.orion/ask-claude-dry-run.jsonl
```

The report also runs standalone with no install:

```bash
python3 scripts/report_ask_claude_dry_run.py
```

## Risks / concerns

- Severity: **medium**
  Concern: `channels.yaml` declares `orion-hub` a consumer of
  `orion:substrate:claude_limit`, but the Hub-side reader is not in this patch.
  The declaration is ahead of the code.
  Mitigation: stated here rather than hidden. The channel is off by default, so
  nothing publishes into a void unless an operator turns it on. The Hub panel is
  the natural next patch and it is what makes the budget visible in the UI
  rather than only in a script.

- Severity: **medium**
  Concern: the knobs are uncalibrated, by design. The trigger currently selects
  1 of 7 priors, but n=7 is a small population and one prior moving to
  `refuted` could take it to 0.
  Mitigation: that is exactly what the week of log output is for. The report
  prints the whole population every run, so a collapse to 0-of-N or N-of-N is
  visible in the log rather than needing to be looked for.

- Severity: **low**
  Concern: the budget's strict fail-closed reading (`unknown` refuses) is
  stricter than `rate_limit_events`' own docstring suggests for a human caller,
  where `unknown` usually means nobody has used Claude recently.
  Mitigation: deliberate and documented at the refusal site. An unread meter and
  a full tank must not authorise the same autonomous spend. If the week of logs
  shows `budget_unobserved` dominating, that is a real finding about the mount,
  not a reason to loosen the gate.

- Severity: **low**
  Concern: no cooldown, daily cap or quiet hours in the trigger module.
  Mitigation: intentional -- those are runtime state, not properties of the
  decision, and `endogenous_outreach` already owns a live version of all three
  (45-min cooldown, cap of 4/day, 23:00-08:00 quiet). Arming this trigger means
  reusing that stack, not growing a second one. Noted so the arming patch does
  not have to rediscover it.

- Severity: **low**
  Concern: spend enforcement remains advisory. Hub holds the docker socket, so
  a Hub-resident agent is root-equivalent on the host and no software cap is
  enforceable wherever it lives (settled with Juniper 2026-08-14).
  Mitigation: unchanged by this patch, which spends nothing. Restated so the
  arming patch does not mistake this budget for a ceiling.

## Follow-ups

1. Hub reader + budget panel for `orion:substrate:claude_limit` -- makes the
   remaining budget visible in the UI, which was half the original ask.
2. Read the week of `~/.orion/ask-claude-dry-run.jsonl` and decide whether the
   trigger discriminates before arming anything.
3. `orion-cocreation-signals` has no `evals/` directory. Not created here (no
   harness to extend); worth one if this producer family grows.
4. Re-run `run_ask_claude_trigger_eval.py` once the log passes 48 runs (about a
   day at the timer's cadence). Its verdict, not this PR, decides whether the
   trigger is worth arming.

## PR link

<filled in after push>
