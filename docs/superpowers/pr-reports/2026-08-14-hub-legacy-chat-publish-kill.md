# Kill the Hub's raw-dict chat-history publish (the `legacy.message` source)

## Summary

- Deleted the third, raw-dict publish to `chat_history_channel` in **both**
  Hub transports — `orion/hub/turn_orchestrator.py` (WS/unified turn) and
  `services/orion-hub/scripts/api_routes.py` (HTTP chat). Published with no
  `kind`, so `orion/core/bus/codec.py:72` stamped each one `legacy.message`,
  which matches no sql-writer route and fell through to `bus_fallback_log`.
- Added `services/orion-hub/tests/test_chat_history_no_raw_publish.py` — 6
  tests: 3 behavioral over the real WS publish path, 3 for a static AST gate
  that fails if a dict literal is ever published to a chat-history channel from
  any Hub source. Mutation-verified.
- **Corrected a false claim in two already-merged PR reports.** Both described
  these fallback rows as lost cognition. They were not: every one is a duplicate
  of a turn already persisted by the two enveloped publishes on the same channel.
- Removed a now-dead `use_recall` assignment left behind in `api_routes.py`.

## Outcome moved

`bus_fallback_log` stops accruing roughly one row per chat turn. That is the
whole point: the backlog watcher shipped hours earlier (PRs #1653/#1658) was
being driven almost entirely by this duplicate, so its escalating email alerts
tracked Juniper's own chat volume rather than any real routing failure. With
this deleted, the baseline goes to near zero and an alert means something.

Secondary: one WARNING per turn disappears from `orion-sql-writer` (288
ERROR/WARN lines per day at the observed rate), and `orion-spark-concept-induction`
stops ingesting a third redundant copy of every turn.

## Current architecture

Each finished Hub turn published to `orion:chat:history:log` **three** times:

1. `chat.history.message.v1` envelope — the user message
2. `chat.history.message.v1` envelope — the assistant message
3. a raw `dict` with `prompt`/`response`/`spark_meta`/`reasoning_trace`

...plus a `chat.history` turn envelope on `orion:chat:history:turn`.

Only (3) was off-contract. `orion/core/bus/codec.py:72` does
`kind = legacy.get("kind") or "legacy.message"`, and sql-writer's route map has
no `legacy.message` entry, so `handle_envelope` reached the terminal
`_write_fallback(..., "Unknown kind")` at `worker.py:2370` every time.

The two publishers matched the two `source` values seen in the live data:

| Publisher | `source` field | Live rows |
|---|---|---|
| `orion/hub/turn_orchestrator.py` (WS) | `source_label` → `hub_orion` | 76 |
| `services/orion-hub/scripts/api_routes.py` (HTTP) | `settings.SERVICE_NAME` → `hub` | 4 |

## Architecture touched

`orion-hub` producer side only. No schema, channel, contract, env, or
dependency change — this deletes an unregistered publish that no contract ever
described. The `orion:chat:history:log` channel keeps its registered
`ChatHistoryMessageV1` traffic unchanged.

## Files changed

- `orion/hub/turn_orchestrator.py`: deleted the raw publish in
  `_publish_unified_turn_chat_history`; left a comment explaining why, and
  explicitly warning off the tempting-but-wrong fix (see below).
- `services/orion-hub/scripts/api_routes.py`: deleted the raw publish, its
  `chat_log_payload` dict, and its `THOUGHT_DEBUG_HUB` block; removed the
  now-unused `use_recall` assignment.
- `services/orion-hub/tests/test_chat_history_no_raw_publish.py`: new.
- `docs/superpowers/pr-reports/2026-08-14-fallback-backlog-alerts.md`:
  correction.
- `docs/superpowers/pr-reports/2026-08-14-fallback-watch-review-fixes.md`:
  correction.

## The trap this deliberately avoids

`PUBLISH_CHAT_HISTORY_LOG` gated the WS raw publish, so flipping that flag to
`false` looks like a one-line fix. **It is not.** The same flag gates the two
*real* publishes at `services/orion-hub/scripts/chat_history.py:288` and `:392`.
Turning it off would silence the duplicate and kill `chat_history_log`
persistence with it. Only the raw block may be deleted. This is recorded as a
comment at the deletion site and asserted by
`test_ws_turn_still_persists_the_turn`.

## Consumer safety check (why deleting this is safe)

`orion/bus/channels.yaml` lists four consumers on `orion:chat:history:log`.
Each was checked against live runtime, not just code:

- **orion-sql-writer** — the entire problem. Drops it to `bus_fallback_log`.
- **orion-vector-writer** — not actually subscribed. `app/settings.py:29`:
  "never subscribed to `orion:chat:history:log` by default". channels.yaml is
  stale here.
- **orion-vector-host** — `_handle_chat_history` (`app/main.py:168`) runs
  `ChatHistoryMessageV1.model_validate(payload_obj)`; the raw dict has no
  `role`/`content` and is rejected. Live: `grep -c "Chat history payload
  invalid"` over 24h of `orion-athena-vector-host` logs → **0**, because
  `_should_skip_memory` discards it even earlier (no `memory_tier`). Either way,
  no behavior change.
- **orion-spark-concept-induction** — the one real consumer.
  `bus_worker.py:306-313` explicitly falls back to merging `prompt`/`response`
  when no `content` key is present, so it *did* ingest the raw dict as a
  `chat_turn`. Two live facts make removing it safe:
  1. It also receives the enveloped copies on the same channel, carrying the
     same text. Over 48h: **24 × `chat.history.message.v1`, 11 ×
     `legacy.message`** — two enveloped messages per turn against one raw dict.
  2. Concept induction is **off**. All **2043** of 2043 trigger decisions in
     that window were `decision=disabled`.

Live evidence for (1) and (2):

```text
$ docker logs orion-athena-spark-concept-induction --since 48h \
    | grep "worker_event_received channel=orion:chat:history:log" \
    | sed 's/.*kind=/kind=/' | sort | uniq -c
     24 kind=chat.history.message.v1
     11 kind=legacy.message

$ docker logs orion-athena-spark-concept-induction --since 48h \
    | grep "trigger_decision" | sed 's/.*decision=\([a-z_]*\).*/\1/' \
    | sort | uniq -c
   2043 disabled
```

## The correction to the earlier PR reports

Both `2026-08-14-fallback-backlog-alerts.md` and
`2026-08-14-fallback-watch-review-fixes.md` stated that `legacy.message` rows
were "real cognition being silently dropped" and "persisted nowhere else."

That was wrong. Measured against live Postgres:

```text
total=80  no_history_row=0  response_matches_exactly=80  differs=0
```

Every one of the 80 rows has a matching `chat_history_log` row with a
byte-identical response length. Nothing was ever lost. Severity in both reports
is corrected down from medium, and both now carry an explicit correction block
rather than a silent edit.

## Schema / bus / API changes

- Added: none
- Removed: an **unregistered** raw-dict publish to `orion:chat:history:log`.
  It was never in `orion/schemas/registry.py` or `orion/bus/channels.yaml`.
- Renamed: none
- Behavior changed: that channel now carries only registered
  `ChatHistoryMessageV1` envelopes.
- Compatibility notes: no consumer depended on it (see consumer safety check).
  `services/orion-sql-writer/tests/test_route_map_completeness.py:47` still
  exempts `legacy.message` under `LEGACY_KIND_ALIASES` as "resolved at runtime."
  That exemption is still wrong in principle and is left alone here — it now has
  no live producer to hide, so tightening it is a separate patch.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: not applicable, no env change
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not
  applicable, no env template changed
- skipped keys requiring operator action: none

Note: `PUBLISH_CHAT_HISTORY_LOG` is deliberately **unchanged** and must stay
`true` — see "The trap this deliberately avoids".

## Tests run

```text
$ cd services/orion-hub && pytest tests/test_chat_history_no_raw_publish.py -q
6 passed, 2 warnings in 3.21s
```

Full Hub suite, branch vs `main`, same command both sides:

```text
branch: 55 failed, 1193 passed, 2 skipped, 2 deselected in 224.81s
main:   55 failed, 1187 passed, 2 skipped, 2 deselected in 224.07s
```

+6 passing = exactly the new file. Failure *sets* diffed with `comm`, not just
counts. One test appeared only on the branch —
`test_substrate_mutation_manual_route_routing.py::test_routing_dry_run_produces_trial_and_decision_without_side_effects`
— and is order-flaky in-suite, not a regression: it passes 3/3 in isolation on
**both** branch and main, and this patch touches nothing it exercises. The two
`test_attention_loops_ui_smoke.py` tests were deselected; they fail on `main`
too (missing static asset path).

The 55-failure baseline is pre-existing on `main` and untouched by this patch.

## Evals run

```text
No eval harness exists for the Hub chat-publish seam.
```

The behavioral tests here run the real `_publish_unified_turn_chat_history`
against a recording bus, which is the closest thing to a contract eval this
seam has. A proper eval would need a live bus + sql-writer round trip; that is
the `orion-sql-writer` contract smoke's job, not this patch's.

## Docker/build/smoke checks

None run. This deletes code from an existing service with no dependency,
config, port, or boot-path change, so a rebuild proves nothing a test does not.
Restart commands below are what actually applies the change.

## Review findings fixed

- Finding: the fix was initially scoped to `orion/hub/turn_orchestrator.py`
  only, from a prior session's analysis.
  - Fix: found the HTTP twin at `services/orion-hub/scripts/api_routes.py:3177`
    and deleted it too.
  - Evidence: the two publishers set different `source` values
    (`source_label`→`hub_orion` vs `settings.SERVICE_NAME`→`hub`), matching the
    76/4 split measured in `bus_fallback_log` exactly. Deleting only the first
    would have left the HTTP path emitting fallbacks.
- Finding: the behavioral test asserted the turn envelope's kind contains
  `"turn"`.
  - Fix: it is `chat.history` on the *turn channel*. Rewrote the assertion to
    pin `(channel, kind)` for all three publishes.
  - Evidence: `['chat.history.message.v1', 'chat.history.message.v1',
    'chat.history']` — the assertion was wrong, the code was right.
- Finding: a behavioral test alone would not catch reintroduction on a Hub path
  it does not exercise.
  - Fix: added the AST-based static gate over four Hub sources.
  - Evidence: mutation test — re-planting the deleted publish fails 3 tests
    (both behavioral, plus the static gate); removing it returns 6 passed.
    `test_static_scanner_detects_a_planted_offender` pins the scanner itself
    against a synthetic offender so the gate cannot rot into a no-op.
- Finding: `use_recall` in `api_routes.py` became a dead assignment.
  - Fix: deleted.
  - Evidence: it had no other reader inside that `try` block.

## Restart required

```bash
sudo docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build hub
```

Verify afterwards — this should return **zero new rows** over a few chat turns:

```bash
psql -h localhost -p 55432 -U postgres -d conjourney -c \
  "select count(*) from bus_fallback_log
   where kind='legacy.message' and created_at_ts > now() - interval '1 hour';"
```

## Risks / concerns

- **Severity: low.** If concept induction is ever re-enabled, it will see the
  user and assistant messages as two separate `chat_turn` triggers rather than
  additionally as one merged `prompt\nresponse` blob. The text content is fully
  present either way; only the packaging differs. Called out because it is the
  one real behavioral difference, and because `bus_worker.py:306`'s
  `prompt`/`response` fallback branch now has no live producer feeding it from
  this channel.
- **Severity: low.** `channels.yaml` still lists `orion-vector-writer` as a
  consumer of `orion:chat:history:log`, which its own settings say is false.
  Pre-existing staleness, surfaced here, not fixed — a contract-registry patch
  should not ride along on a producer deletion.

## PR link

<filled in after push>
