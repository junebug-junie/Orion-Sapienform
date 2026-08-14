# Kill the Hub's raw-dict chat-history publish (the `legacy.message` source)

## Summary

- Deleted the third, raw-dict publish to `chat_history_channel` in **both**
  Hub transports — `orion/hub/turn_orchestrator.py` (WS/unified turn) and
  `services/orion-hub/scripts/api_routes.py` (HTTP chat). Published with no
  `kind`, so `orion/core/bus/codec.py:72` stamped each one `legacy.message`,
  which matches no sql-writer route and fell through to `bus_fallback_log`.
- Added `services/orion-hub/tests/test_chat_history_no_raw_publish.py` — 19
  tests: 3 behavioral over the real WS publish path, 16 covering a static AST
  gate that fails if a raw dict is ever published to a chat-history channel from
  any Hub source. Mutation-verified, including a plant into the real
  `chat_history.py`.
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
  `chat_turn`. Three facts make removing it safe, in descending order of
  durability:
  1. **The merged `prompt`/`response` text it was reading is still delivered,
     verbatim, on a channel it already subscribes to.** `ChatHistoryTurnV1`
     (`orion/schemas/chat_history.py:121-132`) has `prompt` and `response` and
     **no** `content`/`text`/`message`/`summary` field, and
     `orion:chat:history:turn` is in `BUS_INTAKE_CHANNELS`. So `_extract_text`
     falls through to the exact same `f"{prompt}\n{response}"` merge on the turn
     envelope that it used to compute from the raw dict. Nothing is lost even
     with concept induction fully enabled. *This is the argument that matters* —
     it holds regardless of configuration.
  2. It also receives the enveloped messages on the log channel, carrying the
     same text via `content`. Over 48h: **24 × `chat.history.message.v1`, 11 ×
     `legacy.message`** — two enveloped messages per turn against one raw dict.
  3. Concept induction is currently **off**: all **2043** of 2043 trigger
     decisions in that window were `decision=disabled`. Noted last on purpose —
     `CONCEPT_AUTONOMOUS_TRIGGER_ENABLED` is `false` in `.env` and `.env_example`
     but the code default at `concept_induction/settings.py:57` is `True`, so
     this is a flippable flag and must not be the load-bearing part of a safety
     case.

  One non-breaking behavioral delta: `concept_induction/identity.py:78-81` used
  the raw dict's prompt+response to resolve `RELATIONSHIP_SUBJECT` on the log
  channel. That now resolves via `role` instead, and the turn channel still hits
  the relationship branch directly.

Two more raw-shape readers were checked and are non-issues:

- **orion-chat-memory** (not in channels.yaml, but this channel is its *only*
  input): `_normalize_payload` (`app/main.py:64-82`) requires
  `text`/`content`/`message`. The raw dict had none, so it was already a silent
  no-op there.
- **orion-dream** `app/memory_listener.py:23-34` is the one genuine
  `if "prompt" in raw and "response" in raw` reader in the tree. Its
  `mirror_to_buffer()` runs only under `if __name__ == "__main__":` (`:78-79`),
  nothing imports it, and the container runs `uvicorn app.main:app`. Dead path,
  and it degrades gracefully through the pass-through branch at `:36-40` even if
  revived.

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

Run from the **repo root**, the invocation CLAUDE.md §11 documents:

```text
$ pytest services/orion-hub/tests/test_chat_history_no_raw_publish.py -q -p no:randomly
19 passed, 2 warnings in 2.61s
```

Full Hub suite, branch vs `main`, identical command both sides:

```text
branch: 55 failed, 1206 passed, 2 skipped, 2 deselected in 223.06s
main:   55 failed, 1187 passed, 2 skipped, 2 deselected in 224.07s
```

+19 passing = exactly the new file. Failure *sets* diffed with `comm`, not just
counts:

```text
$ comm -23 fail_branch.txt fail_main.txt   # regressions
(empty)
$ comm -13 fail_branch.txt fail_main.txt   # accidentally fixed
(empty)
```

Identical sets. `-p no:randomly` on both sides — `pytest-randomly` is installed
and this suite is order-sensitive, so an unpinned order makes the comparison
unstable (an earlier unpinned run showed
`test_substrate_mutation_manual_route_routing.py::test_routing_dry_run_produces_trial_and_decision_without_side_effects`
on one side only; it passes 3/3 in isolation on **both** branch and main and
this patch touches nothing it exercises). The two
`test_attention_loops_ui_smoke.py` tests are deselected; they fail on `main` too
(missing static asset path).

The 55-failure baseline is pre-existing on `main` and untouched by this patch.

### Mutation test

Re-planting the deleted publish into `turn_orchestrator.py` and re-running:

```text
3 failed, 16 passed    <- both behavioral tests + the static gate fire
```

Restoring the file returns `19 passed`.

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

### Second round — subagent code review of the committed diff

The review confirmed the deletion itself as correct and well-evidenced, and
found four issues, **all in the guard rather than the change**. That is the
right place to have found them: a regression gate whose entire value is firing
years from now is worth more scrutiny than the deletion it protects.

- Finding (HIGH): **the static gate was structurally inert on
  `scripts/chat_history.py`** — one of its own four scanned sources, and the
  file that owns every legitimate chat-history publish in the Hub, so by far the
  most likely place a raw publish gets re-added. The scanner required the
  channel argument to be an `ast.Attribute`, but that module binds the channel
  to a local first (`channel = settings.chat_history_channel` at `:291`), making
  it an `ast.Name`. Every publish in the file was skipped before its payload was
  ever examined. Listing it in `_SCANNED_SOURCES` created the *appearance* of
  coverage.
  - Fix: rewrote `_dict_literal_publishes_to_chat_history` with per-scope
    binding resolution, so a local bound to a chat-history channel is tracked.
  - Evidence: new `test_static_scanner_catches_a_plant_into_the_real_chat_history_module`
    plants a raw publish into the **real** `chat_history.py` source (not a
    synthetic file) and asserts the scanner fires. It also asserts the anchor
    line still exists, so the test cannot rot into a no-op if that file changes.
- Finding (MEDIUM-HIGH): **nine further scanner false negatives**, including the
  two most natural reintroduction forms — a hardcoded `"orion:chat:history:log"`
  channel string (the repo already writes it that way in 20+ places) and keyword
  arguments (`bus.publish(channel=..., msg={...})`, valid against
  `OrionBusAsync.publish`'s real signature).
  - Fix: the scanner now accepts attribute, literal, and locally-bound channels;
    reads `node.keywords` as well as `node.args`; treats `dict(...)`,
    `AnnAssign`, walrus, `.copy()`, and transitive name chains as dict payloads;
    and matches bare `publish(...)` as well as `x.publish(...)`.
  - Evidence: `_EVASION_FORMS` parametrizes all 11 shapes as individual tests.
    Every one was a confirmed miss before this fix and passes now.
- Finding (MEDIUM): **the behavioral tests errored under the invocation
  CLAUDE.md §11 documents.** From the repo root, `pytest
  services/orion-hub/tests/test_chat_history_no_raw_publish.py -q` gave
  `3 passed, 3 errors` — Hub settings read `env_file=".env"`, which pydantic
  resolves relative to the *process cwd*, so from the repo root it picks up the
  5.9K root `.env` (no `CHANNEL_VOICE_*`) instead of the 20.6K hub one. The file
  passed only via my `cd services/orion-hub` habit, or by another test module's
  import-time side effect happening to run first.
  - Fix: added the three `os.environ.setdefault("CHANNEL_VOICE_*", ...)` lines
    already established in `tests/test_social_room_turn_publish.py:6-8`.
  - Evidence: repo-root run went from `3 passed, 3 errors` to `19 passed`. Full
    suite re-run after the change: failure set still **identical** to `main`, so
    the import-time `setdefault` shifted nothing else.
- Finding (LOW-MEDIUM): **the deletion-site comments misstated which channel
  survives**, claiming `publish_chat_turn` publishes "on the same channel." It
  does not — it uses `chat_history_turn_channel` (`orion:chat:history:turn`).
  Materially misleading, because after this patch the *log* channel no longer
  carries a prompt/response pairing at all, only two independent message
  envelopes; someone debugging a log-channel consumer would hunt for turn data
  that is not there.
  - Fix: both comments now name the two channels separately and state that
    consequence explicitly.
  - Evidence: the patch's own `test_ws_turn_still_persists_the_turn` already
    asserted the correct `(channel, kind)` triple — the test was right and the
    prose was wrong.

Two further findings were accepted as **follow-up material, not merge
blockers**, and are recorded under Risks below: `channels.yaml` under-declaring
this channel's live consumers, and two argument-quality notes on the safety
case.

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

### What the fallback log should look like afterwards

Pre-restart baseline, last 24h:

```text
legacy.message              | 21
juniper.affective_state.v1  |  5
```

`legacy.message` is what this patch removes. The other kind was checked and is
**already resolved**, not a second thing to fix: `juniper.affective_state.v1` is
in the live route map (`docker exec orion-athena-sql-writer` →
`JuniperAffectiveStateSQL`, route_map size 75), its newest fallback row is
`04:16` today, and the sql-writer container has only been up since ~07:00. Rows
now land correctly — `juniper_affective_state_log` holds 34 rows with the newest
at `09:21` today. Those 5 fallbacks are stale, from before today's restart, and
will age out of the watcher's 24h window on their own.

So the expected steady state after this restart is a `bus_fallback_log` that
stops growing, and a backlog watcher that goes quiet.

## Risks / concerns

- **Severity: low.** If concept induction is ever re-enabled, the merged
  `prompt\nresponse` text now reaches it via the *turn* envelope rather than
  additionally via the raw dict on the log channel. Content is identical (see
  the consumer safety check); only the arrival path differs.
- **Severity: low, pre-existing, not fixed here — `channels.yaml` under-declares
  this channel's consumers.** It lists three; a sweep found **five more** live or
  wired subscribers of `orion:chat:history:log` that are not declared:
  `orion-chat-memory` (`CHAT_MEMORY_INPUT_CHANNELS`, and this is its only
  input), `orion-signal-gateway` (glob `orion:chat:*`), `orion-bus-mirror`
  (`orion:*`), `orion-bus-tap` (`orion:*`, not running), and `orion-dream`
  (`CHANNEL_CHAT` default). None of them break — each was checked individually —
  but the *method* of "channels.yaml lists the consumers, so check those" would
  not have caught a breakage in any of them. Worth a follow-up given CLAUDE.md
  §6; deliberately not ridden along on a producer deletion.
- **Severity: low, pre-existing, not fixed here.** `channels.yaml` also listed
  `orion-vector-writer` as a consumer, contradicting its own settings. That one
  was already corrected by the previously-merged PR #1662, not by this patch.
- **Severity: low, noted not fixed.** The
  `services/orion-sql-writer/tests/test_route_map_completeness.py:47`
  `LEGACY_KIND_ALIASES` exemption for `legacy.message` — described as "resolved
  at runtime" when it is not — is the test that would have caught this and
  explicitly waived it. It now has no live producer to hide, so tightening it is
  a clean separate patch rather than a change to sql-writer riding on a Hub fix.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1670
