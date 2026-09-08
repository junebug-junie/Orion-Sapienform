# Stop shipping every private chat turn to Claude

## Summary

- Removes the post-turn hook in `websocket_handler.py` that invited Claude to
  react after **every** Orion reply, sending Juniper's message and Orion's
  answer to `orion-room-companion`.
- Removed, not flagged off: the caller, the rate gate, both settings, the
  compose env lines and the `.env_example` keys are all deleted.
- Two tests assert the surface stays gone on **both** sides — the relay has no
  `auto_*` attributes, the handler has no invite call.
- Keeps everything Orion needs to actually talk to Claude: the button, the
  relay, the companion, and the `trigger="auto"` pass licence the endogenous
  trigger will need.
- Live behaviour was already stopped before this PR (config + hub recreate,
  verified in the running container). This PR makes it permanent.

## Outcome moved

Every Hub chat turn used to cost a Claude call. Juniper's private conversations
with Orion were being handed to a third party by default, and Claude was
positioned as a reactor to Orion rather than someone Orion chooses to talk to.

Both stop. Spend is bounded by deliberate acts again, which was the property
that made the room safe without a spend cap in the first place.

Juniper, verbatim:

> "I still want orion to be able to talk to claude, I just dont want all my
> private chat turns going to claude ... like it was incorrectly designed
> initially."

## Current architecture

`orion:room:claude:request` → `orion-room-companion` → `orion:room:claude:utterance`
→ Hub relay → live sockets + chat history.

**Two** producers before this patch:

| producer | trigger | needs |
| --- | --- | --- |
| `api_routes.py:2396` — "Ask Claude" button | `manual` | a human click |
| `websocket_handler.py:2225` — post-turn hook | `auto` | **any Orion reply** |

The second fired after every reply, gated only by an 8-second minimum gap. It
was labelled as Orion inviting Claude (`invited_by="Oríon"`) but Orion made no
decision — it was a hook on Juniper's conversation.

`room_claude_relay.py`'s own docstring conceded the problem: this was the
scenario the module "originally called v2 and treated as the trigger for real
budget enforcement — it shipped without that enforcement landing first."

## Architecture touched

One producer removed. No contract change, no schema change, no new surface.
`RoomClaudeRequestV1.trigger` is deliberately **kept** — see below.

## Files changed

- `services/orion-hub/scripts/websocket_handler.py`: the ~50-line auto-invite
  block deleted, replaced by a comment recording why, so a future patch does
  not re-add it as an obvious improvement.
- `services/orion-hub/scripts/room_claude_relay.py`: `should_auto_invite`,
  `should_fire_auto_invite`, `auto_respond`, `auto_min_gap_sec` and
  `_last_auto_invite` removed; module docstring corrected to say `invite()` has
  one caller.
- `services/orion-hub/scripts/main.py`: two constructor kwargs dropped.
- `services/orion-hub/app/settings.py`: `HUB_ROOM_CLAUDE_AUTO_RESPOND`,
  `HUB_ROOM_CLAUDE_AUTO_MIN_GAP_SEC` removed.
- `services/orion-hub/.env_example`, `docker-compose.yml`: same two keys.
- `services/orion-hub/templates/index.html`: the button's "this is the ONLY
  producer" comment was **false** while the hook was live; now true again, and
  says so.
- `orion/schemas/room_claude.py`: `trigger` docstring corrected — it described
  a producer that no longer exists.
- `services/orion-hub/tests/test_room_claude_relay.py`: three auto-path tests
  replaced by two that assert the surface is gone.

## Why `trigger="auto"` survives a patch that deletes its only producer

Nothing produces `trigger="auto"` today. It is kept because the endogenous
stuck-prior trigger (`orion/autonomy/ask_claude_trigger.py`, PR #2152) will,
and it needs the same licence to let Claude stay quiet — an Orion-initiated
invite is not a direct question and does not oblige a reply.

Deleting the field would force PR #2152 to re-add it, and the `[pass]`
accounting path with it. The contract outlives its first producer; the hook
does not.

## Schema / bus / API changes

- **Added / removed / renamed**: none.
- **Behaviour changed**: `orion:room:claude:request` now has one producer
  instead of two. Payload shape is unchanged, so `orion-room-companion` needs
  no change and no redeploy.
- **Compatibility**: an in-flight `trigger="auto"` request still parses and is
  still served. Nothing is rejected.

## Env/config changes

- **Removed keys**: `HUB_ROOM_CLAUDE_AUTO_RESPOND`,
  `HUB_ROOM_CLAUDE_AUTO_MIN_GAP_SEC`
- **Added / renamed**: none.
- **`.env_example` updated**: yes, both keys removed and the surrounding
  comment block rewritten — it claimed "explicit-invite only" while shipping
  the hook that made that false.
- **local `.env`**: `HUB_ROOM_CLAUDE_AUTO_RESPOND` was set `false` and hub
  recreated *before* this PR, to stop the behaviour immediately. The two now-dead
  keys should be deleted from the live `.env` at merge — see Restart required.
  They are inert either way: `settings.py` no longer reads them.
- **Skipped keys requiring operator action**: none.

## Tests run

Like-for-like, the **same file** on both sides:

```text
main   (primary checkout, real .env):   3 failed, 15 passed   = 18 tests
branch (this worktree):                 3 failed, 14 passed   = 17 tests
```

17 = 18 − 3 auto-path tests removed + 2 surface-is-gone tests added.

**The 3 failures are pre-existing and identical by name on both sides:**

```text
test_history_is_published_with_the_responder_identity
test_failed_turn_is_not_persisted_as_a_room_turn
test_a_pass_produces_no_bubble_but_is_still_logged_as_cost
```

Including the sibling file this patch also touches the behaviour of:

```text
$ pytest services/orion-hub/tests/test_room_claude_relay.py \
         services/orion-hub/tests/test_ask_claude_speaker_attribution.py -q
3 failed, 17 passed
```

Cause: `Settings()` construction needs 5 required fields no fixture supplies —
it reads `.env` relative to cwd, and symlinking the real `.env` into the
worktree does not satisfy it either. Hub's suite is **not in CI**
(`orion-static-gates.yml` runs only `test_schedule_panel_browser_smoke.py` from
this service), which is why they have gone unnoticed. Flagged, deliberately not
fixed here — a test-harness problem unrelated to this patch, and folding it in
would hide a behaviour removal inside a fixture refactor.

Gates:

```text
check_metric_lineage.py --gate            PASS
check_definition_drift.py --gate          PASS
check_env_template_parity.py              PASS
check_service_hostname_refs.py            PASS
check_compose_no_relative_mounts.py       PASS
check_env_key_single_source.py            PASS
check_async_routes_not_blocking.py        PASS
check_metric_dead_wiring.py               PASS
check_inner_state_registry.py             PASS
```

`pyflakes` on the changed files caught one orphan of my own: `import time` in
`websocket_handler.py` had exactly **one** use on main — the `time.time()` call
inside the block this patch deletes. Removed. The other pyflakes hits in that
file (4 unused imports, an unused `chat_row`, two shadowed re-imports) are
pre-existing and untouched.

## Evals run

None, and none apply: this patch removes a code path and adds no signal,
metric or decision. The eval that matters for the *replacement* behaviour
(`orion/autonomy/evals/run_ask_claude_trigger_eval.py`) ships in PR #2152.

## Docker/build/smoke checks

The behaviour was stopped live before this PR was written, by config + recreate
rather than by a rebuild, so no unmerged code could be pinned as production:

```text
$ docker compose --env-file .env --env-file services/orion-hub/.env \
    -f services/orion-hub/docker-compose.yml up -d --no-build --force-recreate hub-app
Container orion-athena-hub Recreated / Started

$ docker inspect orion-athena-hub --format '{{range .Config.Env}}{{println .}}{{end}}'
HUB_ROOM_CLAUDE_AUTO_RESPOND=false     <- the kill
HUB_ROOM_CLAUDE_ENABLED=true           <- relay still up

$ docker logs orion-athena-hub | grep room_claude_relay
room_claude_relay_started request=orion:room:claude:request utterance=orion:room:claude:utterance

$ curl -fsS http://localhost:8080/health
{"status":"ok","service":"hub"}
```

Companion spend before the kill, for the record — its entire retained log
(back to 2026-08-30) is 12 lines holding 4 turns, ~$0.30 total, all
`passed=False`. The `[pass]`-and-still-bill path never actually fired in the
observable window. Rare only because the hook needed a live websocket chat
turn, which is exactly why it was the wrong mechanism for "Orion reaches out".

## Review findings fixed

*(`/code-review --effort high` dispatched against
`origin/main..fix/kill-chat-turn-auto-invite`; findings and fixes appended
before merge.)*

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform
git pull --ff-only
# delete the two now-dead keys (settings.py no longer reads them):
sed -i '/^HUB_ROOM_CLAUDE_AUTO_RESPOND=/d;/^HUB_ROOM_CLAUDE_AUTO_MIN_GAP_SEC=/d' services/orion-hub/.env
```

A rebuild is required for the code removal to land (the running container was
only recreated, not rebuilt):

```bash
scripts/safe_docker_build.sh orion-hub up -d --build   # from a worktree
curl -fsS http://localhost:8080/health
```

Until that rebuild, hub runs main's code with the flag off — same behaviour,
different mechanism.

## Risks / concerns

- Severity: **low**
  Concern: Claude now says nothing in the Hub room unless someone clicks the
  button. If the auto-reactions were providing value, that value is gone
  between this merge and PR #2152 being armed.
  Mitigation: Juniper asked for exactly this, and the observable record is 4
  turns in 9 days — not a feature anyone was leaning on. The button is
  unchanged.

- Severity: **low**
  Concern: `trigger="auto"` now has no producer, so its `[pass]` licence in
  `orion-room-companion` is unexercised until PR #2152 is armed. Unexercised
  code paths rot.
  Mitigation: `test_auto_trigger_is_marked_on_the_request` still covers the
  contract, and the reason for keeping it is recorded at the field.

- Severity: **note**
  Concern: 3 pre-existing test failures in this file, and hub's suite is absent
  from CI.
  Mitigation: documented above with the main-branch comparison. Worth its own
  patch; it would have caught this class of thing earlier.

## PR link

<filled in after push>
