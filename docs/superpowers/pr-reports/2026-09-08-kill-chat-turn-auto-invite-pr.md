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
  replaced by two that assert the surface is gone, plus the pass-path test
  rewritten to actually exercise delivery (it was passing for the wrong
  reason) and a new test pinning that a pass is still scoped, not broadcast.
- `services/orion-room-companion/app/room_prompt.py`: `AUTO_INVITE_CLAUSE`
  reworded — review finding 1, the one with real behavioural consequences.
- `orion/bus/channels.yaml`: catalog entry corrected to one producer.
- `services/orion-hub/scripts/api_routes.py`: inverted docstring reference.
- `orion/substrate/seed_concepts.yaml`: Orion's own concept of Claude.
- `services/orion-hub/tests/conftest.py`: supply the five no-default `Settings`
  fields, which is what makes the pass-path test runnable at all.
- `config/metrics/metric_definitions.lock.json`: re-locked. Inherited a
  `_last_change` block from main (PR #2152 merged mid-work); the drift gate
  checks it on a PR branch but skips it on the base branch, so main passed
  while any branch off it failed. `--update` is the gate's own remedy.

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
main   (primary checkout, real .env):   3 failed, 15 passed
branch (this worktree):                18 passed
```

The 3 failures were pre-existing and identical by name on main. They are now
**fixed**, because review finding 7's fix needed one of them to run — see
"those 3 pre-existing failures" above. Test count 18 = 18 on main − 3 auto-path
tests + 2 surface-is-gone tests + 1 pass-is-still-scoped test.

Mutation check on the finding-7 fix:

```text
fix reverted:   1 failed, 17 passed   <- test_a_pass_pushes_an_empty_frame_so_the_ui_unsticks
fix restored:   18 passed
```

Full hub suite, branch vs main baseline: see the run recorded at merge time.

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

`/code-review --effort high` returned **7 findings. All 7 are fixed.** The
mechanical removal was confirmed correct; what it missed was a set of surfaces
still describing the deleted producer as live — including one that would have
changed Claude's behaviour once the successor trigger arms.

- Finding (**medium**): `AUTO_INVITE_CLAUSE` would have biased the successor
  toward a billed silence. `trigger="auto"` was kept for the endogenous
  trigger, but the only thing it *does* is append a clause reading "You are
  auto-invited after every turn in this room, so you will often have nothing
  worth adding" — calibrated for the firehose. A deliberate "I tested this
  claim ten times and cannot settle it" invite would arrive carrying a system
  prompt telling Claude it is invited constantly and should usually stay quiet,
  billing Orion a `[pass]` on precisely the question it chose to ask.
  - Fix: reworded. The licence survives (Orion is not owed a reply); the
    frequency claim and the nudge toward silence are gone, replaced by "you
    were invited by Orion, deliberately, about something specific."
  - Evidence: `services/orion-room-companion/app/room_prompt.py:52`, with the
    old text and the reason recorded at the site.

- Finding (**medium**): `orion/bus/channels.yaml` still declared **two**
  producers, including "Hub's own auto-invite after every Orion reply", and
  said "spend is not bounded by clicks alone". CLAUDE.md §6 requires the
  contract surface in the same changeset, and this is the file someone auditing
  spend exposure reads first — they would have concluded the firehose was live.
  - Fix: one producer, with the removal and the intended successor recorded.

- Finding (**low**): `api_routes.py:632`'s docstring pointed at a
  websocket_handler comment "for why spend is no longer bounded by clicks
  alone" — that comment now says the opposite. An inverted dangling reference
  sitting in the file that holds the only surviving producer.
  - Fix: rewritten to say it is once again the only producer.

- Finding (**low**): `orion/substrate/seed_concepts.yaml:59` — Orion's own
  seed concept for Claude said Claude is invited "either by an operator click
  or by Orion's own auto-respond path." This file is loaded live by
  `orion/substrate/seed.py`, so it is self/world-model context **Orion reads
  about Claude**, and it was now false.
  - Fix: corrected. Note a re-seed is needed for already-ingested nodes — see
    Restart required.

- Finding (**low**): my anti-regression guard was name-shaped and evadable. It
  asserted on the literal `_room_relay.invite(`, but this very test file holds
  the relay as `room_relay` (no underscore) and the handler uses that spelling
  six times for legitimate register/unregister calls. A re-add in that nearby
  style, or extracted to a helper, would have passed every assertion while
  restoring the exact behaviour the test claims to prevent.
  - Fix: assert on `.invite(` and `trigger="auto"` — each **zero** in the
    handler, and either is unavoidable for a real re-add. Old symbol names kept
    too, so a straight revert is caught by name as well.

- Finding (**low**): stale docstring at `test_room_claude_relay.py:141`
  referencing "the auto-invite path" in a file this PR edits.
  - Fix: corrected.

- Finding (**low, pre-existing, but this PR makes it the only path**): a
  `[pass]` pushed **no frame at all**, so `app.js` — which clears the
  "thinking…" chip and re-enables the Ask Claude button *only* on a
  `room_claude_utterance` frame — left the chip spinning and the button dead
  until reload. Masked while the auto path existed (passes there had no button
  to unstick). The button is now the only path, and it becomes guaranteed once
  the endogenous trigger, where passing is the **expected** outcome, carries a
  `connection_id`.
  - Fix: a pass now pushes a frame with empty `llm_response` and `passed: true`.
    Empty text is exactly right on the client — the chip clears *before* its own
    `if (claudeText)` guard, so nothing is appended and **no client change was
    needed**.
  - Evidence: mutation-checked. With the fix reverted,
    `test_a_pass_pushes_an_empty_frame_so_the_ui_unsticks` fails; restored, it
    passes.

### The test for that last fix was passing for the wrong reason

Worth recording separately. The existing `assert q.qsize() == 0, "a pass must
not render a bubble"` was **not** testing the pass path. `_utterance()` sets no
`session_id`, and with no session and no pending invite, `_push`'s scoping
chain drops the frame **by design** — so the assertion was satisfied by the
frame never being addressed, not by the pass path declining to send one. It
would have passed either way.

Fixed by giving the utterance a `session_id` that matches the registered
connection, which is what makes the test exercise delivery at all. A second
test now pins the inverse: a pass with a non-matching session is still dropped,
so unsticking the UI did not turn a pass into a broadcast.

### And those 3 "pre-existing failures" are now fixed, because this patch needed them

The earlier draft of this report flagged 3 failing tests as pre-existing and
out of scope. That was the right call **until** finding 7 landed: the fix needed
coverage, and the test covering it was one of the three that could not run. A
fix I cannot test is worth nothing.

Root cause: `Settings()` has five fields with **no default**
(`Field(..., alias=...)`), so any test reaching `scripts.chat_history` →
`app.settings.settings` died at import with "5 validation errors". Symlinking
the real `.env` into the worktree does not help — the fields must be in the
environment.

Fixed in `services/orion-hub/tests/conftest.py`'s existing `pytest_configure`
hook (which already runs before any test module imports, for the
control-plane-Postgres detach), supplying the five keys with `setdefault` and
the `.env_example` values — so a real `.env` or a test's own value still wins.

```text
before:  3 failed, 15 passed   (test_room_claude_relay.py, on main)
after:   18 passed             (this branch, same file)
```

Hub's suite is still absent from CI, which is why this went unnoticed. Flagged
in Concerns.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform
git pull --ff-only
# delete the two now-dead keys (settings.py no longer reads them):
sed -i '/^HUB_ROOM_CLAUDE_AUTO_RESPOND=/d;/^HUB_ROOM_CLAUDE_AUTO_MIN_GAP_SEC=/d' services/orion-hub/.env
```

`orion/substrate/seed_concepts.yaml` changed, so Orion's already-ingested
`claude` concept still carries the old "auto-respond path" definition. A re-seed
is needed for the correction to reach it — the concept node is self-model
context Orion reads, not just a config comment.

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
