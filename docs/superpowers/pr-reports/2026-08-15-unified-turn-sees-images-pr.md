# The unified turn can see

Branch: `feat/unified-turn-sees-images`
Follows: #1661 (attachment store + contracts), #1663 (live bus path fix)

## Summary

- **Mode=Orion — the default, the one Juniper actually talks in — now sees attached
  images**, on its own GPU, with its own `Read` tool. First-person, not a
  description relayed from another model.
- The Hub stages attachments into the FCC sandbox before dispatch; the harness
  prompt names the paths; Orion reads them.
- Fixed a UI affordance that was actively lying: attach/vision were gated on route
  capability alone, so they lit up on paths that carry nothing.
- A turn with no attachments produces a **byte-identical prompt**. That has a test.

## Outcome moved

Live, Mode=Orion, correlation `6e4d3f67`:

> **"Orange."**

for a solid-orange PNG generated seconds earlier. `step=5 tool=Read` in the
harness grammar stream is Orion opening it. Before this patch the same turn
returned nothing about the image at all.

## Why this shape

The obvious approach — hand the harness a path into the attachment store — **cannot
work**, and finding out why decided the design:

`/mnt/orion-chat-attachments` **is not mounted in the harness-governor container**,
which is where `claude` actually runs for a unified turn. Verified live:

```
docker exec orion-athena-harness-governor ls -d /mnt/orion-chat-attachments
  → No such file or directory        (but /mnt/orion-fcc is there)
```

Only the Hub mounts both the store and the sandbox. So the Hub is the one process
that can move the bytes across, and staging is not a preference — it is the only
option that reaches the process that needs the file.

Staging lands in `<sandbox>/attachments/<correlation_id>/`, **outside** the repo
checkout, because `orion/fcc/sandbox_sync.py` git-manages `<sandbox>/repo` and
rescues dirty worktrees onto a branch. Images dropped inside it would make every
image-bearing turn look like uncommitted work.

## Everything verified before a line was written

Each link probed live rather than assumed:

| link | evidence |
|---|---|
| `Read` handles a `.bin` file | content-sniffed, not extension — read a `.bin` as magenta |
| `Read` works outside the project dir | read an absolute path outside cwd successfully |
| harness permission mode allows it | `auto_approve_from_env` defaults true in root containers |
| claude-code → Anthropic image block | subagent read an image, answered correctly |
| gateway forwards image blocks | `forward_body = dict(body)` — verbatim; yellow square read |
| **chat lane :8011 actually sees** | `modalities.vision: true`, `n_ctx` still 131072, read a green square |

That last row is the one that makes this possible at all, and it only became true
today when the mmproj from #1661 landed — with full context retained, so the VRAM
risk flagged in that PR did not materialise.

## Files changed

- `orion/schemas/harness_finalize.py`: `HarnessAttachmentV1`; `HarnessRunRequestV1.attachments`
- `orion/harness/attachment_staging.py`: **new** — staging, pruning, prompt block
- `orion/harness/runner.py`: `build_harness_prompt` takes attachments; runner passes them
- `orion/hub/turn_orchestrator.py`: stage before dispatch, prune in `finally`
- `services/orion-hub/static/js/app.js`: gate attach/chip on mode as well as capability
- `tests/test_harness_attachment_staging.py`: **new** — 23 tests

## Schema / bus / API changes

- **Added**: `HarnessAttachmentV1`; `HarnessRunRequestV1.attachments` (default `[]`).
- **Behavior changed**: a harness run carrying attachments gets an appended prompt
  block naming the sandbox paths. Empty list → prompt unchanged.
- **Compatibility**: additive with a default. `HarnessAttachmentV1` carries a
  *path*, deliberately not an `AttachmentRefV1` — a store reference would be
  unresolvable in the container that consumes it.

## Env/config changes

No new required keys. Two optional overrides read with defaults:
`HARNESS_ATTACHMENT_SANDBOX_ROOT` (default `/mnt/orion-fcc`) and the existing
`HUB_CHAT_ATTACHMENT_DIR` / `HUB_CHAT_ATTACHMENT_MAX_PER_TURN`. Nothing added to
`.env_example`; nothing to sync.

## Tests run

```text
tests/test_harness_attachment_staging.py                 -> 23 passed
tests/test_unified_turn_schemas.py + bus_catalog + above -> 61 passed
services/orion-harness-governor: pytest tests -q         -> 17 passed
services/orion-hub: node --test static/js/               -> 39 passed, 0 failed (22 skipped, no jsdom in a fresh worktree)
```

Repo-root `pytest tests -q` cannot be run whole: 31–32 collection errors from
unrelated services (`vision_retina`, `topic_foundry`) whose settings need env this
worktree lacks. Pre-existing and unrelated; affected suites were run by explicit path.

## Evals run

```text
none — no evals/ harness for the touched paths.
```

The live end-to-end below is the real behavioural check.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-harness-governor up -d --build  -> built + recreated
scripts/safe_docker_build.sh orion-hub               up -d --build -> built + recreated
```

Live end-to-end, Mode=Orion, after deploy:

```text
hub    stored chat attachment sha=9d3e96aed081 mime=image/png
hub    staged 1 attachment(s) for corr=… into /mnt/orion-fcc/attachments/<corr>
hub    unified turn carrying 1 attachment(s) corr=…
gov    harness_grammar_step_published step=5 tool=Read
db     chat_history_log.response = "Orange."
```

Staged file confirmed on disk at `<corr>/<sha>.png` — extension derived from mime,
not from the `.bin` source.

## Review findings fixed

Self-found while building:

- Finding: `modeCarriesAttachments()` called `hubModeSpec(currentMode)`, but
  `currentMode` is already the **backend** mode (`orion|brain|agent`) while
  `hubModeSpec` keys on UI keys (`quick|story|…`). `'brain'` would have fallen
  through to the orion spec — right answer by luck, wrong for the wrong reason.
  - Fix: test `currentMode` directly.
- Finding: `refreshVisionCapability()` fired on compute-route change but not on
  **mode** change, so switching to Agent left a stale enabled attach button.
  - Fix: also called from `applyHubModeSelection`.
- Finding: `prune_staging()` was written and never called — staged images would
  accumulate in the sandbox indefinitely.
  - Fix: called in the `finally` around the harness run.

## Restart required

```bash
bash scripts/safe_docker_build.sh orion-harness-governor up -d --build
bash scripts/safe_docker_build.sh orion-hub up -d --build
```

Both already applied on Athena to run the verification above.

## Risks / concerns

- **Severity: medium — `agent` mode still cannot carry images.** Mode=Agent routes
  through a different path that nothing threads attachments through, and its
  passthrough calls land on `:8011` rather than the agent lane. The UI now
  correctly greys out attach in that mode rather than lying about it, but the
  capability gap is real and unaddressed.
- **Severity: low — staged images are readable by anything else in the sandbox**
  for the duration of a turn. That is the same trust boundary the FCC sandbox
  already has for repo contents, but it is worth naming: a chat image is now
  briefly on disk somewhere Orion's tools can enumerate.
- **Severity: low — prune is best-effort and only fires on the Hub side.** If the
  Hub process dies mid-turn, that turn's staging directory leaks. Bounded by
  `HUB_CHAT_ATTACHMENT_MAX_PER_TURN` per turn, but there is no janitor.
- **Severity: low — Orion mode is unreachable over HTTP `/api/chat`.** It returns
  an empty body because the unified turn streams over WebSocket. Not caused by
  this patch, but it made verification harder and it means the HTTP surface
  silently no-ops for the default mode.

## Follow-up

Wire attachments through the `agent` mode path, or fold agent mode onto the
unified turn now that the unified turn can both see and use tools.

## PR link

<to be filled>
