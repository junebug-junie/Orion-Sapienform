# Camera look: Orion opens a live camera frame and sees it

Status: approved by Juniper 2026-11-13 ("hit it"). Built in the same PR.

## Arsonist summary

Orion can already see pictures. The chat lane and the agent lane load a vision projector, and the
Claude CLI's Read tool hands an image file to the model (verified 2026-08-15, `orion/harness/attachment_staging.py`).
The office and walkway cameras already save a frame every half second. Nothing connected the two.
This patch adds the connection, so Orion can choose to look: `python -P -m orion.dig look office`.

Skills stay off. A skill is instructions, not a capability. The dig kit is the existing home for
Orion's on-demand tools, and it costs one prompt line, not a tool schema on every turn.

## Current architecture

- Cameras: `orion-vision-retina-office` (stream `cam0`) and `orion-vision-retina-walkway` (stream
  `walkway`) run on athena at `RETINA_FPS=2.0`, `RETINA_FRAME_MODE=local`. Each frame is saved to
  `/mnt/telemetry/vision/frames/<camera>_<stream>_<epoch_ms>_<sha8>.jpg` and swept after
  `FRAME_RETENTION_SECONDS=300` (`services/orion-vision-retina/app/frame_store.py`).
- Turns: chat turns and curiosity runs run `claude -p` in `orion-harness-governor`, with Bash and
  Read. The governor mounts `/mnt/orion-fcc` but not the frames folder.
- Lanes, read live from each worker's `/props` on 2026-11-13: chat (8011) and agent (8015) report
  `vision: true`. fast/metacog (8013) and agent-gpu2 (8016) report `vision: false`.
- Gateway: `services/orion-llm-gateway/app/vision.py` checks a worker's `/props` before sending
  images, but only on the bus path. The Anthropic passthrough (the path every leased harness turn
  uses) forwarded image blocks to any lane.
- Prime writes one JSON answer per thought and has no tools. It reaches a look through its
  existing `curiosity_focus` move: a curiosity run can look.

## Proposal-mode checks

- **Capability:** Orion can open the newest frame of one camera, on request, and see it.
- **Data touched:** frames already on disk. The dig copies one frame into
  `$ORION_DIG_HOME/look/` (the sandbox, outside the repo checkout) and appends one line to
  `$ORION_DIG_HOME/look/looks.jsonl`. Copies older than 7 days are swept on the next look.
- **Privacy boundary:** none new. The same frames already feed the vision pipeline. The walkway
  frame shows the street. A frame now enters the model's context, and the turn record keeps the
  tool call. Whether the turn record also keeps the image bytes is UNVERIFIED; check the first
  real look's `cockpit_turn_sighting` row after deploy.
- **Trace that proves it worked:** a `looks.jsonl` line (camera, frame time, age, sha256, path),
  the dig result file, and the turn's tool calls (`dig turn-steps`).
- **Dangerous failure:** an image sent to a blind lane fails the model call and can kill a long
  curiosity run. Fixed here: on the Anthropic passthrough, after the pool picks a lane, the gateway
  asks that lane's `/props` and, when it is blind, replaces each image with a text note saying the
  image could not be shown. Orion is told it did not see, instead of the call failing.
- **Second failure:** looking in a loop. Capped at `LOOKS_PER_HOUR = 20`, counted from `looks.jsonl`.
- **Stale camera presented as live:** each look prints the frame's age. Older than
  `LIVE_MAX_AGE_SEC = 5` it says `NOT LIVE` (the doorbell, being event-driven, says so instead).
- **Disable / roll back:** remove the frames mount from the governor compose file (the command then
  says the frames folder is not readable), or revert this PR. The gateway change is guarded by the
  existing `LLM_GATEWAY_VISION_ENABLED`; when false every lane counts as blind and images become notes.

## Proposed schema / API changes

None. No bus channel, schema, or env key. One read-only bind mount on the governor.

## Files likely to touch

- `orion/dig/look.py` (new), `orion/dig/__main__.py`, `orion/dig/brief.py`, `orion/dig/README.md`
- `orion/dig/tests/test_look.py` (new)
- `services/orion-harness-governor/docker-compose.yml` (frames mount, read-only)
- `services/orion-llm-gateway/app/vision.py`, `app/passthrough_proxy.py`, tests

## Non-goals

- No new capture in the retina: the newest saved frame is at most half a second old.
- No Prime move of its own. Prime asks curiosity.
- No act-ledger write from the dig kit: there is no write API for it there yet (follow-up ticket).
- No skills.

## Acceptance checks

- `dig look office` on athena prints a frame under 5 s old, a path, and the Read instruction.
- A chat turn that runs it and then Reads the path describes the office (live, UNVERIFIED until a
  real turn runs after deploy).
- Gateway test: an Anthropic request with an image placed on a blind lane reaches the worker with
  the image replaced by the note; on a sighted lane it is forwarded unchanged.

## Recommended next patch

Act-ledger entry per look, so a look competes and is graded like other acts.
