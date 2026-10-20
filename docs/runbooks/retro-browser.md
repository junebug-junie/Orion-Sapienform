# Retro browser play

Hub `/retro` provides keyboard/gamepad play without an additional access code. The native
core remains in the isolated Retro container. No ROM or core runs in the browser.
No images, audio or inputs enter memory, chat or cognition. No save/load API exists.

## Setup

From clean primary main after merge, sync templates and prepare private credentials:

```sh
python3 scripts/sync_local_env_from_example.py
python3 services/orion-retro/evals/install_catalog.py --env services/orion-retro/.env
python3 services/orion-retro/evals/configure_hub.py --env services/orion-hub/.env --origin https://YOUR-EXACT-HUB-ORIGIN
scripts/safe_docker_build.sh orion-retro up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Use the exact scheme, hostname and port from the browser. No wildcard, automatic
Host trust or forwarded-header trust. HTTPS is recommended. HTTP supports keyboard
play, but does not encrypt game traffic; gamepads require a secure
browser context (HTTPS or localhost). Do not expose this scoped operator interface
on an untrusted HTTP network. Existing Hub routes do not gain authentication.

Retro uses Hub's network access boundary: anyone who can reach Hub can control
Retro. POST requests and WebSockets still require the configured exact Origin.
There is no Retro login, cookie, access code or lock button. The installer creates
only the distinct backend `hub.token` in `~/.local/share/orion/retro/service/hub-auth/`.
It remains private (0600); neither it nor the CLI credential is sent to the browser.
Old `browser.code` files are unused and may be deleted.

Hub mounts only the dedicated Unix socket directory and Hub credential directory.
The emulator gets only `hub.token`, never browser-provided credentials. Its network remains `none`.
CLI and Hub credentials cannot access each other's session or evidence. One game
session still exists at a time. An active CLI session must be stopped by its owner
before a browser can start a game. No browser-supplied principal is trusted.

## Use

Open the Retro tab from Hub's section launcher (`/#retro`). Leaving the tab
unloads its game frame and releases control. The standalone `/retro` page also works. Choose a game, and start a
new game. Click the screen. Arrow keys move; Z jumps/flies; X is B; Enter is Start;
Shift is Select. The Start button also sends Start. Sound begins after a click.

A second window can connect to the current game using Player 2. Both windows share the Hub operator. Select the player before connecting. Standard gamepad
buttons map to the same controls. A competing controller is refused until explicit
Take this controller; the previous lease then becomes invalid. Focus loss, hidden
tab, gamepad removal and disconnect release control. The last controller leaving
pauses the game. Connect / resume reacquires a new lease. Restart never resumes play.

## Transport contract and limits

This first browser implementation uses same-origin WebSockets rather than the
proposed WebRTC/UDP transport. Measured browser play works through the existing
Hub HTTP path, preserving the emulator's complete network isolation. It avoids
new ICE/UDP exposure and native codec dependencies. It is a LAN prototype, not a
claim of equivalent WAN latency or congestion handling to WebRTC.

The native child copies video before the callback expires, strips row padding,
and copies stereo s16le PCM. Media is transient. Every two emulated frames it
tries a bounded two-item media queue; full queues drop packets without blocking
the controller loop. A separate API task compresses packets off the event loop.
Only the latest compressed packet is retained. At most two viewers exist; sends
have a 250 ms deadline. Paused disconnects release viewer slots.

Wire v1 is zlib of: four-byte big-endian JSON length, UTF-8 JSON, packed video,
stereo s16le audio. Header: version, native frame, width, height, libretro format
(0=1555, 1=XRGB8888, 2=565), video_bytes, sample_rate, cumulative sample_end,
audio_frames. Maximum geometry 512×512; maximum PCM chunk 65536 bytes. Current NES
core yields 256×240, ~60.10 emulated FPS, ~30.05 delivered frames/s. Browser keeps
one decode in flight plus one latest packet. Audio gaps discard queued sources;
playout scheduling stays within 120 ms. Associated video is presented at the
packet's audio start. This is a bounded scheduling policy, not measured A/V drift.

The separate input socket preserves one upstream connection per browser. Each
valid input is acknowledged after its native frame. Hub never multiplexes owners.
No changes to global schemas, bus channels, model metrics or autonomy loops.

## Evidence and acceptance

```sh
python -m pytest tests/retro services/orion-retro/tests -q
python -m pytest services/orion-hub/tests/retro -q
PYTHONPATH=. python services/orion-hub/evals/retro_browser.py --output /tmp/retro-browser-check
```

Install Hub npm dependencies first; the eval uses Puppeteer. Use a venv with Retro
requirements and Hub's aiohttp. The eval builds no image: run the safe Retro build
first. It starts its own isolated native container plus the actual Hub router,
rendered template and static file in a local test server, then removes its container.
CI uses only the original authored Pad Lab ROM. An explicit `--catalog` and
`--game spacegulls` exercise the operator-owned game; no ROM is uploaded to CI.
Reports contain rendered pixel diversity, nonzero PCM scheduled through Web Audio,
applied keyboard and synthetic standard-gamepad actions, blur pause, reconnect,
and cancellation of a delayed resume after blur. A screenshot is retained.

Local Spacegulls check passed beyond 1,850 native frames with visible gameplay and
nonzero scheduled audio. The real-container fault eval still exercises worker
crash, quota, TTL, takeover ordering, and egress denial. **UNVERIFIED:** physical
controller, actual speaker output, user-device input-to-photon p95 <=100 ms and
30-minute actual-client A/V drift <=80 ms. Automated PCM scheduling is not proof
of sound heard at speakers. These acceptance gaps remain explicit before claiming
full P2 performance qualification.

Disable: set `HUB_RETRO_ENABLED=false` and recreate Hub through the wrapper. This
closes browser sockets; lease expiry pauses play. Preserve catalog and evidence.
Do not remove operator artifacts or downgrade to a save-capable core implicitly.

### Input latency and reconnect regression

Inputs are sent every 50 ms independently of acknowledgment arrival. A browser
allows at most 24 outstanding inputs, 4 KiB of WebSocket send buffering, and one
second without acknowledgment of its oldest input; crossing a bound releases
control. The server's 250 ms input expiry remains unchanged. Negative replies for
validated input include port/epoch/sequence so old replies cannot clear a newer
lease. Repeated Connect / resume reuses an open input connection. An expired,
unleased connection closing does not cancel another connection's resume grace.

The browser eval's `--ack-delay-ms 350` variant reproduces the original stall with
the old client. The corrected client continues applying input; deliberately
suppressing every acknowledgment still stops play within the bounded window.
This checks delayed replies, not arbitrary network loss or user-device latency.

The Retro input socket reads continuously; each input waits for its native frame
in its own task. Before 2026-10-20 the socket read the next input only after the
previous acknowledgment. One slow acknowledgment (measured up to 217 ms live)
then held newer input past the 250 ms lease, and the game paused mid-play
(`no_controller` in evidence). An older input that a newer one replaces before its
frame is answered `superseded`; the browser counts that as resolved, not as lost
control. Regression: `test_slow_ack_does_not_gate_reading_the_next_input`.
