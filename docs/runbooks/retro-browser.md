# Retro browser play

Hub `/retro` provides a scoped operator login and keyboard/gamepad play. The native
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
play, but does not encrypt the login/session traffic; gamepads require a secure
browser context (HTTPS or localhost). Do not expose this scoped operator interface
on an untrusted HTTP network. Existing Hub routes do not gain authentication.

The installer creates a distinct backend `hub.token` and browser `browser.code`
in `~/.local/share/orion/retro/service/hub-auth/`, each mode 0600, directory 0700.
Read `browser.code` locally and enter it on `/retro`; never use the backend or CLI
credential in the browser. Hub stores up to eight random HttpOnly, SameSite=Strict
sessions for eight hours, in memory; restart requires login again. The cookie is
Secure for HTTPS origins. Logout revokes its sockets; neither code is logged or
placed in a URL. Login attempts are limited to 20/minute across this single operator
interface. This is one fixed `hub_operator`, not a multi-user account system.
Rotate `browser.code` and restart Hub to revoke all browser sessions.

Hub mounts only the dedicated Unix socket directory and Hub credential directory.
The emulator gets only `hub.token`, not the browser code. Its network remains `none`.
CLI and Hub credentials cannot access each other's session or evidence. One game
session still exists at a time. An active CLI session must be stopped by its owner
before a browser can start a game. No browser-supplied principal is trusted.

## Use

Open `/retro` from Hub's Retro games link. Unlock, choose Spacegulls, and start a
new game. Click the screen. Arrow keys move; Z jumps/flies; X is B; Enter is Start;
Shift is Select. The Start button also sends Start. Sound begins after a click.

A second window can connect to the current game using Player 2. Both windows use
the scoped operator login. Select the player before connecting. Standard gamepad
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
