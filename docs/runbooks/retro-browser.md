# Retro browser play

Hub `/retro` provides keyboard/gamepad play without an additional access code. The native
core remains in the isolated Retro container. No ROM or core runs in the browser.
No images, audio or inputs enter memory, chat or cognition. No save/load API exists.

## Scripted baseline

After starting the game and connecting a controller, open More and choose
**Run scripted baseline · 60s**. It transfers only your current port to the
visibly labeled `scripted-right-jump-v1` baseline. It repeats right/jump;
it does not perceive, learn, or demonstrate Orion's game competence.
**Take this controller** interrupts it with the existing neutral-frame handoff.
Pause, leaving/closing the sponsoring browser, a missed policy deadline, or the
60-second budget stops it. Resume returns human control, never restarts the script.
Status lists active scripted ports; the evidence journal attributes applied
actions by policy ID and epoch. No cognition or memory writes occur.

The owner-scoped service API also accepts a 5–1800 second budget. Start requires
the live input connection, generation, port, and `expected_epoch` of the human
lease being transferred. Stop names the current script epoch. Neither operation
implicitly starts or resumes a game. There is no arbitrary-code upload.
See [remaining original-spec gates](../superpowers/specs/2026-10-20-retro-continuation-status.md).

## Setup

For a new installation, from clean primary main after merge, sync templates and
prepare private credentials. **Do not rerun `install_catalog.py` on an existing
custom catalog:** it reconstructs the kit catalog and can remove imported entries.
An existing installation should preserve its catalog and credentials.

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
tries a fixed-size single-packet shared slot; a busy/full slot drops packets without blocking
the controller loop. A separate API task compresses packets off the event loop.
Slot locks are acquired without waiting. If a worker dies mid-copy, the API
discards the abandoned slot with that session rather than blocking on a partial pipe packet.
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
control. The server's lease expiry is 2 s (see Lease expiry below). Negative replies for
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
Trade-offs: during a stall, a press and release that both arrive before one
frame collapse to the release. Input from both browsers shares the API's 24-command
cap, so a stall longer than about 0.5 s can return `unavailable_or_busy` (shown as
Control interrupted). Timeout cleanup is shielded from socket cancellation
(`test_cancelled_input_cannot_abandon_timeout_cleanup`).

### Grant grace

A fresh lease waits 1 s for its first input; after that, each input still expires
after 250 ms. Before 2026-10-20 the 250 ms clock started at the grant, before the
grant reply had crossed Hub and Tailscale to the browser. Every Connect / resume
from a remote laptop then lost control with zero applied inputs. A granted lease
holds the neutral mask, so the longer first window cannot hold a button down.
Regression: `test_fresh_lease_waits_for_first_input_across_the_grant_round_trip`.

### One-button play

The page shows the game picker, New game and Mute. Player choice, Connect /
resume, Pause, Stop, Start and Take this controller are under More. After New
game (or a click on the game), lost control heals itself: up to three reconnects
in 10 s, then the page asks for New game. Leaving the window still releases
control and cancels pending reconnects; returning focus or clicking the game
reacquires it. Pause and Stop turn self-healing off until the next click on the
game. The browser eval's delayed-ack variant now asserts that play resumes after
acknowledgments return, with no Connect click.

### Lease expiry

A lease now expires after 2 s without input (`LEASE_TTL`), at grant and after
each input. Live 2026-10-20, after #2772: on each grant the first input arrived
about 500 ms later, and the second never arrived within 250 ms. The browser ran the
Retro page's 50 ms timer at about 1 Hz, as browsers do for pages they treat as
background. Key presses and releases are still sent at once. Socket close,
blur and hidden tabs still release control immediately. The cost: a browser
that freezes without closing its socket can hold its last buttons for up to 2 s.
Regression: `test_lease_survives_throttled_browser_heartbeat`.

## Dedicated relay (hub-retro-gateway)

Game traffic (`/api/retro/*`) runs in its own small process, `hub-retro-gateway`
(`services/orion-hub/scripts/retro_gateway.py`, loopback port
`HUB_RETRO_GATEWAY_PORT`, default 8092). It uses the same router, origin check and
credentials as Hub, and it starts in under a second. Hub keeps its own copy of the
routes, so play still works through Hub until the route below exists.

Why: Hub's single event loop also serves every open Hub page. A/B with
`retro_play.mjs` on 2026-10-20, with a Hub page open in another tab:
- Through Hub: 55 inputs reached the server, acknowledgments took 720 ms p50, and
  play dropped 4 times.
- Through the relay: 944 inputs, 17 ms p50, no drops.

One-time route (operator, on athena; tailscale serve strips the mount path):

```sh
sudo tailscale serve --bg --set-path /api/retro http://127.0.0.1:8092/api/retro
```

Check: `tailscale serve status` lists `/api/retro`, and
`curl -s http://127.0.0.1:8092/api/retro-gateway/health` returns `{"status":"ok"}`.
Undo: `sudo tailscale serve --bg --set-path /api/retro off`.

The Retro tab also unloads Hub's other embedded pages (e.g. the `/spark/ui` WebGL
visualizer) while it is open. That visualizer used ~22% of the shared browser
thread. Leaving Retro reloads those pages, so their in-page state (scroll,
filters, a loaded Gephi graph) starts fresh. New game ends the existing game
instead of failing with `session_active`. Every Hub window shares one Hub
credential, so New game in one window also ends a game started in another; saves
are off. The relay has its own 8-socket cap, separate from Hub's; Retro itself
still allows 2 input sockets.
A paused game's inputs now answer `session_paused:<why>`, so the page reconnects
after a lapsed lease, a dropped controller or an overload pause (at most 3 times in
10 s). It honors an explicit Pause, including one pressed on an already auto-paused game.

## Play harness (`evals/retro_play.mjs`)

Plays like a person: real Chromium, real Hub tab, scripted Mario input. It records
input cadence, acknowledgment latency, socket closes, status text, main-thread long
tasks and the server journal.

```sh
node services/orion-hub/evals/retro_play.mjs --url https://athena.tail348bbe.ts.net \
  --hub --game super-mario-bros --seconds 30 --cpu 4 \
  --evidence-dir ~/.local/share/orion/retro/service/evidence --out /tmp/retro-play
```

- `--hub` plays inside Hub's tab (iframe); omit it for standalone `/retro`.
- `--cpu N` throttles the renderer.
- `--override /static/js/x.js=local/file.js[,...]` serves working-copy files in
  place of the deployed ones, so client fixes can be tried against live without a
  deploy.

`--url` is required. A run starts a real game on that service and ends any game
already running there, so do not run it while someone is playing.

### Quick taps

A newer input that replaces a pending one before the next frame no longer erases
the older one's buttons: they are held for exactly one frame (`latch` in
`app/control.py`). A newer latched direction replaces an older one, and a latched direction yields to
the current opposite one. A `superseded` reply therefore no longer means "never
reached a frame": its buttons are held for that one frame.
Live 2026-10-20: on Crystalis's title screen, 1 of 4 quick Start taps reached the
game; with the latch, 4 of 4 did and the intro started. Regression:
`test_tap_inside_one_frame_still_reaches_the_game`. `retro_play.mjs` now taps
Start 4 times and reports `start_presses_applied` from the server journal.

### Start readiness

A command (such as the browser's resume right after New game) first waits for the
worker to report ready, up to the 10 s start limit. Then its 0.75 s command
timeout starts. Before 2026-10-20 the command clock started while the core was
still loading (~0.48 s measured). Nova then failed at frame 0 with
`worker_unavailable`. With the wait, all 6 catalog games pass `retro_play.mjs`
(4/4 Start taps, no drops). Regression:
`test_command_waits_for_a_starting_worker_instead_of_failing_the_session`.


### Optional goal experiment

`Try Orion goals · 60s` requires default-off `RETRO_COGNITION_ENABLED` (Retro)
and `HUB_RETRO_COGNITION_ENABLED` (Hub/relay). It hands off only the currently
owned port. The UI displays goal/rationale/outcome; `Take this controller`,
Pause, blur and disconnect retain priority. Five-second motor goals expire to
wait if background planning is unavailable. The gateway uses GPU-pool background
admission; thermal refusal is not an emulator failure and must not be bypassed.

See the [goal-loop implementation record](../superpowers/specs/2026-10-20-retro-goal-loop-implementation.md)
for the fingerprinted Nova observation scope, unknown cases, local evidence,
and the real-planner acceptance gate. No deployment or game restart is implied.

## Zelda play: planner v2, play panel, solo leisure

Spec: [Orion plays The Legend of Zelda](../superpowers/specs/2026-10-22-orion-plays-zelda-design.md)
(Hub half: sections D, E and the Hub view). Needs the Retro-side Zelda reader,
navigator and atlas (Lane A) to be deployed; until then the Hub code is idle.

- **Planner v2** (`scripts/retro_planner.py`, `plan_zelda`). One request per Retro
  decision (room entry, goal finished, death, stuck), at most one in flight per
  process, local GPU route `metacog_background` only. The prompt holds a 22x32
  text map of the room (Link, enemies, drops, caves, learned-blocked tiles; walls
  only when Retro includes the grid in its offer, otherwise the map says so), the
  atlas neighbourhood and the last 5 decisions with their verdicts (the Hub keeps
  them from Retro's `last_finished`). A parse failure, an unsupported goal or an unknown observation
  produces no goal; Retro's motor holds `wait`. A heat shed returns
  `503 {"detail":"gpu_pool_unavailable","reason":"shed:cabinet_hot","retry_after_sec":N}`
  and the shared `orion/gpu_pool/shed_backoff.py` rule spaces the retries.
- **Play panel** (`/retro`, "Orion's map and goal"). The atlas grid (16x8 rooms;
  bright edge = exit Orion used, `c` = cave, `×` = deaths, outline = current room),
  the current goal with its reason and prediction, and the last 5 verdicts. During
  a shed it reads "Orion is waiting: cabinet hot". A solo session is shown live
  when the page opens, by watching only (media socket); clicking the game or
  `Take this controller` acquires a controller, which preempts Orion.
- **Solo leisure** (`scripts/retro_leisure.py`). Started only by the allocator's
  `play` act (Lane C calls `should_start(config)` and then
  `run_leisure_session(config, act_id=...)`). Doer gates, checked at start and every
  30 s, each with a named reason in the journal: arousal known and not strained,
  no `cabinet_hot`, no relay socket open and no human Retro action in 10 min, no
  game already running, one start per 3 h, 90 min per rolling 24 h, 30 min per
  session, `RETRO_LEISURE_ENABLED=true` (the rollback lever). Ends after 3
  consecutive sheds, or earlier (`shed_wait_policy_lost`) when Retro drops the
  policy after 60 s with no goal; the paused orphan game is then closed. Loads the newest compatible save or starts fresh, and saves
  at the end. When a human may be in control (they acted, or presence cannot be
  read), it never saves over or stops their game: it only releases its port.
- **Shared state** (`scripts/retro_presence.py`, Redis on `ORION_BUS_URL`):
  `orion:retro:presence:human_at`, `orion:retro:presence:browser:<id>` (15 s TTL),
  `orion:retro:leisure:{status,ledger,journal,lock}`. Read the journal with
  `redis-cli -u $ORION_BUS_URL lrange orion:retro:leisure:journal 0 20`.
- **Browser eval** (fixture Retro, real template/asset/router):
  `python services/orion-hub/evals/retro_play_panel.py --output /tmp/retro-play-panel`
  (set `RETRO_PUPPETEER_PACKAGE` to a `package.json` with puppeteer installed if
  the local `node_modules` is incomplete).
