# Retro P1 deployment and verification

See [service README](../../services/orion-retro/README.md) for install, auth,
endpoints, limits and exact operator smoke. The older
[diagnostic kit](retro-local-setup.md) remains separately runnable and disposable.

The new service has a compose definition, so the pull-hook classifier can select
`orion-retro` on a service code change. Local catalog and credential must be
provisioned before the first deployment; a git pull alone cannot supply private
local artifacts. The deterministic installer consumes existing authorized kits.
Do not expose the socket directory to other users or route it through Hub yet.

Expected evidence for a successful deployment:

1. Clean main equals origin/main; safe wrapper build/up succeeds.
2. `orion-athena-retro` runs with read-only root, non-root user, dropped
   capabilities, resource limits and disabled network; only private Unix-socket API.
3. Authenticated readiness distinguishes idle from a running session. Unauthenticated
   requests and browser Origin requests are denied.
4. Operator smoke returns actual applied frame numbers, increasing callbacks,
   then paused on disconnect and stopped on explicit stop.
5. Authored-ROM pad evaluation proves both physical controller ports and nonzero
   audio; preflight records actual core/ROM hashes and strict replay outcome.
6. Worker crash/timeout, expired lease, queue pressure and storage failure produce
   bounded failure, never a silently continuing game or a completed evidence claim.

Spacegulls is an operator-local download from
[its publisher](https://morphcatgames.itch.io/spacegulls), version 1.1, SHA-256
`b69bd1809e26400336af288bc04403c00d77030b931bc31875594c9a0ae92f67`.
The publisher advertises 1–2 players. A local checkpoint experiment verified independent visible effects: P1 moved the
first bird; P2 A/right joined and moved a second bird while the first remained
in place. Nonzero PCM peak was 8172. This is not a sustained cooperative-play
acceptance test or a browser latency measurement.
ROM redistribution rights are not asserted and the ROM is not bundled.

Nova remains available only for fresh starts: strict state equality fails on the
pinned core despite matching video. All P1 save/load APIs are disabled, including
for Spacegulls; the implementation does not weaken the diagnostic replay gate.

P2 remains unimplemented: Hub auth integration, WebRTC audio/video, browser
keyboard/gamepad handling, input-to-photon measurement and a 30-minute interactive
session. P3 policy/cognition is also unimplemented. This service does not claim
Orion can competently play or learn from either game.

P1 packaging decision: an internal Docker network on this host did not publish
its API port. P1 therefore uses `network_mode: none` and an operator-only Unix
socket directory instead of the design's proposed loopback TCP API. Hub P2 must
proxy this socket; it cannot assume a TCP port or reuse the chat WebSocket.
