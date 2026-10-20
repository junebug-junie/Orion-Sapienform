## Summary
Remove Retro's extra access-code login so opening the Hub tab loads the game picker directly. Retro uses Hub's existing network access boundary.

## Outcome moved
No code entry or lock button; configured-origin checks still protect mutations and WebSockets. Private Hub-to-emulator credentials remain server-side.

## Current architecture
Hub proxies browser control/media over a private Unix socket to the isolated native emulator.

## Architecture touched
Only the Retro browser adapter and its credential setup helpers; no bus or cognition changes.

## Files changed
Hub router, template, JavaScript, boundary tests/browser eval; Retro setup helpers and runbooks.

## Schema / bus / API changes
Removed /api/retro/login and /logout. Existing game/session APIs no longer require a browser cookie. Exact-origin POST/WS checks remain.

## Env/config changes
No env keys changed. browser.code is no longer created or required; existing files are unused.

## Tests run
Focused Hub and Retro regression suites.

## Evals run
Real Chromium/native emulator browser eval with operator-provided Super Mario Bros.; no ROM committed.

## Docker/build/smoke checks
Retro restarted from synchronized clean main; live catalog includes Tetris, Crystalis and Super Mario Bros. Hub deploy follows merge.

## Review findings fixed
Independent review caught the browser eval's obsolete 401 expectation; updated to 200.

## Restart required
Rebuild/recreate Hub from clean main after merge.

## Risks / concerns
Anyone who can reach Hub can control Retro, as requested. Physical controller and user-device audio/latency remain unverified.

## PR link
Added on creation.
