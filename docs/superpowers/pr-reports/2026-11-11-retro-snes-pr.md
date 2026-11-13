## Summary

- Add pinned Snes9x alongside the unchanged NES core; one catalog and one game session.
- Show SNES titles with `(SNES)` in Hub's existing game list.
- Map the 8BitDo Pro 2's X/Y/L/R and keyboard A/S/Q/W; retain NES controls and Select+L2/R2 saves.
- Add an additive ROM importer and preserve operator entries when rerunning the base installer.
- Verify all three operator ROMs through real browser/native-core runs, plus a ROM-free CI diagnostic.

## Outcome moved

Previously only `.nes` catalog entries and NES buttons were accepted. Mixed NES/SNES
catalogs now select the correct core, deliver all SNES buttons and use each game's
core hash for save compatibility. Existing NES entries need no migration.

The actual operator ROMs passed isolated Docker/browser checks on Carbon:

| Game | ROM SHA-256 | Video colors observed | PCM peak (normalized) |
| --- | --- | --- | --- |
| A Link to the Past (USA) | `66871d66be19ad2c34c927d6b14cd8eb6fc3181965b6e517cb361f7316009cfb` | 5 | 0.3113 |
| Super Mario Kart (USA) | `2ada8919688087be60a6a48cace8f877add60c45d2e5d09e2442faa55be62a49` | 45 | 0.4533 |
| Super Mario World (USA) | `0838e531fe22c077528febe14cb3ff7c492f1f5fa8de354192bdff7137c27f5b` | 26 | 0.2323 |

These are eval observations, not new runtime metrics or cognition signals. ROMs,
screenshots, credentials and save states remain outside git. Production is not yet changed.

## Current architecture

- Service: `orion-retro`, offline native worker; Hub proxies its private Unix socket.
- Entry points: `app.main`, spawned `app.worker`, Hub `retro_routes`/`retro_gateway`, `retro-panel.js`.
- Config: native `settings.py`, service `.env_example`/local `.env`; Hub `scripts/settings.py`.
  Hub has no root `settings.py` (its settings modules live under `scripts/` and `app/`).
- Bus channels / schema registry: no native bus connection; this patch adds no global event.
- Compose: `services/orion-retro/docker-compose.yml`, `services/orion-hub/docker-compose.yml`.
- Tests: `tests/retro`, native `tests`, Hub `tests/retro`.
- Evals: native container/Pad Lab and Hub browser/play-panel harnesses.
- Gap: single NES core/hash, `.nes` catalog restriction, eight-button input and UI.

## Architecture touched

One private HTTP contract gains `system`; Settings selects one of two packaged cores.
Worker authority validates buttons by the running system. The frontend uses session
system, so changing the unstarted dropdown selection cannot remap a running game.
The native process boundary, media protocol, leases and cognition gates remain in place.

## Files changed

- `services/orion-retro/settings.py`, `app/session.py`, `app/worker.py`, `app/control.py`: core selection, identity and allowed buttons.
- `services/orion-retro/Dockerfile`: separate pinned Snes9x build, retained license.
- `services/orion-retro/evals/import_rom.py`, `evals/install_catalog.py`: additive operator import/install.
- `services/orion-retro/tests/test_snes.py`: mixed catalogs, identity, rejection and preservation regressions.
- `orion/retro/snes_pad_lab.py`, `snes_pad_eval.py`: authored SNES joypad diagnostic and real-core eval.
- Hub JS/template/tests/evals: unified labels, full controls, asset v17, browser coverage.
- `.github/workflows/retro-preflight.yml`: run SNES diagnostic in the built container and retain evidence.
- Native/Hub READMEs and `docs/runbooks/retro-browser.md`: contracts, controls and operator setup.

## Schema / bus / API changes

- Added: optional catalog `system` (`nes`/`snes`); additive `system` in games/session responses and start evidence.
- Removed: none.
- Renamed: none.
- Behavior changed: `.sfc`/`.smc` accepted for SNES; worker loads Snes9x and permits bits 0–11.
- Compatibility notes: missing `system` means NES. Save records keep the existing schema and exact ROM/core hashes.
  No bus or global schema registry update is needed for this private HTTP/catalog contract.

## Env/config changes

- Added/removed/renamed keys: none.
- `.env_example` updated: no.
- local `.env` sync: no template change; ignored Carbon build configuration created from the existing template.
- skipped keys requiring operator action: none.
- Local catalog: live Athena has 20 NES entries; planned additive import produces 23 total.
- No bus URL is introduced or changed; native Retro remains network-disabled.

## Tests run

```text
PYTHONPATH=/tmp/retro-pytest-deps:. python3 -m pytest tests/retro services/orion-retro/tests -q
182 passed; follow-up SNES regression suite after review: 16 passed
PYTHONPATH=/tmp/retro-pytest-deps:. python3 -m pytest services/orion-hub/tests/retro -q
240 passed, 2 skipped (existing optional local coverage)
node --check services/orion-hub/static/js/retro-panel.js
node --check services/orion-hub/evals/retro_browser.mjs
python3 scripts/check_env_template_parity.py orion-retro: PASS
git diff --check: PASS
```

The suggested `scripts/check_schema_registry.py` and `scripts/check_bus_channels.py`
do not exist in this checkout. No bus/global schema changes are made. Carbon has no
`graphify` executable or `/mnt/storage-warm` graph store; read-only mapping unavailable,
no graphs modified. Git worktree/file collision checks were used.

## Evals run

```text
Authored SNES Pad Lab against packaged Snes9x: PASS, 24 button/port checks + restore/video + callbacks
Hub real browser + real native core: PASS for A Link to the Past, Super Mario Kart, Super Mario World
Hub NES Pad Lab browser regression: PASS
Hub play-panel fixture browser: PASS
Native service container lifecycle/fault/offline smoke: PASS
```

Each game browser run verifies mixed list labels, nonempty video, nonzero PCM,
keyboard and simulated standard-pad extra buttons, saves, in-place and fresh-session
loads, focus loss and reconnection. Two-port games exercise P2. Actual game progress,
physical controller input and speaker output are not inferred from these checks.
Evidence is under `/tmp/orion-retro-snes/{browser-kart,browser-world,browser-zelda,browser-nes,play-panel,service-smoke}`
and `/tmp/orion-retro-snes/pad-eval.json` on Carbon.

## Docker/build/smoke checks

`scripts/safe_docker_build.sh orion-retro build` passes on Carbon. Disposable eval
containers use no network, read-only roots and non-root users. The native core hashes:

- NES: `b8ba56fbeba5bf7658da89fba97ae1049eafe583320db490c2bf891934bdda22`, identical to live Athena.
- SNES: `8d7679e1bf88bb3ae2de0f707ebd09551b7232d9714438f6122aaab4b7b31cf3`.

Snes9x source pin: `1bcc369e89f08243e0a462882fb1f3e42e51de3a`.

## Review findings fixed

- Finding (P2): preserving operator imports could let the base installer publish
  more than 64 games and make the next service startup fail.
  - Fix: refuse adding a kit when the catalog is full, before publishing JSON.
  - Evidence: regression test seeds 64 entries and confirms a refused 65th leaves
    the existing catalog bytes intact.
- Review-agent verified the fix and returned no findings on the final patch.

## Restart required

Pending Juniper's production confirmation under AGENTS.md section 13: merge this PR,
copy the three extracted operator ROMs from Carbon to Athena, append them using the
importer, then rebuild/recreate Retro and Hub from clean synchronized primary main.
Save/stop any active game first. Example after merge and source sync on Athena:

```bash
cd /mnt/scripts/Orion-Sapienform
# After staging the extracted files under /tmp/orion-retro-snes-import:
PYTHONPATH=.:services/orion-retro python3 services/orion-retro/evals/import_rom.py \
  --catalog /home/athena/.local/share/orion/retro/service/catalog/catalog.json \
  --rom '/tmp/orion-retro-snes-import/Super Mario Kart (USA).sfc' \
  --game-id super-mario-kart --title 'Super Mario Kart' --players 2
PYTHONPATH=.:services/orion-retro python3 services/orion-retro/evals/import_rom.py \
  --catalog /home/athena/.local/share/orion/retro/service/catalog/catalog.json \
  --rom '/tmp/orion-retro-snes-import/Legend of Zelda, The - A Link to the Past (USA).sfc' \
  --game-id a-link-to-the-past --title 'The Legend of Zelda: A Link to the Past' --players 1
PYTHONPATH=.:services/orion-retro python3 services/orion-retro/evals/import_rom.py \
  --catalog /home/athena/.local/share/orion/retro/service/catalog/catalog.json \
  --rom '/tmp/orion-retro-snes-import/Super Mario World (USA).sfc' \
  --game-id super-mario-world --title 'Super Mario World' --players 2
scripts/safe_docker_build.sh orion-retro up -d --build --force-recreate
scripts/safe_docker_build.sh orion-hub up -d --build
```

Before catalog publication, back up the existing catalog JSON. Verify 23 games,
actual native session identities, and Hub `(SNES)` labels. Reload Hub to asset v17.
Rollback: stop the SNES session, restore the prior catalog, redeploy the prior NES-only
code; retain all ROMs/saves. Never feed SNES entries to the old NES-only service.

## Risks / concerns

- Production deployment: UNVERIFIED until approved and executed; isolated runtime path is verified.
- Physical 8BitDo Pro 2 / firmware: UNVERIFIED. Prior PRs #2935/#2936 identify the device;
  Carbon currently reports no connected USB or Bluetooth controller. Hub mapping does not flash it.
- SNES cognition/game competence: out of scope; no observer, atlas or leisure eligibility added.
- Snes9x's upstream license is preserved; ROMs remain private operator artifacts.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3203
