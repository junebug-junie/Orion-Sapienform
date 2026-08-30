# Whisper-TTS: fix the dead smoke script and the silently-dropped `options.speaker`

Branch: `fix/tts-smoke-and-speaker-precedence`

## Summary

Two real bugs found by actually using the service on circe after the P100 move,
plus the docs correction that move earned.

- **`scripts/smoke_xtts.py` crashed on import, in the container, always.** The
  documented invocation could never have worked.
- **`options.speaker` was silently discarded** whenever
  `TTS_DEFAULT_SPEAKER_WAV` was set — the live config on every host — so there
  was no way to request a built-in XTTS voice over the bus at all.
- README now documents the resolve order truthfully, and records how circe was
  actually deployed (shared checkout, **not** a worktree) and the two env traps
  that bit during it.

## Outcome moved

A voice-quality A/B against the built-in speakers is now possible through the
normal bus path. Before this it was not — which is exactly the comparison the
whole voice-cloning effort needed in order to be judged, and it had to be done
by executing a hand-written script inside the container instead.

And the service's only smoke test runs.

## Bug 1 — `smoke_xtts.py` dead on arrival

`scripts/smoke_xtts.py:15` did `REPO_ROOT = SERVICE_ROOT.parents[1]`
unconditionally. In the container `SERVICE_ROOT` is `/app`, whose only parent
is `/` — so `parents[1]` raises `IndexError` before any TTS code runs. The
comment three lines above even states the container layout (`/app = service
root`), then indexes past it.

Every documented invocation hit this. README:

```bash
docker compose exec whisper-tts python3 scripts/smoke_xtts.py
```

Verified before and after **in the running circe container**:

```text
=== BEFORE (shipped file) ===
  File "/usr/lib/python3.10/pathlib.py", line 530, in __getitem__
    raise IndexError(idx)
IndexError: 1

=== AFTER (fixed file) ===
Wrote /tmp/smoke_fixed.wav metadata={... 'synthesis_ms': 2338, 'gpu_enabled': True}
```

Fix: only add a repo root when one exists, i.e. when running from a checkout
rather than from inside the image.

## Bug 2 — `options.speaker` silently dropped

`resolve_synthesis_plan` popped `speaker` from options and then never gave it
precedence over `cfg.tts_default_speaker_wav`. Since the kwargs builder prefers
`speaker_wav`, an explicit request for a named voice was discarded — no error,
and nothing in the returned metadata indicating it had been ignored.

Because `TTS_DEFAULT_SPEAKER_WAV` is set on every host running a cloned voice,
the practical effect was: **you cannot request a built-in speaker over the bus,
at all.** Found 2026-08-30 while trying to A/B the new cloned voice against
`Ana Florence`; the comparison sample had to be produced by a bespoke script
`docker exec`'d into the container.

New order: `options.speaker_wav` > `options.speaker` > `TTS_DEFAULT_SPEAKER_WAV`
> `voice_id` > `TTS_DEFAULT_SPEAKER`.

**Deliberately narrow.** This branch can regress nothing, because the value it
now honours was previously discarded outright — no caller can depend on it
being ignored. `voice_id` is intentionally *not* re-ranked in the same patch:
it is already routed and Hub sends it, so moving it above the host default
would change live behaviour rather than un-break dead behaviour. That asymmetry
is documented in the README rather than left as a surprise.

## Files changed

- `services/orion-whisper-tts/scripts/smoke_xtts.py`: guard the repo-root path.
- `services/orion-whisper-tts/app/tts.py`: `options.speaker` precedence.
- `services/orion-whisper-tts/tests/test_tts_voice_resolution.py`: two tests.
- `services/orion-whisper-tts/README.md`: corrected resolve order; how to
  request a built-in speaker; the real circe deploy procedure and its two env
  traps.
- `config/metrics/metric_definitions.lock.json`: routine per-branch re-lock.

## Docs correction: how circe was actually deployed

The README previously told you to deploy circe **from a worktree**, per
AGENTS.md §8. That is wrong for a long-running service and was corrected after
Juniper caught it: a container whose compose project points into a branch
worktree dies the moment that worktree is pruned, and
`make prune-merged-worktrees` will remove it, taking the compose context and
the gitignored `.env` with it. The §8 guard exists to stop dev agents
clobbering each other's uncommitted work — it is not a statement about
production topology. Persistent services deploy from the shared checkout on
clean, merged `main`.

This happened for real on circe: the first container came up with
`workdir=/mnt/scripts/Orion-Sapienform-whisper-tts-deploy/...` and was
recreated from the shared checkout.

Two env traps recorded in the same section, both hit live:

1. **Never `cp` a service `.env` between hosts.** athena's carried
   `PROJECT=orion-athena`, and the service `.env` is the *last* `--env-file`,
   so it silently overrode circe's root `.env` — the container came up on circe
   named `orion-athena-whisper-tts`, no error.
   `services/orion-vision-host/docker-compose.circe-qwen.yml`'s header
   documents this identical trap.
2. **A key missing on the new host is equally a divergence.** circe's root
   `.env` lacks `ORION_BUS_VELOCITY_TRACKING_ENABLED`, which `.env_example` and
   athena set to `true`; it arrived blank (= false) with only a compose
   warning. No env-parity gate catches this, because they all compare against
   `.env_example` on a single host.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: `options.speaker` is now honoured when it was previously
  discarded. No other resolve-order rank moved.
- Compatibility: no caller can regress, per the reasoning above.

## Env/config changes

None. (The circe `.env` corrections are host-side, gitignored, and already
applied.)

## Tests run

```text
cd services/orion-whisper-tts && PYTHONPATH=. python -m pytest tests -q
  59 passed        (was 57 on main)
```

Mutation check on the new precedence branch — a green from a new test is not
evidence until it has been seen to fail. Disabling it (`elif speaker:` ->
`elif False and speaker:`):

```text
FAILED tests/test_tts_voice_resolution.py::test_request_speaker_beats_default_speaker_wav
1 failed, 10 passed
```

`app/tts.py` restored and confirmed clean afterwards.

The smoke-script fix is verified live in the container (above) rather than by
unit test — a container-layout bug is not reproducible from a checkout, which
is precisely why it survived this long.

## Evals run

```text
none -- services/orion-whisper-tts still has no evals/ directory.
```

## Docker/build/smoke checks

```text
docker exec orion-circe-whisper-tts python3 scripts/smoke_xtts_fixed.py
  -> Wrote /tmp/smoke_fixed.wav ... synthesis_ms=2338 gpu_enabled=True
```

Live service state on circe after the P100 move (unchanged by this PR):

```text
workdir   = /mnt/scripts/Orion-Sapienform/services/orion-whisper-tts
DeviceIDs = ["4"]                      (Tesla P100-PCIE-16GB)
PROJECT   = orion-circe
health    = ok, bus connected, cuda_available true
synthesis = 2940ms  (vs ~3300-3900ms on athena's P4)
```

## Review findings fixed

Not yet reviewed — opening for review now.

## Restart required

For the `options.speaker` fix to take effect on circe, the image must be
rebuilt and the container recreated:

```bash
cd /mnt/scripts/Orion-Sapienform
docker compose --env-file .env --env-file services/orion-whisper-tts/.env \
  -f services/orion-whisper-tts/docker-compose.yml up -d --build
```

Not required for the smoke-script fix alone unless you want it in the image.

## Risks / concerns

- **Severity: low.** Concern: `voice_id` still ranks below
  `TTS_DEFAULT_SPEAKER_WAV`, so `{"voice_id": "Ana Florence"}` on a cloning
  host still silently synthesizes with the clone and echoes the id back in
  metadata. Mitigation: documented explicitly, and `options.speaker` now
  provides the working path. Re-ranking `voice_id` is a live-behaviour change
  and deserves its own patch.
- **Severity: low.** Concern: this service still has no `evals/`, so voice
  quality remains ungated. Carried over, not introduced.

## Status

DONE_WITH_CONCERNS.
