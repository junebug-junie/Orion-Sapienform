# Local NES diagnostic kit

This installs the missing prerequisites for design #2751: a pinned FCEUmm NES
core, an original two-controller test ROM, a restricted runtime, and reproducible
evidence commands. It does not install a commercial game, browser player, learned
policy, SNES core, or the later Retro service.

## One-time setup

From a checkout containing this patch, as the ordinary Docker-enabled operator:

```bash
python3 -m orion.retro.setup_local
python3 -m orion.retro.local_runner pads
python3 -m orion.retro.local_runner smoke
python3 -m orion.retro.local_runner baseline
```

The first command fetches the public source, cleans build artifacts and recompiles with at most two jobs,
generates the original ROM, retains licenses, pulls the digest-pinned runner,
and publishes `~/.local/share/orion/retro/manifest.json`. Prerequisites are Python
3.11+, `git`, `make`, `cc`, and Docker; no sudo or host package installation.
The default setup was exercised on Linux x86_64. A core built against a newer
host libc may be incompatible with the runner: the smoke must pass on each host.
There is no silent fallback to executing the core on the host.

After installation, the kit works independently of the checkout:

```bash
python3 ~/.local/share/orion/retro/run.py pads
python3 ~/.local/share/orion/retro/run.py smoke
python3 ~/.local/share/orion/retro/run.py baseline
```

The generated launcher uses an immutable, versioned copy of the Python runtime
under the kit root, selected through the atomically published manifest. Removing a development worktree does not break the kit.

Artifacts:

- `cores/fceumm-ac0d00a7.so`: locally built core; its exact SHA-256 is in the manifest.
- `roms/orion-pad-lab-v1.nes`: self-authored mapper-0 diagnostic, generated from source.
- `sources/fceumm-ac0d00a7/`: pinned source, retained for inspection/license compliance.
- `runtime/<digest>/` and `run.py`: installed code and launcher.
- `licenses/`: upstream GPL-2.0 and original-ROM MIT terms.
- `evidence/<run-id>/`: result, progress, manifest and actual container configuration.

Use `--root /private/path` on both commands to change the install root. Installation
is idempotent for identical artifacts and refuses to overwrite differing core/ROM
bytes. Existing ROMs are not deleted. Source must be clean at the pinned commit.
Do not copy compiled cores or generated ROMs into Git.

## Container boundary

Emulation runs in a disposable container, not on base Athena. The *source build*
uses the existing host compiler; the resulting shared library is loaded only in
Docker. No system emulator is installed and no existing services are restarted.

The runner enforces non-root UID/GID, no network, read-only root filesystem,
all capabilities dropped, no-new-privileges, 64 PIDs, 256 MiB RAM with no extra
swap, two CPUs, no core dumps, and a 64 MiB per-file ceiling. It mounts only the
Python probe directory, one core and one ROM read-only, and that run's evidence
directory writable. `/tmp` is a 16 MiB noexec/nosuid tmpfs. No Docker socket,
credentials, database, ROM directory or production storage is exposed inside.

The command records actual `docker inspect` configuration before starting,
limits the baseline to 40 minutes, and stops/removes only its own generated
container on success, error, timeout or interruption. SIGINT/SIGTERM flow through cleanup; SIGKILL or host/daemon loss cannot run Python
cleanup and may leave a named diagnostic container for operator removal.
No host ports or permanent service are created. This is a constrained diagnostic sandbox, not a claim of
proof against all native emulator vulnerabilities.

## Original diagnostic behavior

`orion/retro/pad_lab.py` generates the NROM bytes directly from an original 6502
program and original CHR tile. No downloaded game, BIOS, art or sample is needed.
Pressing any P1 button increments only zero-page counter `$02` and changes the
left panel's palette. P2 owns `$03` and the right panel. Neither input changes the
other counter. A pulse channel emits a tone.

`pad_eval` checks the **exact generated ROM hash** before reading those two RAM
bytes. It restores one checkpoint for neutral, P1-only, P2-only and both-port
trials, checks the separate counters and corresponding framebuffer pixels, and
requires nonzero PCM. Those addresses are not a universal game decoder. No
outcome/progress metric is sent to cognition. Passing this test proves two-port
actuation in the diagnostic; it does not prove any commercial game's co-op mode.

`smoke` requires all timing/replay checks except 30-minute duration. The underlying
report deliberately still says `INCOMPLETE_OR_FAILED` for the full baseline gate;
runner exit 0 means the short smoke checks passed. `baseline` runs 110,000 paced
frames (about 30.5 minutes), then a 1,000-frame save/restore replay, and only exits
0 on full baseline PASS. Progress is emitted every 600 frames; the runner prints
the exact `tail -f` command. Neither mode measures browser/display/audio latency.

## Pins and upstream sources

- [FCEUmm source revision](https://github.com/libretro/libretro-fceumm/tree/ac0d00a7fc01166f84c7bac01d1084ebfd2b865e), GPL-2.0.
- Python runner image: `python@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89` (3.11 slim Bookworm).
- Generated diagnostic SHA-256: `e853994621733bb30053961cc5c88d40b5e44ae3f63ec954361a777d2ae14082`.
- Original ROM/license source: `orion/retro/pad_lab.py`, `orion/retro/PAD_LAB_LICENSE`.

Do not infer a binary hash from the source commit: compiler/libc/build inputs can
change output. The setup records the actual compiler, build command, core digest,
source commit and generator digest. Runner checks both input digests before load.

## Verification and rollback

```bash
python3 -m pytest tests/retro -q
```

CI additionally builds the real pinned core and executes the authored-ROM port
and audio eval. The native ABI fixture remains a distinct unit-test lane.
The initial real-ROM eval caught a PPU nametable/scroll reset bug that a fake core
could not catch; the real-core CI now guards the corrected behavior.

No env template, bus, schema registry, database, service or deployment changes.
No restart required. To disable, stop launching the offline commands; the runner
cleans up its container. Keep the private core/ROM/evidence directory for later
P1 work. Removal of operator evidence is deliberately not automatic.

P0 host-load matrix, alternative runner comparison, SNES validation and subsequent
Hub/control/cognition milestones remain separate work. Host inventory in this
session found two Xeon Platinum 8168 sockets (48 physical/96 logical cores), not
the older speculative Xeon Gold 6138 sizing in the design. Actual headroom must
be measured; logical CPU count is not a capacity guarantee.
