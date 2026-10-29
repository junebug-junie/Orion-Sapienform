#!/usr/bin/env python3
"""Hold circe's per-card GPU power limits at the checked-in values (config/gpu_power_limits.json).

Design: docs/superpowers/specs/2026-10-29-gpu-power-cap-design.md. Runbook (install, check,
rollback): docs/runbooks/2026-10-29-circe-gpu-power-limits.md.

A card that is not listed in the config is never touched. For each listed card:
  apply   set the target limit if the card is not already at it, and log the change. The timer runs
          this every 5 minutes, so a reboot, driver reload or hand-typed `nvidia-smi -pl` is undone.
  remove  put the card back at its default limit. It waits (up to REMOVE_WAIT_S) for a check or a
          bench that holds the card's lock, and fails loudly rather than skip: a reset must not be lost.
          It is the service's ExecStop; the timer is PartOf the service, so stopping the service stops
          the checks too, and `uninstall` runs it explicitly.
  status  one JSON line per card: target, current, default. Writes nothing.
A target outside the card's [min, max] range, or a config written for another host, is refused
and the card is left alone. `"watts": "default"` holds a card at its default limit: the way to
un-cap a card through a PR alone (a card simply deleted from the file is no longer managed, and
keeps whatever limit it had). While scripts/gpu_power_bench.py holds its per-card lock, the card is
skipped; this job holds that same lock while it sets a limit, so the two never interleave.

Usage (as root on circe), one line from athena:
    ssh -t circe@circe sudo /mnt/scripts/Orion-Sapienform/scripts/ops/gpu_power_limits.py install
  install   copy this script to /usr/local/sbin and the units to /etc/systemd/system, enable the
            service and the timer, apply now, print status
  uninstall stop and disable both (the cards go back to their defaults), delete the installed copies
DRY_RUN=1 prints install/uninstall commands instead of running them (used by the tests).

Stdlib only: it runs with the host's python3 as root, outside any Orion venv. The installed copy
in /usr/local/sbin is what root runs; a `git pull` changes the config it reads, never its code.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

CONFIG = Path(os.environ.get("ORION_GPU_POWER_LIMITS_CONFIG",
                             "/mnt/scripts/Orion-Sapienform/config/gpu_power_limits.json"))
LOCK_DIR = Path(os.environ.get("ORION_GPU_POWER_LIMITS_LOCK_DIR", "/run"))  # the bench's --lock-dir
SMI_TIMEOUT_S = 30
REMOVE_WAIT_S = 120  # how long `remove` waits for a running check (or bench) to let go of a card
SBIN = "/usr/local/sbin/orion-gpu-power-limits"
UNIT_DIR = "/etc/systemd/system"
UNITS = ("orion-gpu-power-limits.service", "orion-gpu-power-limits-check.service",
         "orion-gpu-power-limits.timer")


class ConfigError(ValueError):
    pass


def load_config(path: Path | None = None) -> dict[int, int | None]:
    """{nvidia-smi index: watts, or None for the card's default}. Raises ConfigError for a file this
    job must not act on."""
    path = CONFIG if path is None else path  # looked up per call, not frozen at import
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be an object")
    host = socket.gethostname()
    if raw.get("host") != host:
        raise ConfigError(f"{path} is for host {raw.get('host')!r}, this is {host!r}")
    cards = raw.get("cards")
    if not isinstance(cards, dict):
        raise ConfigError(f"{path}: 'cards' must be an object")
    out: dict[int, int | None] = {}
    for key, entry in cards.items():
        try:
            gpu = int(key)
            watts = entry["watts"]
            if isinstance(watts, bool) or not (watts == "default" or isinstance(watts, int)):
                raise ValueError(f"watts {watts!r}")
        except (TypeError, ValueError, KeyError) as exc:
            raise ConfigError(f"{path}: card {key!r} needs integer 'watts' or \"default\" ({exc})") from exc
        if gpu in out:
            raise ConfigError(f"{path}: card {gpu} is listed twice")
        out[gpu] = None if watts == "default" else watts
    return out


def smi(*args: str) -> str:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True,
                          timeout=SMI_TIMEOUT_S, start_new_session=True).stdout


def read_limits(gpu: int) -> tuple[float, float, float, float]:
    """(current, default, min, max) watts."""
    out = smi("-i", str(gpu), "--query-gpu=power.limit,power.default_limit,power.min_limit,power.max_limit",
              "--format=csv,noheader,nounits")
    cur, default, lo, hi = (float(x) for x in out.strip().split(","))
    return cur, default, lo, hi


def set_limit(gpu: int, watts: float) -> None:
    smi("-i", str(gpu), "-pl", f"{watts:.0f}")


def card_lock(gpu: int, wait_s: float = 0.0):
    """The bench's per-card flock (scripts/gpu_power_bench.py), also taken by every run of this job.
    None if someone else still holds it after `wait_s` seconds."""
    fh = open(LOCK_DIR / f"gpu-power-bench-{gpu}.lock", "w")  # noqa: SIM115
    deadline = time.monotonic() + wait_s
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except BlockingIOError:
            if time.monotonic() >= deadline:
                fh.close()
                return None
            time.sleep(1.0)


def drive(gpu: int, target_of: str, target: int | None = None) -> bool:
    """Move one card to `target`, or its default when None. apply: True when it ends there or a
    bench (or another run of this job) holds the card -- the next check catches up. remove: waits
    for the lock and returns False if it never comes, so a lost reset is a failure, not a skip."""
    removing = target_of == "remove"
    lock = card_lock(gpu, REMOVE_WAIT_S if removing else 0.0)
    if lock is None:
        if removing:
            print(f"gpu{gpu}: NOT reset to default: its lock was still held after {REMOVE_WAIT_S} s "
                  f"(a power bench is running?); run `remove` again when it ends", flush=True)
            return False
        print(f"gpu{gpu}: power bench (or another check) holds the card, skipped", flush=True)
        return True
    try:
        cur, default, lo, hi = read_limits(gpu)
        want = default if target is None else target
        if not lo <= want <= hi:
            print(f"gpu{gpu}: target {want} W is outside this card's range {lo:.0f}-{hi:.0f} W; "
                  f"refused, left at {cur:.0f} W", flush=True)
            return False
        if round(cur) == round(want):
            return True
        set_limit(gpu, want)
        after = read_limits(gpu)[0]
        print(f"gpu{gpu}: power limit {cur:.0f} -> {after:.0f} W ({target_of})", flush=True)
        return round(after) == round(want)
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        print(f"gpu{gpu}: nvidia-smi failed: {exc!r}", flush=True)
        return False
    finally:
        lock.close()


def apply(cards: dict[int, int | None]) -> int:
    ok = [drive(gpu, "drift" if os.environ.get("ORION_GPU_POWER_LIMITS_CHECK") else "apply", watts)
          for gpu, watts in sorted(cards.items())]
    return 0 if all(ok) else 1


def remove(cards: dict[int, int | None]) -> int:
    ok = [drive(gpu, "remove") for gpu in sorted(cards)]
    return 0 if all(ok) else 1


def status(cards: dict[int, int | None]) -> int:
    for gpu, watts in sorted(cards.items()):
        try:
            cur, default, lo, hi = read_limits(gpu)
            want = default if watts is None else watts
            row = {"gpu": gpu, "target_w": want, "current_w": cur, "default_w": default,
                   "at_target": round(cur) == round(want)}
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            row = {"gpu": gpu, "target_w": watts, "error": repr(exc)}
        print(json.dumps(row), flush=True)
    return 0


def _run(cmd: list[str]) -> None:
    if os.environ.get("DRY_RUN"):
        print(" ".join(cmd))
    else:
        subprocess.run(cmd, check=True)


def install() -> int:
    here = Path(__file__).resolve().parent
    _run(["install", "-m", "0755", str(here / "gpu_power_limits.py"), SBIN])
    for unit in UNITS:
        _run(["install", "-m", "0644", str(here / unit), f"{UNIT_DIR}/{unit}"])
    _run(["systemctl", "daemon-reload"])
    _run(["systemctl", "enable", "orion-gpu-power-limits.service", "orion-gpu-power-limits.timer"])
    # restart, not just enable --now: re-applies when the script or config changed on a re-install
    _run(["systemctl", "restart", "orion-gpu-power-limits.service"])
    _run(["systemctl", "start", "orion-gpu-power-limits.timer"])
    _run([SBIN, "status"])
    return 0


def uninstall() -> int:
    """Checks off first, then an explicit reset (ExecStop only runs if the service is active, and a
    oneshot whose boot apply failed is not), then the units and the binary go."""
    failed = False
    for cmd in (["systemctl", "disable", "--now", "orion-gpu-power-limits.timer"],
                ["systemctl", "stop", "orion-gpu-power-limits-check.service"],
                [SBIN, "remove"],
                ["systemctl", "disable", "--now", "orion-gpu-power-limits.service"]):
        try:
            _run(cmd)
        except (subprocess.CalledProcessError, OSError) as exc:
            failed |= cmd[0] == SBIN
            print(f"{' '.join(cmd)}: {exc}", file=sys.stderr)
    if failed:
        print("cards were NOT all reset; units left installed. Fix the cause and run uninstall again.",
              file=sys.stderr)
        return 1
    _run(["rm", "-f", SBIN, *(f"{UNIT_DIR}/{u}" for u in UNITS)])
    _run(["systemctl", "daemon-reload"])
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    verb = argv[0] if argv else ""
    if verb == "install":
        return install()
    if verb == "uninstall":
        return uninstall()
    if verb not in ("apply", "remove", "status"):
        print("usage: gpu_power_limits.py install|uninstall|apply|remove|status", file=sys.stderr)
        return 2
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not found", file=sys.stderr)
        return 1
    try:
        cards = load_config()
    except ConfigError as exc:
        print(f"refused: {exc}; no card touched", file=sys.stderr)
        return 1
    return {"apply": apply, "remove": remove, "status": status}[verb](cards)


if __name__ == "__main__":
    sys.exit(main())
