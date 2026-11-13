"""The broker's readers: each one runs an existing, short text render and returns its text.

Readers never decide anything and never redact or cap (the broker does both, for every reader alike).
A reader that cannot read raises ``ReaderUnavailable`` with a short reason: the broker then answers
"Not available: <reason>", never stale or partial data shown as current.

* ``prime_now``: ``orion/prime/block.py`` ``fetch_prime_block`` (one Redis read, 600 characters).
* ``body_economy``: ``python -P -m orion.dig --dig claude-tools body-economy ...`` as an argument list
  (no shell). Arguments arrive already validated by ``BodyEconomyArguments`` (closed sets only).
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from orion.claude_tools.registry import BodyEconomyArguments, PrimeNowArguments

DIG_ID = "claude-tools"
BODY_ECONOMY_TIMEOUT_SEC = 15.0
FULL_REPORT_PREFIX = "full report: "


class ReaderUnavailable(RuntimeError):
    """The reader could not read. The message is short and safe to show."""


async def read_prime_now(redis: Any, args: PrimeNowArguments) -> str:
    from orion.prime.block import fetch_prime_block

    block = await fetch_prime_block(redis)
    if block.status == "absent":
        # "Prime's state could not be read" is a reader failure, not Prime's present.
        raise ReaderUnavailable(block.text.strip() or "Prime's state could not be read")
    # ok, or stale: a stale block only says when Prime last ticked; it never shows old state as current.
    return block.text


def body_economy_argv(args: BodyEconomyArguments, *, python: Optional[str] = None) -> list[str]:
    argv = [python or sys.executable, "-P", "-m", "orion.dig", "--dig", DIG_ID, "body-economy"]
    if args.section and args.section != "overview":
        argv.append(args.section)
    argv += ["--window", args.window]
    if args.host:
        argv += ["--host", args.host]
    return argv


def _drop_report_file(lines: Sequence[str], dig_home: Path) -> list[str]:
    """Drop dig's "full report: <path>" line (a path in this container means nothing to Claude) and
    delete that file: it is this call's own copy of a ~400 KB report nobody will read."""
    root = (dig_home / "dig" / DIG_ID).resolve()
    kept: list[str] = []
    for line in lines:
        if line.startswith(FULL_REPORT_PREFIX):
            path = Path(line[len(FULL_REPORT_PREFIX):].strip())
            try:
                if path.resolve().parent == root and path.suffix == ".json":
                    path.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        kept.append(line)
    return kept


def run_body_economy(
    args: BodyEconomyArguments,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    env: Optional[Mapping[str, str]] = None,
    timeout_sec: float = BODY_ECONOMY_TIMEOUT_SEC,
) -> str:
    argv = body_economy_argv(args)
    run_env = dict(env if env is not None else os.environ)
    try:
        proc = runner(argv, capture_output=True, text=True, timeout=timeout_sec, env=run_env, cwd="/", check=False)
    except subprocess.TimeoutExpired as exc:
        raise ReaderUnavailable(f"the body economy report did not answer within {timeout_sec:g} s") from exc
    except OSError as exc:
        raise ReaderUnavailable(f"the body economy reader could not start ({type(exc).__name__})") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        raise ReaderUnavailable(detail[-1][:200] if detail else f"the body economy reader exited {proc.returncode}")
    home = Path(run_env.get("ORION_DIG_HOME") or "/mnt/orion-fcc")
    text = "\n".join(_drop_report_file((proc.stdout or "").splitlines(), home)).strip()
    if not text:
        raise ReaderUnavailable("the body economy reader printed nothing")
    return text


async def read_body_economy(args: BodyEconomyArguments) -> str:
    return await asyncio.to_thread(run_body_economy, args)


__all__ = [
    "DIG_ID",
    "ReaderUnavailable",
    "body_economy_argv",
    "read_body_economy",
    "read_prime_now",
    "run_body_economy",
]
