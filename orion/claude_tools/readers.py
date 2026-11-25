"""The broker's readers: each one runs an existing, short text render and returns its text.

Readers never decide anything and never redact or cap (the broker does both, for every reader alike).
A reader that cannot read raises ``ReaderUnavailable`` with a short reason: the broker then answers
"Not available: <reason>", never stale or partial data shown as current.

* ``prime_now``: ``orion/prime/block.py`` ``fetch_prime_block`` (one Redis read, 600 characters).
* ``body_economy``: ``python -P -m orion.dig --dig claude-tools body-economy ...`` as an argument list
  (no shell). Arguments arrive already validated by ``BodyEconomyArguments`` (closed sets only).
* ``consults``: ``dig consult [<id>]``; ``arc``: ``dig arc <id>`` or ``dig act <id>``. Same runner. Ids
  arrive already matched against fixed hex patterns (``ConsultsArguments``, ``ArcArguments``).
* ``dreams``, ``curiosity``, ``reading_results``, ``orion_day``: Orion's own introspect read tools
  (``orion/introspect/tools.py`` ``IntrospectTools.invoke``, one bus RPC each), rendered as short text.
  Items marked ``intimate`` are dropped and counted, never sent.

Not here on purpose: ``dig body`` and ``dig portrait`` take ~55 s live (2026-11-25), which with the 41 s
consent wait would pass the 62 s MCP call limit.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from pydantic import BaseModel

from orion.claude_tools.registry import (
    INTROSPECT_TOOLS,
    ArcArguments,
    BodyEconomyArguments,
    ConsultsArguments,
    PrimeNowArguments,
)

DIG_ID = "claude-tools"
DIG_TIMEOUT_SEC = 15.0
BODY_ECONOMY_TIMEOUT_SEC = DIG_TIMEOUT_SEC
# dig's last line names the file holding the untrimmed result: "full report: <path>", "full consult
# (every turn, untrimmed): <path>", "full atlas: <path>".
FULL_LINE_RE = re.compile(r"^full [^:]{1,60}: (/\S+)$")
ITEM_TEXT_CHARS = 1200


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


def dig_argv(sub: Sequence[str], *, python: Optional[str] = None) -> list[str]:
    return [python or sys.executable, "-P", "-m", "orion.dig", "--dig", DIG_ID, *sub]


def body_economy_argv(args: BodyEconomyArguments, *, python: Optional[str] = None) -> list[str]:
    sub = ["body-economy"]
    if args.section and args.section != "overview":
        sub.append(args.section)
    sub += ["--window", args.window]
    if args.host:
        sub += ["--host", args.host]
    return dig_argv(sub, python=python)


def consults_argv(args: ConsultsArguments, *, python: Optional[str] = None) -> list[str]:
    return dig_argv(["consult", *([args.consult_id] if args.consult_id else [])], python=python)


def arc_argv(args: ArcArguments, *, python: Optional[str] = None) -> list[str]:
    return dig_argv(["act" if args.id.startswith("act_") else "arc", args.id], python=python)


def _drop_report_file(lines: Sequence[str], dig_home: Path) -> list[str]:
    """Drop dig's "full ...: <path>" line (a path in this container means nothing to Claude) and
    delete that file: it is this call's own copy of a result nobody will read."""
    root = (dig_home / "dig" / DIG_ID).resolve()
    kept: list[str] = []
    for line in lines:
        m = FULL_LINE_RE.match(line.strip())
        if m:
            path = Path(m.group(1))
            try:
                if path.resolve().parent == root and path.suffix == ".json":
                    path.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        kept.append(line)
    return kept


def run_dig(
    argv: list[str],
    what: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    env: Optional[Mapping[str, str]] = None,
    timeout_sec: float = DIG_TIMEOUT_SEC,
) -> str:
    run_env = dict(env if env is not None else os.environ)
    try:
        proc = runner(argv, capture_output=True, text=True, timeout=timeout_sec, env=run_env, cwd="/", check=False)
    except subprocess.TimeoutExpired as exc:
        raise ReaderUnavailable(f"the {what} did not answer within {timeout_sec:g} s") from exc
    except OSError as exc:
        raise ReaderUnavailable(f"the {what} reader could not start ({type(exc).__name__})") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        raise ReaderUnavailable(detail[-1][:200] if detail else f"the {what} reader exited {proc.returncode}")
    home = Path(run_env.get("ORION_DIG_HOME") or "/mnt/orion-fcc")
    text = "\n".join(_drop_report_file((proc.stdout or "").splitlines(), home)).strip()
    if not text:
        raise ReaderUnavailable(f"the {what} reader printed nothing")
    return text


def run_body_economy(args: BodyEconomyArguments, **kw: Any) -> str:
    return run_dig(body_economy_argv(args), "body economy report", **kw)


async def read_body_economy(args: BodyEconomyArguments) -> str:
    return await asyncio.to_thread(run_body_economy, args)


async def read_consults(args: ConsultsArguments) -> str:
    return await asyncio.to_thread(run_dig, consults_argv(args), "consult")


async def read_arc(args: ArcArguments) -> str:
    return await asyncio.to_thread(run_dig, arc_argv(args), "arc")


# --- Orion's introspect read tools -----------------------------------------------------------------

def render_introspect(tool: str, result: Mapping[str, Any]) -> str:
    """An IntrospectResultV1 as text: one block per item, intimate items dropped and counted."""
    items = list(result.get("items") or [])
    shown = [i for i in items if i.get("sensitivity") != "intimate"]
    withheld = len(items) - len(shown)
    total = result.get("total_available")
    head = f"{tool}: {len(shown)} item(s)" + (f" of {total} available" if total is not None else "")
    head += f", as of {result.get('as_of')}"
    lines = [head]
    if withheld:
        lines.append(f"({withheld} intimate item(s) withheld; they are never sent to Claude)")
    if not items:
        lines.append("nothing matched")
    for item in shown:
        text = " ".join(str(item.get("text") or "").split())
        if len(text) > ITEM_TEXT_CHARS:
            text = text[: ITEM_TEXT_CHARS - 1] + "…"
        tag = f"[{item.get('kind')}, {item.get('epistemic_status')}]"
        lines.append(f"\n{item.get('id')} {item.get('occurred_at')} {tag}\n{text}")
        extra = {k: v for k, v in (item.get("extra") or {}).items() if v not in (None, "", [], {})}
        if extra:
            lines.append("extra: " + json.dumps(extra, default=str, ensure_ascii=False)[:400])
    return "\n".join(lines)


def introspect_reader(tools: Any, name: str) -> Callable[[BaseModel], Any]:
    """A reader for one introspect tool. ``tools`` is an ``IntrospectTools`` (one bus RPC per call)."""
    introspect_name, parent = INTROSPECT_TOOLS[name]

    async def read(args: BaseModel) -> str:
        # Re-validate against Orion's own argument model: its limits are the only limits.
        own = parent.model_validate(args.model_dump(exclude={"reason"}, exclude_none=True))
        try:
            result = await tools.invoke(introspect_name, own.model_dump(mode="json", exclude_none=True))
        except Exception as exc:  # noqa: BLE001 -- IntrospectUnknownError and transport errors
            raise ReaderUnavailable(str(exc)[:200] or f"{name}: answer unknown") from exc
        return render_introspect(name, result)

    return read


__all__ = [
    "DIG_ID",
    "ReaderUnavailable",
    "arc_argv",
    "body_economy_argv",
    "consults_argv",
    "dig_argv",
    "introspect_reader",
    "read_arc",
    "read_body_economy",
    "read_consults",
    "read_prime_now",
    "render_introspect",
    "run_body_economy",
    "run_dig",
]
