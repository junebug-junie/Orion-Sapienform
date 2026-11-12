"""A prompt too big for one exec argument goes to the CLI on stdin (E2BIG regression).

Live 2026-11-12: Orion's 11-11 day letter prompt was ~131,300 chars. Passed as ``claude -p
<prompt>``, every attempt died with ``[Errno 7] Argument list too long`` before the CLI
started (Linux MAX_ARG_STRLEN = 131,072 bytes per argument)."""
from __future__ import annotations

import asyncio
import json
import sys

from orion.harness.fcc_motor import ARGV_PROMPT_MAX_BYTES, _SpawnTurnIO, split_oversized_prompt

# Reports what the child actually received: its argv tail and its stdin.
_ECHO = "import sys,json;print(json.dumps({'argv':sys.argv[1:],'stdin':sys.stdin.read()}))"


def test_small_prompt_stays_in_argv():
    argv = ["claude", "-p", "hello", "--model", "m"]
    assert split_oversized_prompt(argv) == (argv, None)


def test_warm_pool_argv_has_no_prompt_to_move():
    argv = ["claude", "-p", "--input-format", "stream-json"]
    assert split_oversized_prompt(argv) == (argv, None)


def test_oversized_prompt_leaves_argv():
    big = "x" * (ARGV_PROMPT_MAX_BYTES + 1)
    argv, data = split_oversized_prompt(["claude", "-p", big, "--resume", "s1"])
    assert argv == ["claude", "-p", "--resume", "s1"]
    assert data == big.encode()


def test_limit_counts_utf8_bytes_not_chars():
    big = "é" * (ARGV_PROMPT_MAX_BYTES // 2 + 1)  # under the limit in chars, over it in bytes
    assert split_oversized_prompt(["claude", "-p", big])[1] is not None


def test_kernel_limit_prompt_reaches_the_process_on_stdin(tmp_path):
    """The real spawn: a 140 KB prompt that E2BIGs as an argument arrives whole on stdin."""
    big = "Orion's day. " * 11_000
    assert len(big.encode()) > 131_072
    argv = [sys.executable, "-c", _ECHO, "-p", big, "--output-format", "stream-json"]

    async def run():
        io = _SpawnTurnIO(argv, cwd=str(tmp_path), env={}, limit=1 << 20, claude_bin=sys.executable)
        await io.start()
        out = await io.proc.stdout.read()
        assert await io.wait() == 0
        return json.loads(out)

    seen = asyncio.run(run())
    assert seen["argv"] == ["-p", "--output-format", "stream-json"]
    assert seen["stdin"] == big
