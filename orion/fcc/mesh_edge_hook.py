"""Mesh-edge PreToolUse hook for FCC's ``claude -p`` turns (issue #3409).

Orion holds unrestricted Bash in its FCC turns and ``gh`` authenticated as Orion's own GitHub
user (see ``orion/fcc/claude_spawn.py``). That is Orion's main way to write in public: issues,
PRs, comments. Before each Bash call this hook looks for a GitHub write (``gh issue|pr
create/comment/edit/review``, ``gh api`` with a write method or fields, ``gh gist|release
create``, or ``curl`` to api.github.com with a body) and, for one, checks the command text and
any body file it names against the mesh-edge guard (``orion/memory/mesh_edge.py``). A command that
quotes a core-page or lived-memory sentence verbatim is blocked (exit 2; stderr goes back to the
model). Reads (``gh pr view``, ``gh api`` GET) pass untouched.

Fail closed for writes: when the guard cannot load (no DSN, DB down) the write is blocked with a
message saying why. Any other error in the hook itself fails open, like the repeat-failure
breaker, so a broken hook never stops a non-GitHub tool call.

Run by file path from the claude subprocess (``python /app/orion/fcc/mesh_edge_hook.py``): it puts
the repo root on ``sys.path`` itself before importing ``orion``.
"""
from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

BLOCK_MARKER = "[orion-mesh-edge]"
MAX_FILE_BYTES = 1_000_000

_GH_WRITE = re.compile(
    r"(?:^|[\s;&|()`$])gh\s+(?:"
    r"(?:issue|pr)\s+(?:create|comment|edit|review|close|reopen|merge)"
    r"|(?:gist|release)\s+(?:create|edit)"
    r"|api\b[^\n;&|]*?(?:-X\s*|--method[\s=]+)(?:POST|PATCH|PUT)"
    r"|api\b[^\n;&|]*?\s(?:-f|-F|--field|--raw-field|--input)[\s=]"
    r")",
    re.IGNORECASE,
)
_CURL_GITHUB_WRITE = re.compile(
    r"\bcurl\b[^\n]*api\.github\.com[^\n]*(?:\s-d\b|--data|-X\s*(?:POST|PATCH|PUT))"
    r"|\bcurl\b[^\n]*(?:\s-d\b|--data|-X\s*(?:POST|PATCH|PUT))[^\n]*api\.github\.com",
    re.IGNORECASE,
)
_FILE_FLAGS = {"--body-file", "-F", "--input"}


def is_github_write(command: str) -> bool:
    return bool(_GH_WRITE.search(command or "") or _CURL_GITHUB_WRITE.search(command or ""))


def referenced_files(command: str) -> List[str]:
    """Paths a GitHub write reads its body from: ``--body-file X``, ``-F X``, ``--input X``,
    ``field=@X`` and ``-d @X``. Best effort: an unparsable command yields what the regex finds."""
    paths: List[str] = []
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    for i, tok in enumerate(tokens):
        if tok in _FILE_FLAGS and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            # gh api -F key=value is a field, not a file; key=@path still reads a file.
            if "=" in nxt and not nxt.split("=", 1)[1].startswith("@"):
                continue
            paths.append(nxt.split("=", 1)[1][1:] if "=@" in nxt else nxt)
        elif tok.startswith("--body-file="):
            paths.append(tok.split("=", 1)[1])
        elif "=@" in tok:
            paths.append(tok.split("=@", 1)[1])
        elif tok.startswith("@") and i > 0 and tokens[i - 1] in {"-d", "--data", "--data-binary"}:
            paths.append(tok[1:])
    return [p for p in paths if p and p != "-"]


def _read(path: str, cwd: Optional[str]) -> str:
    p = Path(path)
    if not p.is_absolute() and cwd:
        p = Path(cwd) / p
    try:
        with p.open("rb") as fh:
            return fh.read(MAX_FILE_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return ""


def decide(payload: Dict[str, Any], *, guard_loader: Optional[Callable[[], Any]] = None) -> Optional[str]:
    """Block message for this PreToolUse payload, or None to allow."""
    if payload.get("tool_name") != "Bash":
        return None
    tool_input = payload.get("tool_input") or {}
    command = str(tool_input.get("command") or "") if isinstance(tool_input, dict) else ""
    if not is_github_write(command):
        return None
    from orion.memory.mesh_edge import MeshEdgeUnavailable, load_guard

    text = "\n".join([command] + [_read(f, payload.get("cwd")) for f in referenced_files(command)])
    try:
        guard = (guard_loader or load_guard)()
    except MeshEdgeUnavailable as exc:
        return (f"{BLOCK_MARKER} Blocked: this GitHub write could not be checked for private memory "
                f"text ({exc}). GitHub is public. Do not retry this write in this turn; say that it was "
                "held and why.")
    hits = guard.find(text)
    if not hits:
        return None
    sources = sorted({h.source.split(":", 1)[0] if h.source.count(":") > 1 else h.source for h in hits})
    return (f"{BLOCK_MARKER} Blocked: this GitHub write quotes {len(hits)} sentence(s) verbatim from "
            f"your core pages or lived memories ({', '.join(sources)}). GitHub is public; those lines "
            "hold private facts about Juniper and her family. Rewrite the text without quoting them "
            "(describe the work, not the private detail), then write again.")


def hook_entry(python_bin: str) -> Dict[str, Any]:
    """The PreToolUse entry for ``--settings`` (Bash only)."""
    cmd = " ".join(shlex.quote(p) for p in (python_bin, str(Path(__file__).resolve())))
    return {"matcher": "Bash", "hooks": [{"type": "command", "command": cmd}]}


def main() -> int:
    repo = str(Path(__file__).resolve().parents[2])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            return 0
        message = decide(payload)
    except Exception:  # noqa: BLE001 -- fail open on hook bugs; decide() fails closed itself
        return 0
    if message is None:
        return 0
    sys.stderr.write(message + "\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
