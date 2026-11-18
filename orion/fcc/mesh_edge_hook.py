"""Mesh-edge PreToolUse hook for FCC's ``claude -p`` turns (issue #3409).

Orion holds unrestricted Bash in its FCC turns and ``gh`` authenticated as Orion's own GitHub
user (see ``orion/fcc/claude_spawn.py``). That is Orion's main way to write in public: issues,
PRs, comments. Before each Bash call this hook looks for a GitHub write (``gh issue|pr
create/comment/edit/review``, ``gh api`` with a write method or fields, ``gh gist|release
create``, or ``curl`` to api.github.com with a body) and, for one, checks the command text and
any body file it names against the mesh-edge guard (``orion/memory/mesh_edge.py``). A command that
quotes a core-page or lived-memory sentence verbatim, or repeats a run of 8+ of its words (#3554),
is blocked (exit 2; stderr goes back to the
model). Reads (``gh pr view``, ``gh api`` GET) pass untouched.

Every ``gh`` call is treated as a write unless it is a known read (view, list, status, ...; ``gh
api`` without a write method or field). Fail closed for writes: when the guard cannot load, when
the hook itself errors, or when the body is built at run time where the hook cannot read it
(``$(...)``, backticks, a variable, a pipe into ``--body-file -``), the write is blocked with a
message saying why. A non-GitHub tool call is never blocked by a hook error.

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

# Any gh invocation (bare, absolute path, after a pipe/subshell). Every gh call counts as a write
# unless it is a known read: an unknown subcommand is checked, not trusted.
_GH_CALL = re.compile(r"(?:^|[\s;&|()`$/])gh\s+([a-z-]+)(?:\s+([a-z-]+))?", re.IGNORECASE)
_GH_READ_VERBS = frozenset({"view", "list", "status", "diff", "checks", "browse", "search", "ls"})
_GH_READ_GROUPS = frozenset({"auth", "help", "version", "completion", "config", "alias", "search", "browse",
                             "status", "run", "workflow", "cache", "extension", "codespace", "gpg-key", "ssh-key"})
_GH_API_WRITE = re.compile(
    r"(?:-X\s*|--method[\s=]+)(?:POST|PATCH|PUT|DELETE)|\s(?:-f|-F|--field|--raw-field|--input)[\s=]", re.IGNORECASE)
_CURL_GITHUB = re.compile(r"\bcurl\b[^\n]*(?:api\.github\.com|uploads\.github\.com)", re.IGNORECASE)
_CURL_WRITE = re.compile(
    r"\s(?:-d|--data(?:-raw|-binary|-urlencode)?|--json|-F|--form|-T|--upload-file)(?:[\s=@]|$)"
    r"|-X\s*(?:POST|PATCH|PUT)|\s-d@", re.IGNORECASE)
_FILE_FLAGS = {"--body-file", "-F", "--input", "--json", "--notes-file", "-T", "--upload-file"}
_REDIRECT = re.compile(r"<\s*([^\s<>|;&]+)")
# A body Orion's shell builds at run time: the hook cannot read it, so it cannot clear it.
_HIDDEN_BODY = re.compile(r"\$\(|`|\$\{?[A-Za-z_]")


def _unfold(command: str) -> str:
    return (command or "").replace("\\\n", " ")


def is_github_write(command: str) -> bool:
    cmd = _unfold(command)
    for m in _GH_CALL.finditer(cmd):
        group, verb = (m.group(1) or "").lower(), (m.group(2) or "").lower()
        if group == "api":
            tail = cmd[m.end(1):].split("\n", 1)[0]
            if _GH_API_WRITE.search(" " + tail):
                return True
            continue
        if group in _GH_READ_GROUPS or verb in _GH_READ_VERBS:
            continue
        return True
    return bool(_CURL_GITHUB.search(cmd) and _CURL_WRITE.search(cmd))


def referenced_files(command: str) -> List[str]:
    """Paths a GitHub write reads its body from: ``--body-file X``, ``-F X``, ``--input X``,
    ``field=@X``, ``-d @X`` / ``-d@X``, ``--json @X`` and ``< X`` redirects."""
    cmd = _unfold(command)
    paths: List[str] = [m.group(1) for m in _REDIRECT.finditer(cmd) if not m.group(1).startswith("<")]
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        tokens = cmd.split()
    for i, tok in enumerate(tokens):
        if tok in _FILE_FLAGS and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            # gh api -F key=value is a field, not a file; key=@path still reads a file.
            if "=" in nxt and not nxt.split("=", 1)[1].startswith("@"):
                continue
            paths.append(nxt.split("=", 1)[1][1:] if "=@" in nxt else nxt.lstrip("@"))
        elif tok.startswith(("--body-file=", "--input=", "--notes-file=")):
            paths.append(tok.split("=", 1)[1])
        elif "=@" in tok:
            paths.append(tok.split("=@", 1)[1])
        elif tok.startswith("-d@") or tok.startswith("--data@"):
            paths.append(tok.split("@", 1)[1])
        elif tok.startswith("@") and i > 0 and tokens[i - 1] in {"-d", "--data", "--data-binary", "--data-raw", "--json"}:
            paths.append(tok[1:])
    return list(dict.fromkeys(p for p in paths if p and p != "-"))


def hidden_body(command: str) -> bool:
    """True when the write's body comes from somewhere the hook cannot read: stdin without a
    redirect or heredoc, ``$(...)``, backticks or a shell variable."""
    cmd = _unfold(command)
    if _HIDDEN_BODY.search(cmd):
        return True
    reads_stdin = bool(re.search(r"(?:--body-file|-F|--input|--notes-file)[\s=]+-(?:\s|$)", cmd))
    if reads_stdin and "<<" not in cmd and not _REDIRECT.search(cmd):
        return True  # piped in from another command
    return False


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
    command = _command(payload)
    if command is None or not is_github_write(command):
        return None
    from orion.memory.mesh_edge import MeshEdgeUnavailable, load_guard

    if hidden_body(command):
        return (f"{BLOCK_MARKER} Blocked: this GitHub write builds its text at run time ($(...), "
                "backticks, a shell variable or a pipe), so it cannot be checked for private memory "
                "text. GitHub is public. Write the body to a file and pass --body-file <path> instead.")
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
    kinds = sorted({h.source.split(":", 1)[0] for h in hits})
    verbatim = sum(1 for h in hits if h.kind == "verbatim")
    near = len(hits) - verbatim
    what = ", ".join(part for part in (
        f"quotes {verbatim} sentence(s) verbatim" if verbatim else "",
        f"repeats a run of {guard.shingle_n}+ words from {near} sentence(s)" if near else "",
    ) if part)
    return (f"{BLOCK_MARKER} Blocked: this GitHub write {what} from your core pages or stored "
            f"memories ({', '.join(kinds)}). GitHub is public and those lines stay inside your mesh. "
            "Rewrite the text in your own words without quoting them, then write again.")


def _command(payload: Dict[str, Any]) -> Optional[str]:
    if payload.get("tool_name") != "Bash":
        return None
    tool_input = payload.get("tool_input") or {}
    return str(tool_input.get("command") or "") if isinstance(tool_input, dict) else ""


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
    except Exception:  # noqa: BLE001 -- unreadable payload: nothing to judge
        return 0
    if not isinstance(payload, dict):
        return 0
    try:
        message = decide(payload)
    except Exception as exc:  # noqa: BLE001
        # Fail open only for calls that are not GitHub writes; a write we could not check is held.
        try:
            command = _command(payload)
            is_write = command is not None and is_github_write(command)
        except Exception:  # noqa: BLE001
            is_write = False
        if not is_write:
            return 0
        message = (f"{BLOCK_MARKER} Blocked: the mesh-edge check failed on this GitHub write "
                   f"({type(exc).__name__}). GitHub is public; it is held.")
    if message is None:
        return 0
    sys.stderr.write(message + "\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
