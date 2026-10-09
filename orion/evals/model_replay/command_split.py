"""Split a model's Bash command into simple commands so the parts that reach a database or the
network can be routed to read-only host executors, and everything else to the no-network sandbox.

Deliberately conservative: anything it cannot split with confidence (a database/network tool
inside ``$(...)``, backticks, a subshell or a function body) comes back as ``Unsplittable`` and
the replay refuses that one command with a plain message. A refusal is recorded, never executed.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Optional

# Tools that can reach state outside the sandbox. Anything else runs inside the no-network,
# read-only container, where a write has nothing to land on.
EXTERNAL_TOOLS = frozenset({"redis-cli", "psql", "curl", "docker"})

_HEREDOC_RE = re.compile(r"<<-?[ \t]*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1[^\n]*\n(.*?)\n[ \t]*\2[ \t]*(?=\n|$)", re.S)
_SEPARATORS = {"|", "||", "&&", ";", "&"}
_REDIRECTS = {">", ">>", "<", "<>"}


class Unsplittable(ValueError):
    pass


@dataclass
class Simple:
    """One simple command: argv tokens, the operator that FOLLOWS it, optional heredoc stdin."""

    argv: list[str]
    op: Optional[str] = None
    stdin: Optional[str] = None
    raw: str = ""

    @property
    def tool(self) -> str:
        for tok in self.argv:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tok):
                continue  # FOO=bar env prefix
            return tok.rsplit("/", 1)[-1]
        return ""


@dataclass
class Split:
    commands: list[Simple] = field(default_factory=list)


def mentions_external(command: str) -> bool:
    return any(re.search(rf"(^|[\s;|&(`$/]){re.escape(t)}(\s|$)", command) for t in EXTERNAL_TOOLS)


def _extract_heredocs(command: str) -> tuple[str, list[str]]:
    bodies: list[str] = []

    def repl(m: re.Match[str]) -> str:
        bodies.append(m.group(3))
        return f" __REPLAY_HEREDOC_{len(bodies) - 1}__\n"

    return _HEREDOC_RE.sub(repl, command), bodies


def split_command(command: str) -> Split:
    """Simple commands with their trailing operators. Raises Unsplittable."""
    text = command.replace("\\\r\n", " ").replace("\\\n", " ")
    text, bodies = _extract_heredocs(text)
    if "<<" in text:
        raise Unsplittable("unterminated heredoc")
    # stderr plumbing is cosmetic here (every executor returns stdout+stderr together) and
    # `2>&1` would otherwise lex into three tokens.
    text = re.sub(r"\s[12]?>&[12]\b|\s2>\s*/dev/null", " ", text)
    text = _unquoted_newlines_to_semicolons(text)
    for marker in ("$(", "`", "<(", ">("):
        if marker in text:
            raise Unsplittable(f"{marker} with a database/network tool")
    lex = shlex.shlex(text, posix=True, punctuation_chars=";&|()<>")
    lex.whitespace = " \t\r"
    lex.whitespace_split = True
    lex.commenters = ""
    try:
        tokens = list(lex)
    except ValueError as exc:  # unbalanced quotes
        raise Unsplittable(str(exc)) from exc
    out = Split()
    cur: list[str] = []
    for tok in tokens:
        if tok in ("(", ")", "{", "}"):
            raise Unsplittable("subshell or group")
        if tok in _SEPARATORS:
            if cur:
                out.commands.append(Simple(argv=cur, op=tok))
            cur = []
            continue
        cur.append(tok)
    if cur:
        out.commands.append(Simple(argv=cur, op=None))
    for cmd in out.commands:
        if "&" == cmd.op:
            raise Unsplittable("background job")
        stdin_parts: list[str] = []
        kept: list[str] = []
        for tok in cmd.argv:
            m = re.fullmatch(r"__REPLAY_HEREDOC_(\d+)__", tok)
            if m:
                stdin_parts.append(bodies[int(m.group(1))] + "\n")
            elif tok == "<<":
                continue
            else:
                kept.append(tok)
        cmd.argv = kept
        cmd.stdin = "".join(stdin_parts) or None
        cmd.raw = " ".join(t if t in _REDIRECTS else shlex.quote(t) for t in kept)
    return out


def _unquoted_newlines_to_semicolons(text: str) -> str:
    """A newline outside quotes ends a command, like ``;``. shlex would otherwise glue it into a word."""
    out: list[str] = []
    quote: Optional[str] = None
    escaped = False
    for ch in text:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\" and quote != "'":
            out.append(ch)
            escaped = True
            continue
        if quote:
            if ch == quote:
                quote = None
            out.append(ch)
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            continue
        out.append(" ; " if ch == "\n" else ch)
    return "".join(out)
