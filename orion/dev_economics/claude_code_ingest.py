"""Read-only parser for local Claude Code session transcripts (``~/.claude/projects/**/*.jsonl``).

Extracts *only* Juniper's own typed words -- not tool results, not hook output,
not slash-command scaffolding, not Orion's/the assistant's turns. This is the
shared parsing pass named (but not yet built) in
``docs/superpowers/specs/2026-07-30-dev-economics-signal-design.md`` and reused
by ``docs/superpowers/specs/2026-07-30-juniper-affective-state-signal-proposal.md``
("no new data source, no new collection surface" -- Juniper's message text is
already something Orion receives every turn; this just re-reads the same
transcript files Claude Code already writes to local disk).

Filtering rules, in order:

1. ``type == "user"`` and ``message.content`` is a plain ``str``. A list
   ``content`` is almost always a tool-result echo (the harness re-injects
   tool output under role ``user``) or synthetic harness text (e.g.
   ``"[Request interrupted by user]"``), never something Juniper typed --
   confirmed by direct inspection of this repo's real local transcript
   corpus 2026-08-11. That same inspection found a small number (~1 in a
   corpus of ~83k list-content turns) of genuine short human replies
   embedded as a lone ``{"type": "text", ...}`` block inside a list, which
   this rule drops. Accepted, disclosed gap: distinguishing that rare real
   case from the far more common synthetic-injection case reliably was not
   worth the risk of instead *admitting* synthetic harness text as if
   Juniper had typed it, which would be the worse failure for a signal
   whose whole point is measuring Juniper's own words.
2. ``promptSource == "typed"`` and ``origin.kind == "human"`` when present --
   distinguishes a real keystroke turn from synthetic/injected turns. Older
   transcript lines may lack these fields; treated as typed/human by default
   (conservative toward inclusion, not exclusion, since the alternative is
   silently losing real signal from every transcript written before these
   fields existed).
3. Known synthetic wrapper tags are stripped from the text before scoring --
   ``<local-command-caveat>``, ``<command-name>``, ``<command-message>``,
   ``<command-args>``, ``<system-reminder>`` -- since their content is harness
   boilerplate, not something Juniper composed. A message that is *entirely*
   wrapper text (e.g. a bare ``/compact`` with no trailing prose) yields
   ``None`` after stripping and is dropped, not scored as an empty message.
4. A message with no parseable ``timestamp`` (missing, or not an ISO-8601
   string) is dropped entirely, not included with a null timestamp -- every
   real message in this repo's local corpus as of 2026-08-11 has one, so
   this has zero observed real-world impact today, but a future transcript
   format that omits it would silently lose messages rather than crash.

No write path. Never persists raw text -- callers of this module are
responsible for holding the returned text only as long as it takes to compute
a derived score, per the privacy boundary in the affective-state proposal doc.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

DEFAULT_PROJECTS_ROOT = Path.home() / ".claude" / "projects"

_WRAPPER_TAG_PATTERN = re.compile(
    r"<(local-command-caveat|command-name|command-message|command-args|system-reminder)>"
    r".*?</\1>",
    re.DOTALL,
)


@dataclass(frozen=True)
class HumanMessage:
    """One real, Juniper-typed turn, already stripped of harness scaffolding."""

    session_id: str
    timestamp: datetime
    text: str
    transcript_path: Path


def iter_transcript_files(root: Path | str = DEFAULT_PROJECTS_ROOT) -> Iterator[Path]:
    """Every ``*.jsonl`` transcript under ``root``, recursively. Read-only glob --
    does not open or validate the files, just enumerates candidates."""
    root_path = Path(root)
    if not root_path.exists():
        return
    yield from sorted(root_path.glob("**/*.jsonl"))


def _strip_wrapper_tags(raw: str) -> str | None:
    cleaned = _WRAPPER_TAG_PATTERN.sub("", raw).strip()
    return cleaned or None


def _is_real_typed_human_turn(obj: dict) -> bool:
    if obj.get("type") != "user":
        return False
    message = obj.get("message")
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if not isinstance(content, str):
        # list content is a tool-result echo re-injected under role "user",
        # never something Juniper typed.
        return False
    prompt_source = obj.get("promptSource")
    if prompt_source is not None and prompt_source != "typed":
        return False
    origin = obj.get("origin")
    if isinstance(origin, dict) and origin.get("kind") not in (None, "human"):
        return False
    return True


def parse_transcript_file(path: Path | str) -> Iterator[HumanMessage]:
    """Yields one ``HumanMessage`` per real typed turn in ``path``. Malformed
    JSON lines and turns that fail the filter above are skipped silently --
    a transcript file is append-only harness output, not a validated contract,
    so a single corrupt line must not abort the whole file."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or not _is_real_typed_human_turn(obj):
                continue
            raw_text = obj["message"]["content"]
            cleaned = _strip_wrapper_tags(raw_text)
            if cleaned is None:
                continue
            timestamp_raw = obj.get("timestamp")
            timestamp = None
            if isinstance(timestamp_raw, str) and timestamp_raw:
                try:
                    timestamp = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
                except ValueError:
                    timestamp = None
            if timestamp is None:
                # Also covers a non-string timestamp field (e.g. a raw epoch
                # int) -- a corrupt/unexpected line drops just this message,
                # never raises, matching this function's own "single corrupt
                # line must not abort the whole file" guarantee.
                continue
            session_id = obj.get("sessionId") or file_path.stem
            yield HumanMessage(
                session_id=session_id,
                timestamp=timestamp,
                text=cleaned,
                transcript_path=file_path,
            )


def iter_all_human_messages(root: Path | str = DEFAULT_PROJECTS_ROOT) -> Iterator[HumanMessage]:
    """Convenience: every real typed human turn across every transcript under
    ``root``, in file-then-line order (not globally time-sorted -- callers
    that need chronological order across sessions should sort the result)."""
    for transcript_path in iter_transcript_files(root):
        yield from parse_transcript_file(transcript_path)
