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
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

logger = logging.getLogger("orion.dev_economics.claude_code_ingest")

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
    """Every ``*.jsonl`` transcript under ``root``, recursively. Read-only walk --
    does not open or validate the files, just enumerates candidates.

    Uses ``os.walk(..., onerror=...)`` rather than ``Path.glob("**/*.jsonl")``
    deliberately: code review 2026-08-11 correctly flagged that a plain glob
    walk is not covered by ``iter_all_human_messages()``'s per-file
    vanished-file handling -- if an entire subdirectory (not just one file)
    disappears mid-walk (the same kind of harness cleanup that caused the
    confirmed live single-file race this module's own docstring documents),
    ``glob``'s internal ``os.scandir`` raises and aborts the whole walk
    before a single file is ever yielded. ``onerror`` makes that failure
    mode a silent "skip this subtree" instead, matching this module's "one
    vanished thing must not abort the whole walk" principle at the directory
    level too, not just the file level."""
    root_path = Path(root)
    if not root_path.exists():
        return
    collected: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root_path, onerror=lambda _exc: None):
        for name in filenames:
            if name.endswith(".jsonl"):
                collected.append(Path(dirpath) / name)
    yield from sorted(collected)


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
                else:
                    if timestamp.tzinfo is None:
                        # Every real timestamp observed so far has a "Z"/
                        # offset suffix (naive-then-comparison-crash confirmed
                        # only as a hypothetical by code review 2026-08-11,
                        # not seen live) -- guard anyway so a genuinely naive
                        # ISO string doesn't make this function's contract
                        # depend on the caller separately checking tzinfo
                        # before comparing timestamps (affective_state.py's
                        # window filter does exactly that comparison).
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
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
    that need chronological order across sessions should sort the result).

    A file present in ``iter_transcript_files()``'s listing can still fail to
    open by the time ``parse_transcript_file()`` reaches it. First diagnosed
    live 2026-08-11 as a suspected delete-mid-walk race; root-caused the same
    day to something more specific and permanent, not a race at all: Claude
    Code represents a cross-project subagent transcript as an *absolute-path
    symlink* (a file under one session's ``subagents/`` dir pointing at
    ``/home/athena/.claude/projects/<other-project>/.../agent-*.jsonl``).
    Mounting this tree at a different in-container path than its real host
    path (a deployment/config choice, not a Claude Code bug) makes every such
    symlink unresolvable, raising ``FileNotFoundError`` for that file every
    single time, not intermittently. The real fix is deployment-side --
    mount at the identical path (see ``services/orion-cocreation-signals/
    docker-compose.yml``'s comment on its transcript mount) -- but this
    module still catches the failure defensively, both because a real
    transient race (deletion mid-walk) is also plausible and would look
    identical, and because a misconfigured mount should degrade to "skip the
    unreachable messages" rather than "the whole tick silently never
    publishes." Same "a single corrupt/unreachable file must not abort the
    whole walk" principle ``parse_transcript_file()`` already applies at the
    line level, extended here to the file level."""
    for transcript_path in iter_transcript_files(root):
        try:
            yield from parse_transcript_file(transcript_path)
        except OSError:
            logger.warning(
                "claude_code_ingest_transcript_vanished_during_walk path=%s", transcript_path
            )
            continue
