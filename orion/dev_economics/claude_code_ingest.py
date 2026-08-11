"""Read-only parser for local Claude Code session transcripts (``~/.claude/projects/**/*.jsonl``).

Two extraction passes over the same real, already-existing local data, per
``docs/superpowers/specs/2026-07-30-dev-economics-signal-design.md``'s framing
("no new data source, no new collection surface"):

- ``parse_transcript_file``/``iter_all_human_messages`` -- Juniper's own typed
  words only (not tool results, not hook output, not slash-command
  scaffolding, not Orion's/the assistant's turns). Built first, for the
  Juniper affective-state signal
  (``docs/superpowers/specs/2026-07-30-juniper-affective-state-signal-proposal.md``).
- ``parse_session_usage_record``/``iter_all_session_usage_records`` -- one
  normalized ledger record per transcript file: real token counts, model,
  effort tier, wall-clock duration, and word counts for *both* assistant and
  human turns (the dev-economics spec's own "Recommended next patch" --
  offline, no pricing, no bus, no service, per that doc's explicit phasing).

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
            # _parse_timestamp always returns a timezone-aware datetime or
            # None -- covers both an unparseable string and a non-string
            # timestamp field (e.g. a raw epoch int); either way this drops
            # just this one message, never raises, matching this function's
            # own "single corrupt line must not abort the whole file"
            # guarantee. Naive-then-comparison-crash (a genuinely naive ISO
            # string) confirmed only as a hypothetical by code review
            # 2026-08-11, not seen live -- guarded anyway.
            timestamp = _parse_timestamp(obj.get("timestamp"))
            if timestamp is None:
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


# ---------------------------------------------------------------------------
# Session usage ledger (dev-economics spec's "Recommended next patch")
# ---------------------------------------------------------------------------


def _parse_timestamp(raw: object) -> datetime | None:
    """Shared with parse_transcript_file's inline logic -- both need the same
    "string, ISO-8601, always end up timezone-aware" contract."""
    if not isinstance(raw, str) or not raw:
        return None
    try:
        timestamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp


def _assistant_visible_word_count(content: object) -> int:
    """Word count of only the *visible* text an assistant turn produced --
    ``type == "text"`` content blocks. Deliberately excludes ``thinking``
    blocks (never shown to Juniper) and ``tool_use``/``tool_result`` blocks
    (structured data, not prose) -- this is meant as a "how much do I have
    to read" proxy (Juniper's own framing, 2026-07-30), not a raw token
    count of everything the model produced internally."""
    if isinstance(content, str):
        return len(content.split())
    if not isinstance(content, list):
        return 0
    total = 0
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                total += len(text.split())
    return total


@dataclass(frozen=True)
class SessionUsageRecord:
    """One normalized ledger record per transcript file. Deliberately one
    record per *file*, not per logical "session" in the human sense -- a
    subagent's transcript (``<session>/subagents/agent-*.jsonl``) gets its
    own record rather than being folded into its parent's, because it
    represents its own real, separately-billed API usage. Folding subagent
    token/cost usage into "not really part of the ledger" would make the
    ledger's own stated purpose (real $ cost accounting) systematically
    undercount -- a single review/build session in this repo can dispatch
    many subagents whose combined token usage is a large fraction of the
    real total. ``is_subagent`` distinguishes the two for callers that want
    to report them separately or re-attribute later.
    """

    session_id: str
    transcript_path: Path
    is_subagent: bool
    models: tuple[str, ...]
    effort_tiers: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    assistant_turn_count: int
    assistant_word_count: int
    human_turn_count: int
    human_word_count: int
    started_at: datetime | None
    ended_at: datetime | None

    @property
    def duration_sec(self) -> float | None:
        if self.started_at is None or self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )


def parse_session_usage_record(path: Path | str) -> SessionUsageRecord | None:
    """One pass over ``path``, building a normalized usage record. Returns
    ``None`` for a transcript with zero assistant turns and zero real human
    turns -- e.g. a metadata-only or empty file -- rather than a record full
    of zeros that would misreport "checked, found nothing" the same as a
    real all-zero session.

    Malformed JSON lines are skipped, same "single corrupt line must not
    abort the whole file" guarantee as ``parse_transcript_file``."""
    file_path = Path(path)
    is_subagent = file_path.parent.name == "subagents"

    session_id: str | None = None
    models: list[str] = []
    effort_tiers: list[str] = []
    input_tokens = output_tokens = 0
    cache_creation_input_tokens = cache_read_input_tokens = 0
    assistant_turn_count = assistant_word_count = 0
    human_turn_count = human_word_count = 0
    started_at: datetime | None = None
    ended_at: datetime | None = None
    # A single logical assistant turn (thinking -> text -> N tool_use blocks)
    # is logged as multiple separate JSONL lines sharing one `message.id`,
    # each repeating the identical, final, cumulative `usage` for that turn.
    # Confirmed live 2026-08-11 by code review: summing per-*line* (the first
    # cut of this function) overcounted real tokens by 1.8x-2.7x and turn
    # count by ~2x across the full real corpus -- one message.id counted
    # once, first occurrence, is the real per-turn number. A missing/
    # non-string `id` (older transcript format, never observed live but not
    # ruled out) is never deduped -- every such line counts independently,
    # same as this function's original behavior, rather than silently
    # dropping data it can't safely recognize as a duplicate.
    seen_assistant_message_ids: set[str] = set()

    def _observe(timestamp: datetime | None) -> None:
        nonlocal started_at, ended_at
        if timestamp is None:
            return
        if started_at is None or timestamp < started_at:
            started_at = timestamp
        if ended_at is None or timestamp > ended_at:
            ended_at = timestamp

    with file_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue

            obj_session_id = obj.get("sessionId")
            if session_id is None and isinstance(obj_session_id, str):
                session_id = obj_session_id

            obj_type = obj.get("type")
            if obj_type in ("user", "assistant"):
                # Real transcripts carry timestamps on many other line types
                # too (attachment, pr-link, queue-operation, ...) -- a
                # session ID can get reused long after real chat activity
                # ended just to record something unrelated (confirmed live
                # 2026-08-11: a `pr-link` line appended 11.7 days after a
                # session's real last chat turn skewed duration_sec to
                # ~11.9 days for that one file). Restricting to actual
                # conversational turns keeps duration_sec meaning "real
                # elapsed engaged time", not "how long ago this file was
                # last touched for any reason."
                _observe(_parse_timestamp(obj.get("timestamp")))

            message = obj.get("message")
            if (
                obj_type == "assistant"
                and isinstance(message, dict)
                and message.get("model") != "<synthetic>"
            ):
                # model == "<synthetic>" is a harness-injected placeholder
                # turn, not a real model response -- excluded entirely, not
                # just from token counting. Confirmed live 2026-08-11: 111
                # occurrences carry isApiErrorMessage=True ("API Error:
                # Server error mid-response"), but a further 25 do not (a
                # broader class of synthetic/interrupt turn, same
                # model="<synthetic>" marker) -- checking the model field
                # directly catches both, where checking isApiErrorMessage
                # alone only caught the first. All-zero usage either way, so
                # harmless to token totals, but counting it in
                # `models`/`assistant_turn_count` would misreport a harness
                # placeholder as if a real model had answered.
                # Word count is accumulated for *every* line unconditionally,
                # not gated by the dedup check below -- confirmed live
                # 2026-08-11 (self-caught regression while fixing the token
                # overcount): the real "text" content block for a multi-line
                # turn does not reliably land on the first line of the
                # group (thinking/tool_use-only lines commonly come first),
                # so gating word-count behind "first occurrence of this
                # message.id" silently dropped ~66% of real assistant word
                # count in this same patch's first attempt. Safe to sum
                # across every line of a duplicate group regardless: at most
                # one line per real turn ever carries a `text`-type block
                # (verified against the real corpus), so this never
                # double-counts, it just correctly finds the text wherever
                # it actually landed.
                assistant_word_count += _assistant_visible_word_count(message.get("content"))

                message_id = message.get("id")
                # `and message_id` excludes an empty string too, not just
                # non-string/missing -- an empty id is falsy-but-a-str, so a
                # bare isinstance check would treat it as a real dedup key
                # and collide across unrelated turns that all happened to
                # have id=="". Zero real occurrences found in this repo's
                # corpus (verified live 2026-08-11), but the code should
                # match its own "missing/non-string id is never deduped"
                # contract exactly, not just in the common case.
                if isinstance(message_id, str) and message_id:
                    if message_id in seen_assistant_message_ids:
                        continue
                    seen_assistant_message_ids.add(message_id)
                assistant_turn_count += 1
                model = message.get("model")
                if isinstance(model, str) and model not in models:
                    models.append(model)
                effort = obj.get("effort")
                if isinstance(effort, str) and effort not in effort_tiers:
                    effort_tiers.append(effort)
                usage = message.get("usage")
                if isinstance(usage, dict):
                    input_tokens += usage.get("input_tokens") or 0
                    output_tokens += usage.get("output_tokens") or 0
                    cache_creation_input_tokens += usage.get("cache_creation_input_tokens") or 0
                    cache_read_input_tokens += usage.get("cache_read_input_tokens") or 0
            elif (
                obj_type == "user"
                and not is_subagent
                and _is_real_typed_human_turn(obj)
            ):
                # is_subagent transcripts' first "user" line is the
                # orchestrator's own Task-tool dispatch prompt -- a plain
                # string with promptSource/origin genuinely absent (not
                # because it's old data) and isSidechain=true. Confirmed
                # live 2026-08-11: essentially every real subagent
                # transcript's dispatch line otherwise passes this filter
                # and gets miscounted as something Juniper typed. Gating on
                # is_subagent is simpler and more robust than adding an
                # isSidechain check to the shared filter (which top-level
                # transcripts' own real human turns never carry either way).
                raw_text = obj["message"]["content"]
                cleaned = _strip_wrapper_tags(raw_text)
                if cleaned is not None:
                    human_turn_count += 1
                    human_word_count += len(cleaned.split())

    if assistant_turn_count == 0 and human_turn_count == 0:
        return None

    return SessionUsageRecord(
        session_id=session_id or file_path.stem,
        transcript_path=file_path,
        is_subagent=is_subagent,
        models=tuple(models),
        effort_tiers=tuple(effort_tiers),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        assistant_turn_count=assistant_turn_count,
        assistant_word_count=assistant_word_count,
        human_turn_count=human_turn_count,
        human_word_count=human_word_count,
        started_at=started_at,
        ended_at=ended_at,
    )


def iter_all_session_usage_records(
    root: Path | str = DEFAULT_PROJECTS_ROOT,
) -> Iterator[SessionUsageRecord]:
    """Convenience: one ``SessionUsageRecord`` per transcript file under
    ``root`` (skipping files that yield ``None`` -- nothing real to report).
    Same per-file ``OSError`` resilience as ``iter_all_human_messages`` --
    see that function's docstring for why (broken absolute symlinks, not a
    transient race, confirmed live 2026-08-11)."""
    for transcript_path in iter_transcript_files(root):
        try:
            record = parse_session_usage_record(transcript_path)
        except OSError:
            logger.warning(
                "claude_code_ingest_transcript_vanished_during_walk path=%s", transcript_path
            )
            continue
        if record is not None:
            yield record
