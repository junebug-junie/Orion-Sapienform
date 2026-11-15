"""Reading choice point: the links Orion was shown after a read, and which ones Orion picked.

Personality rev 4, Move 1, row "Reading"
(docs/superpowers/specs/2026-10-16-orion-dynamic-personality-graph-design.md).

What already happened before this module: a Stage 2 pass ends with ``need_stage1_urls``, the
links Orion wants to read next. Each one that a fetched page actually contained re-enters Stage 1
as a new read (``fetch_provenance.hop_allowlist``). That list IS Orion's pick, and it is already a
structured JSON field of the call Stage 2 makes. Live 2026-11-08: "The world-pulse run queued the
PBS piece ..., and I picked the PolitiFact link myself" was such a hop. What was missing is the
other side of the choice: which links Orion saw and left.

So the Stage 2 prompt now shows a numbered list of the links Stage 1's read contained
(``next_read_options`` / ``render_next_read_section``), and after the pass this module turns
"shown" + "picked" into ``PersonalityChoiceV1`` rows (``reading_choices``). No new model call.

Market rank = the order the list shows the links: the order Stage 1's fetches were made, then the
order each link appears in that fetch's retained text. Nothing ranks them; it is page order.

The shown list is parsed back out of the prompt that was actually bound to the durable turn
(``parse_next_read_section``), never recomputed: the first committed prompt is authoritative
(orion/world_pulse_read/durable.py), and recomputing later would see a different "already read" set.
A prompt with no section (bound before this shipped, or the flag off) yields no choice.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional, Sequence
from urllib.parse import unquote, urlsplit

from orion.physiology.weights import endorsement_option_key
from orion.schemas.personality import (
    CHOICE_OPTION_TEXT_MAX,
    MAX_CHOICE_LIST,
    ChoiceOptionV1,
    PersonalityChoiceV1,
)
from orion.world_pulse_read.fetch_provenance import hop_url_key, urls_in_text
from orion.world_pulse_read.reader_contract import NEXT_READ_BEGIN, NEXT_READ_END

READING_FAMILY = "hub.world_pulse_read"
NEXT_READ_MAX_OPTIONS = 12
_LABEL_MAX = 200
# Not documents: a pick of one is never a read.
_ASSET_EXT = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".css", ".js", ".woff", ".woff2",
    ".ttf", ".mp4", ".mp3", ".webm", ".m4a", ".zip", ".gz",
)
_MD_LINK = re.compile(r"\[([^\]\n]{1,300})\]\((https?://[^)\s]+)\)")
_LINE = re.compile(r"^\s*\[(\d{1,3})\] (.+) -> (\S+)\s*$")
_SLUG_SPLIT = re.compile(r"[-_+.]+")
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class NextReadLink:
    url: str
    label: str
    rank: int

    @property
    def key(self) -> str:
        return option_key(self.url)

    @property
    def text(self) -> str:
        return option_text(self.label, self.url)


def option_key(url: str) -> str:
    """Stable id for a link: its hop identity hashed, under the reading family."""
    ident = hop_url_key(url) or url
    return endorsement_option_key(READING_FAMILY, "link:" + hashlib.sha256(ident.encode("utf-8")).hexdigest()[:16])


def _host(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _slug_words(url: str) -> str:
    """Plain words from a URL's path when the page gave no link text: the last path parts."""
    try:
        path = unquote(urlsplit(url).path)
    except ValueError:
        return ""
    parts = [p for p in path.split("/") if p and not p.isdigit()]
    words = " ".join(_SLUG_SPLIT.sub(" ", p) for p in parts[-2:])
    return _WS.sub(" ", words).strip()


def _clean(text: str) -> str:
    text = _WS.sub(" ", str(text or "")).strip().strip("-*•:|").strip()
    # Never let a label carry our list syntax or a fence.
    return text.replace("<<<", "").replace(" -> ", " - ")[:_LABEL_MAX].strip()


def option_text(label: str, url: str) -> str:
    """Subject first, plain words: "<what the link says> (<host>)"; at most 400 chars."""
    host = _host(url)
    body = _clean(label) or _slug_words(url) or url
    text = f"{body} ({host})" if host and host not in body else body
    return text[:CHOICE_OPTION_TEXT_MAX]


def link_labels(text: str) -> dict[str, str]:
    """hop key -> the words a page wrote for a link: markdown anchor text, else its line."""
    out: dict[str, str] = {}
    for m in _MD_LINK.finditer(text):
        k = hop_url_key(m.group(2))
        if k and k not in out and _clean(m.group(1)):
            out[k] = _clean(m.group(1))
    for line in text.splitlines():
        for url in urls_in_text(line):
            k = hop_url_key(url)
            if not k or k in out:
                continue
            rest = _clean(_MD_LINK.sub(lambda m: m.group(1), line).replace(url, " "))
            if len(rest) >= 4:
                out[k] = rest
    return out


def _is_asset(url: str) -> bool:
    try:
        path = urlsplit(url).path.lower()
    except ValueError:
        return True
    return path.endswith(_ASSET_EXT)


def candidate_links(
    texts: Iterable[tuple[str, str]], *, exclude_urls: Iterable[str] = ()
) -> list[NextReadLink]:
    """Every link written in Stage 1's retained texts, in fetch order then page order, deduplicated
    by hop identity. The pages Stage 1 itself fetched (``exclude_urls``) and asset files are left out.
    Ranks are 1.. in that order. Pure."""
    skip = {k for k in (hop_url_key(u) for u in exclude_urls) if k}
    seen: set[str] = set()
    out: list[NextReadLink] = []
    for page_url, text in texts:
        labels = link_labels(text or "")
        for url in urls_in_text(text, page_url):
            k = hop_url_key(url)
            if not k or k in skip or k in seen or _is_asset(url):
                continue
            seen.add(k)
            out.append(NextReadLink(url=url, label=labels.get(k, ""), rank=len(out) + 1))
    return out


def next_read_options(
    texts: Iterable[tuple[str, str]],
    *,
    exclude_urls: Iterable[str] = (),
    already_read_keys: Iterable[str] = (),
    limit: int = NEXT_READ_MAX_OPTIONS,
) -> list[NextReadLink]:
    """The links the prompt shows: ``candidate_links`` minus links already read, first ``limit``,
    re-ranked 1..n in the order shown. Pure."""
    done = set(already_read_keys)
    kept = [c for c in candidate_links(texts, exclude_urls=exclude_urls) if hop_url_key(c.url) not in done]
    return [NextReadLink(url=c.url, label=c.label, rank=i) for i, c in enumerate(kept[: max(0, limit)], start=1)]


def render_next_read_section(options: Sequence[NextReadLink]) -> str:
    """The prompt block (input data, outside the reader contract fence). Empty when no options."""
    if not options:
        return ""
    lines = [f"[{o.rank}] {o.text} -> {o.url}" for o in options]
    return (
        "\nLinks written in what you read, in the order they appeared (input data). You may "
        "list any of them in need_stage1_urls to read next, a link you fetched yourself this "
        "turn, or none:\n"
        f"{NEXT_READ_BEGIN}\n" + "\n".join(lines) + f"\n{NEXT_READ_END}\n"
    )


def parse_next_read_section(prompt: Optional[str]) -> list[NextReadLink]:
    """The options a bound prompt showed. Empty when the prompt has no section, more than one, or
    a malformed one (fail-open: no choice is recorded, the read is unaffected)."""
    text = str(prompt or "")
    if text.count(NEXT_READ_BEGIN) != 1 or text.count(NEXT_READ_END) != 1:
        return []
    start = text.index(NEXT_READ_BEGIN) + len(NEXT_READ_BEGIN)
    end = text.index(NEXT_READ_END)
    if end < start:
        return []
    out: list[NextReadLink] = []
    for line in text[start:end].strip("\n").splitlines():
        m = _LINE.match(line)
        if m is None or int(m.group(1)) != len(out) + 1:
            return []
        url = m.group(3)
        host = _host(url)
        label = m.group(2)
        suffix = f" ({host})"
        if host and label.endswith(suffix):
            label = label[: -len(suffix)]
        out.append(NextReadLink(url=url, label=label, rank=len(out) + 1))
    return out


def reading_choices(
    *,
    seed_id: str,
    shown: Sequence[NextReadLink],
    picked_urls: Sequence[str],
    allowlist: set[str],
    occurred_at: datetime,
    labels_for_unlisted: Optional[dict[str, str]] = None,
    act_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    situation: Optional[dict] = None,
) -> tuple[list[PersonalityChoiceV1], str]:
    """Shown links + Orion's ``need_stage1_urls`` -> choice rows, and a reason label for the log.

    * A picked URL that matches a shown link is that option (its market rank).
    * A picked URL Orion found itself this turn (in the hop allowlist, not in the list) is a chosen
      option with no market rank: Orion went past the list.
    * A picked URL outside the allowlist (guessed, never fetched) is not an option and is ignored.
    * Nothing picked at all, with links shown: ``declined_all`` (one row, every shown link declined).
    * Guessed URLs only: no row (Orion wanted more but named nothing real; not a decline).
    * Every shown link picked: no row (nothing declined, so no revealed preference).
    One row per pick, ``picks_in_decision`` = the number of picks, like curiosity."""
    if not shown:
        return [], "no_links_shown"
    by_key = {hop_url_key(o.url): o for o in shown}
    labels_for_unlisted = labels_for_unlisted or {}
    chosen: list[ChoiceOptionV1] = []
    chosen_hops: set[str] = set()
    any_named = False
    for url in picked_urls:
        k = hop_url_key(url)
        if not k:
            continue
        any_named = True
        if k in chosen_hops:
            continue
        if k in by_key:
            o = by_key[k]
            chosen.append(ChoiceOptionV1(key=o.key, text=o.text, market_rank=o.rank))
        elif k in allowlist:
            chosen.append(ChoiceOptionV1(key=option_key(url), text=option_text(labels_for_unlisted.get(k, ""), url)))
        else:
            continue
        chosen_hops.add(k)
    alternatives = [ChoiceOptionV1(key=o.key, text=o.text, market_rank=o.rank)
                    for k, o in by_key.items() if k not in chosen_hops][:MAX_CHOICE_LIST]
    if not alternatives:
        return [], "all_shown_picked"
    sit = dict(situation or {})
    sit["links_shown"] = len(shown)
    if not chosen:
        if any_named:
            return [], "only_unfetched_urls"
        try:
            return [PersonalityChoiceV1(
                choice_id=f"reading:{seed_id}:none"[:300], lane="reading", occurred_at=occurred_at,
                alternatives=alternatives, declined_all=True, situation=sit,
                act_id=act_id, correlation_id=correlation_id)], "declined_all"
        except Exception:  # noqa: BLE001 - a bad row never touches the read
            return [], "invalid_model"
    picks = min(len(chosen), MAX_CHOICE_LIST)
    out: list[PersonalityChoiceV1] = []
    for c in chosen[:MAX_CHOICE_LIST]:
        try:
            out.append(PersonalityChoiceV1(
                choice_id=f"reading:{seed_id}:{c.key.rsplit(':', 1)[-1]}"[:300], lane="reading",
                occurred_at=occurred_at, chosen=c, alternatives=alternatives, picks_in_decision=picks,
                situation=sit, act_id=act_id, correlation_id=correlation_id))
        except Exception:  # noqa: BLE001 - one bad pick never sinks the others
            continue
    return (out, "picked") if out else ([], "invalid_model")


ALREADY_READ_URLS_SQL = """
SELECT DISTINCT r.url FROM world_pulse_read_seed r
WHERE r.url = ANY($1::text[]) AND r.duplicate_of IS NULL AND r.status = 'done'
  AND (r.handoff_json->'read_evidence'->0) IS NOT NULL
"""


async def publish_choice(bus, choice: PersonalityChoiceV1, *, source) -> bool:
    """One choice on ``orion:personality:choice`` (durable-runs writes ``personality_choice``).
    Best-effort; never raises. The correlation id is derived from ``choice_id``, so a re-publish
    is the same row."""
    import logging
    from uuid import NAMESPACE_URL, uuid5

    from orion.core.bus.bus_schemas import BaseEnvelope
    from orion.schemas.personality import PERSONALITY_CHOICE_CHANNEL, PERSONALITY_CHOICE_KIND

    log = logging.getLogger(__name__)
    if bus is None:
        log.warning("reading_choice_not_published choice_id=%s reason=no_bus", choice.choice_id)
        return False
    try:
        env = BaseEnvelope(kind=PERSONALITY_CHOICE_KIND, source=source,
                           correlation_id=uuid5(NAMESPACE_URL, choice.choice_id),
                           payload=choice.model_dump(mode="json"))
        await bus.publish(PERSONALITY_CHOICE_CHANNEL, env)
    except Exception as exc:  # noqa: BLE001
        log.warning("reading_choice_publish_failed choice_id=%s err=%s", choice.choice_id, type(exc).__name__)
        return False
    return True
