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

Only links to something else to read are options (#3443, live 2026-11-15: the paper's own DOI, a
Creative Commons deed and XMP namespace URIs were shown, and Orion "picked" the paper's own arXiv
page). ``non_content_reason`` drops a link by its structure, never by words in its text:

* ``same_document``: the link carries the read page's own identifier (its arXiv id or DOI), or its
  query embeds the read page's URL. That is the page again: another format, a citation lookup, a
  share button.
* ``license``: a path segment ``license`` / ``licenses`` / ``publicdomain`` (a licence deed or file).
* ``schema_namespace``: a vocabulary URI, not a document: host ``schema.org`` / ``purl.org`` /
  ``xmlns.com``, a host whose first label is ``ns`` (``ns.adobe.com``), or ``w3.org/<year>/...``.
* ``code_only``: the link is written only inside code: a line of a fenced block, or a backtick span,
  that holds more than bare URLs (``xmlns:dc="http://..."``, ``curl https://...``, a URL template).
  A digest that wraps a URL in backticks or lists URLs in a fence is formatting, not code: live
  digests do both for most links.

Option text is the link's anchor text, else the sentence the link sits in with URLs and markdown
taken out, else words from the URL's path.
"""

from __future__ import annotations

import hashlib
import html
import re
from collections import Counter
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
# Structural drop rules (module docstring). Short, documented sets: registries, not topic words.
_LICENSE_SEGMENTS = frozenset({"license", "licenses", "publicdomain"})
_NAMESPACE_HOSTS = frozenset({"schema.org", "purl.org", "xmlns.com"})
_YEAR_SEGMENT = re.compile(r"^(?:19|20)\d\d$")
# Identifiers that name one document whatever URL carries them.
# Not inside a longer number ("12611.17316", "1.2611.17316"); "arXiv.2611.17316" is fine.
_ARXIV_ID = re.compile(r"(?<!\d)(?<!\d\.)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)")
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s?#&\"'<>`]+)", re.IGNORECASE)
_FENCE = re.compile(r"^[ \t]*(```|~~~)[^\n]*\n.*?^[ \t]*\1[ \t]*$", re.MULTILINE | re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_URL_DRESSING = " \t\"'<>()[]{},;:."
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_MARKUP = re.compile(r"\*\*|__|`|^#+\s*|^\s*(?:[-*•+]|\d{1,3}[.)])\s+")
_EMPTY_BRACKETS = re.compile(r"\(\s*\)|\[\s*\]|<\s*>")
_EMPTY_SEPARATORS = re.compile(r"([|;,])(?:\s*[|;,])+")  # "a | | b" once its URL is out -> "a | b"


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


def _id_words(url: str) -> str:
    """"arXiv 2611.07965" / "DOI 10.1000/x" for a link that names a paper by identifier, else ""."""
    ids = sorted(document_ids(url))
    arxiv = next((i for i in ids if i.startswith("arxiv:")), None)
    if arxiv:
        return "arXiv " + arxiv.split(":", 1)[1]
    doi = next((i for i in ids if i.startswith("doi:")), None)
    return "DOI " + doi.split(":", 1)[1] if doi else ""


_EDGE = " \t-*•:;,|—–→←>"
_QUOTES = "\"“”"


def _clean(text: str) -> str:
    text = _WS.sub(" ", str(text or "")).strip(_EDGE)
    # A wrapping pair of quotes goes ('“A title” —' -> 'A title'); a quote inside the text stays.
    while len(text) > 1 and text[0] in _QUOTES and text[-1] in _QUOTES \
            and not any(q in text[1:-1] for q in _QUOTES):
        text = text[1:-1].strip(_EDGE)
    # Never let a label carry our list syntax or a fence.
    return text.replace("<<<", "").replace(" -> ", " - ")[:_LABEL_MAX].strip()


def option_text(label: str, url: str) -> str:
    """Subject first, plain words: "<what the link says> (<host>)"; at most 400 chars."""
    host = _host(url)
    body = _clean(label) or _id_words(url) or _slug_words(url) or url
    text = f"{body} ({host})" if host and host not in body else body
    return text[:CHOICE_OPTION_TEXT_MAX]


def _plain(text: str) -> str:
    """Markdown links to their words, URLs and markup out, empty brackets gone."""
    text = _MD_LINK.sub(lambda m: m.group(1), text)
    for url in urls_in_text(text):
        text = text.replace(url, " ").replace(url.removeprefix("https://"), " ")
    text = _MARKUP.sub("", text.strip())
    text = _EMPTY_SEPARATORS.sub(r"\1", _EMPTY_BRACKETS.sub(" ", text))
    return _clean(_WS.sub(" ", text))


def _sentence_with(line: str, url: str) -> str:
    """The words of ``line`` that belong to ``url``: the sentence holding it (a URL's own dots never
    end a sentence: the split needs whitespace after the stop). When that sentence holds other URLs
    too ("Alerts -> <url> ; Brochures -> <url>"), only the words between the previous URL and this
    one, else the words after it up to the next one."""
    if url not in line:  # a scheme-less link (``nasa.gov/x``) is returned with "https://" put in front
        url = url.removeprefix("https://")
    sentence = next((x for x in _SENTENCE_END.split(line) if url in x), line)
    others = [u if u in sentence else u.removeprefix("https://")
              for u in urls_in_text(sentence) if url not in (u, u.removeprefix("https://"))]
    if not others:
        return sentence
    at = sentence.find(url)
    if at < 0:
        return sentence
    before, after = sentence[:at], sentence[at + len(url):]
    for u in others:
        if u in before:
            before = before[before.rfind(u) + len(u):]
        if u in after:
            after = after[:after.find(u)]
    return before if len(_plain(before)) >= 4 else after


_LIST_ITEM = re.compile(r"^(\s*)(?:[-*+•]|\d{1,3}[.)])\s+")


def _heads(lines: Sequence[str]) -> list[int]:
    """For each line, the index of the line that heads its block: a list item heads itself and the
    more-indented lines under it; other lines are headed by the first line of their paragraph."""
    heads: list[int] = []
    para = 0
    items: list[tuple[int, int]] = []  # (indent, line index) of list items in this paragraph
    for i, line in enumerate(lines):
        if not line.strip():
            para, items = i + 1, []
            heads.append(i)
            continue
        indent = len(line) - len(line.lstrip())
        m = _LIST_ITEM.match(line)
        if m:
            items = [it for it in items if it[0] < indent] + [(indent, i)]
            heads.append(i)
            continue
        owner = [idx for ind, idx in items if ind < indent]
        heads.append(owner[-1] if owner else para)
    return heads


def link_labels(text: str) -> dict[str, str]:
    """hop key -> the words a page wrote for a link: markdown anchor text, else the sentence it sits
    in (URLs and markup taken out). A label two or more links share ("Full URL:") does not tell them
    apart: those take the head line of their list item or paragraph (the title above the link)."""
    out: dict[str, str] = {}
    text = html.unescape(text)  # as urls_in_text does, so the hop keys agree
    lines = text.splitlines()
    heads = _heads(lines)
    head_of: dict[str, str] = {}
    for i, line in enumerate(lines):
        for m in _MD_LINK.finditer(line):
            k = hop_url_key(m.group(2))
            label = _plain(m.group(1))
            if k and k not in out and label:
                out[k] = label
                head_of.setdefault(k, _plain(lines[heads[i]]) if heads[i] != i else "")
        for url in urls_in_text(line):
            k = hop_url_key(url)
            if not k or k in out:
                continue
            rest = _plain(_sentence_with(line, url))
            if len(rest) >= 4:
                out[k] = rest
                head_of.setdefault(k, _plain(lines[heads[i]]) if heads[i] != i else "")
    shared = Counter(out.values())
    for k, label in out.items():
        head = head_of.get(k, "")
        if shared[label] > 1 and len(head) >= 4 and shared.get(head, 0) <= 1:
            out[k] = head
    return out


def _only_urls(text: str) -> bool:
    """Is ``text`` nothing but URLs (and quotes, brackets, separators around them)?"""
    rest = text
    for url in urls_in_text(text):
        # A scheme-less link (``nasa.gov/x``) is returned with "https://" put in front.
        rest = rest.replace(url, " ").replace(url.removeprefix("https://"), " ")
    return not rest.strip(_URL_DRESSING)


def _code_free(text: str) -> str:
    """``text`` with real code blanked. A fenced block keeps only its lines that are bare URLs (live
    digests list a page's links in ```text fences); a backtick span is blanked unless it is a bare URL."""

    def fence(m: re.Match) -> str:
        return "\n".join(line for line in m.group(0).splitlines()[1:-1] if _only_urls(line))

    text = _FENCE.sub(fence, html.unescape(text or ""))
    return _INLINE_CODE.sub(lambda m: m.group(0) if _only_urls(m.group(1)) else " ", text)


def document_ids(url: str) -> set[str]:
    """Identifiers that name the document behind ``url``: ``arxiv:<id>`` (an arXiv URL, or an arXiv
    DOI) and ``doi:<doi>`` (a DOI in its path). Empty for an ordinary page."""
    try:
        parts = urlsplit(url)
        path = unquote(parts.path)
    except ValueError:
        return set()
    host = _host(url)
    ids: set[str] = set()
    for m in _DOI.finditer(path):
        doi = m.group(1).rstrip(".,;)").lower()
        ids.add("doi:" + doi)
        a = _ARXIV_ID.search(doi) if doi.startswith("10.48550/") else None
        if a:
            ids.add("arxiv:" + a.group(1))
    if host == "arxiv.org" or host.endswith(".arxiv.org"):
        a = _ARXIV_ID.search(path)
        if a:
            ids.add("arxiv:" + a.group(1))
    return ids


def _carries_document(url: str, ids: set[str], page_keys: set[str]) -> bool:
    """Does ``url`` name the read page itself: one of its identifiers anywhere in the URL, or the
    page's own URL inside the query (a share or lookup link)."""
    flat = unquote(url).lower()
    for ident in ids:
        kind, _, value = ident.partition(":")
        if kind == "arxiv" and re.search(r"(?<!\d)(?<!\d\.)" + re.escape(value) + r"(?!\d)", flat):
            return True
        if kind == "doi" and value in flat:
            return True
    try:
        query = unquote(urlsplit(url).query)
    except ValueError:
        return False
    if not query or not page_keys:
        return False
    return bool(page_keys & _hop_keys_in(query))


def _hop_keys_in(text: str) -> set[str]:
    """Hop keys of every URL written in ``text``."""
    return {k for k in (hop_url_key(u) for u in urls_in_text(text)) if k}


def non_content_reason(url: str, *, ids: set[str] = frozenset(), page_keys: set[str] = frozenset()) -> Optional[str]:
    """Why ``url`` is not something else to read (module docstring), or None when it is. Pure."""
    try:
        parts = urlsplit(url)
        segs = [s.lower() for s in unquote(parts.path).split("/") if s]
    except ValueError:
        return "unparseable"
    host = _host(url)
    if _carries_document(url, set(ids), set(page_keys)):
        return "same_document"
    if _LICENSE_SEGMENTS & set(segs):
        return "license"
    if host in _NAMESPACE_HOSTS or host.split(".", 1)[0] == "ns" \
            or (host == "w3.org" and segs and _YEAR_SEGMENT.match(segs[0])):
        return "schema_namespace"
    return None


def _is_asset(url: str) -> bool:
    try:
        path = urlsplit(url).path.lower()
    except ValueError:
        return True
    return path.endswith(_ASSET_EXT)


def _is_site_nav(url: str, page_host: str) -> bool:
    """A link back into the read page's own site at path depth 0-1 (home, /about, /newshour):
    navigation a fetch digest lists first, not a piece to read. Links to other sites always stay."""
    if not page_host or _host(url) != page_host:
        return False
    try:
        parts = [p for p in urlsplit(url).path.split("/") if p]
    except ValueError:
        return True
    return len(parts) <= 1 and not urlsplit(url).query


def classified_links(
    texts: Iterable[tuple[str, str]], *, exclude_urls: Iterable[str] = ()
) -> list[tuple[NextReadLink, Optional[str]]]:
    """Every link written in Stage 1's retained texts, in fetch order then page order, deduplicated by
    hop identity, each with why it is not an option (None: it is one). The pages Stage 1 itself
    fetched (``exclude_urls``), asset files and the read site's own top-level navigation
    (``_is_site_nav``) are not listed at all. Ranks are list positions 1.. here. Pure."""
    texts = list(texts)
    excluded = [u for u in exclude_urls if u]
    skip = {k for k in (hop_url_key(u) for u in excluded) if k}
    own = [*excluded, *(p for p, _ in texts if p)]
    ids: set[str] = set().union(*(document_ids(u) for u in own)) if own else set()
    page_keys = {k for k in (hop_url_key(u) for u in own) if k}
    seen: set[str] = set()
    out: list[tuple[NextReadLink, Optional[str]]] = []
    for page_url, text in texts:
        labels = link_labels(text or "")
        page_host = _host(page_url or "")
        prose = {k for k in (hop_url_key(u) for u in urls_in_text(_code_free(text or ""), page_url)) if k}
        for url in urls_in_text(text, page_url):
            k = hop_url_key(url)
            if not k or k in skip or k in seen or _is_asset(url) or _is_site_nav(url, page_host):
                continue
            seen.add(k)
            reason = non_content_reason(url, ids=ids, page_keys=page_keys)
            if reason is None and k not in prose:
                reason = "code_only"
            out.append((NextReadLink(url=url, label=labels.get(k, ""), rank=len(out) + 1), reason))
    return out


def candidate_links(
    texts: Iterable[tuple[str, str]], *, exclude_urls: Iterable[str] = ()
) -> list[NextReadLink]:
    """The readable links of ``classified_links`` (reason None), re-ranked 1.. in page order. Pure."""
    kept = [c for c, reason in classified_links(texts, exclude_urls=exclude_urls) if reason is None]
    return [NextReadLink(url=c.url, label=c.label, rank=i) for i, c in enumerate(kept, start=1)]


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
    * A picked URL outside the shown list but in the hop allowlist is a chosen option with no market
      rank: Orion went past the list. That is a link Orion fetched or met this turn, or a parent-page
      link the list left out (past the cap, navigation, already read). ``situation["unlisted_picks"]``
      counts them.
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
    sit["unlisted_picks"] = sum(1 for c in chosen if c.market_rank is None)
    if not chosen:
        if any_named:
            return [], "only_unfetched_urls"
        try:
            return [PersonalityChoiceV1(
                choice_id=f"reading:{seed_id[:250]}:none", lane="reading", occurred_at=occurred_at,
                alternatives=alternatives, declined_all=True, situation=sit,
                act_id=act_id, correlation_id=correlation_id)], "declined_all"
        except Exception:  # noqa: BLE001 - a bad row never touches the read
            return [], "invalid_model"
    picks = min(len(chosen), MAX_CHOICE_LIST)
    out: list[PersonalityChoiceV1] = []
    for c in chosen[:MAX_CHOICE_LIST]:
        try:
            out.append(PersonalityChoiceV1(
                choice_id=f"reading:{seed_id[:250]}:{c.key.rsplit(':', 1)[-1]}", lane="reading",
                occurred_at=occurred_at, chosen=c, alternatives=alternatives, picks_in_decision=picks,
                situation=sit, act_id=act_id, correlation_id=correlation_id))
        except Exception:  # noqa: BLE001 - one bad pick never sinks the others
            continue
    return (out, "picked") if out else ([], "invalid_model")


# Exact URL match, as READ_URL_SQL: a www/slash/scheme variant of a read page is still shown, and
# picking it is a no-op hop (ALREADY_READ). Pending reads are shown too.
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
