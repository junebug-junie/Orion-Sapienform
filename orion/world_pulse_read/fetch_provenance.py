"""What a fetch actually returned, and which URLs a reading may hop to.

Two deterministic rules, both decided from tool-trace text, never model prose:

1. ``fetch_http_error_status`` -- WebFetch answers a 4xx/5xx with its own fixed
   status notice instead of a digest of the page::

       The server returned HTTP 404 Not Found.

       The response body was not retrieved. If this URL requires ...

   Live 2026-10-13: three Stage 1 re-entry reads of guessed CoreWeave URLs
   (``/architecture``, ``/infrastructure``, ``/gpu-cpu-compute``) each "read"
   exactly that 215-char notice and were marked ``done``; the 200-char floor in
   read_evidence.py cannot catch it (a real CoreWeave digest that day was 202
   chars). The 74 harness fetches with retained text (since 2026-10-11) showed
   the notice with codes 400/403/404/429, always as the first line. This is the fetch
   tool's status channel, the same kind of signal as the ``Redirect detected``
   notice reading_receipts.py already rejects -- one anchored line carrying a
   numeric code, not a list of error words.

2. ``hop_allowlist`` -- Stage 2's ``need_stage1_urls`` re-enters Stage 1, which
   costs an agent-GPU turn per URL. Only URLs that appear in text Orion actually
   fetched (or that a Stage 2 fetch itself reached with real content) may be
   followed. The CoreWeave hops above were model-guessed siblings that 404'd.
"""

from __future__ import annotations

import html
import re
from typing import Any, Iterable, Optional, Sequence
from urllib.parse import unquote, urljoin, urlsplit

# Anchored at the start of the tool_result: a page that merely *quotes* this
# sentence further down is still a page.
_WEBFETCH_STATUS = re.compile(r"\AThe server returned HTTP (\d{3})\b")

# Every tool whose status notice format is known. Firecrawl reports failure as
# ``success: false`` (handled in reading_receipts.py); Context Mode has no
# observed status notice.
_STATUS_NOTICE_TOOLS = frozenset({"WebFetch"})


def fetch_http_error_status(tool_name: str, text: Optional[str]) -> Optional[int]:
    """The HTTP error code when ``text`` is the fetch tool's own error notice."""
    if tool_name not in _STATUS_NOTICE_TOOLS or not text:
        return None
    match = _WEBFETCH_STATUS.match(text.lstrip())
    if match is None:
        return None
    code = int(match.group(1))
    return code if code >= 400 else None


# A URL inside prose or markdown. Stops at whitespace, quotes, angle brackets
# and backticks; trailing sentence punctuation is trimmed below.
_URL = re.compile(r"https?://[^\s<>\"'`]+", re.IGNORECASE)
_TRAILING = ".,;:!?*_"
_CLOSERS = {")": "(", "]": "[", "}": "{"}


def _trim(url: str) -> str:
    while url:
        last = url[-1]
        if last in _TRAILING:
            url = url[:-1]
        elif last in _CLOSERS and url.count(last) > url.count(_CLOSERS[last]):
            # ``(see https://x.org/a)`` -- the paren belongs to the prose. A
            # balanced one (``/wiki/Foo_(bar)``) belongs to the URL.
            url = url[:-1]
        else:
            break
    return url


# A link written without a scheme, as WebFetch digests often do ("details at
# nasa.gov/maxpower" -- live 2026-10-11, and the hop to it was a real 1.8k read).
# Needs a dotted host ending in a letters-only label AND a path, so "v1.5.2",
# "e.g." and bare product names never match; not preceded by a URL/email char.
_BARE = re.compile(
    r"(?<![\w@/.:-])((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,63}/[^\s<>\"'`]*)",
    re.IGNORECASE,
)


# A root-relative link target in markdown or HTML: ``](/docs/x)``, ``href="/paper"``.
_RELATIVE = re.compile(r"(?:\]\(|href=[\"'])(/[^\s)\"'<>]*)", re.IGNORECASE)


def urls_in_text(text: Optional[str], base_url: Optional[str] = None) -> list[str]:
    """http(s) URLs written in ``text`` (absolute, ``host.tld/path`` read as https,
    and -- given the fetched page's ``base_url`` -- root-relative link targets),
    in order, deduplicated. Bare ``host.tld/path`` can also match a file-name-like
    token (``llama.cpp/server``); that only widens the allowlist by a URL that was
    literally written in fetched text."""
    if not text:
        return []
    text = html.unescape(text)
    out: list[str] = []
    seen: set[str] = set()
    found = [m.group(0) for m in _URL.finditer(text)]
    stripped = _URL.sub(" ", text)
    found += ["https://" + m.group(1) for m in _BARE.finditer(stripped)]
    if base_url and hop_url_key(base_url):
        found += [urljoin(base_url, m.group(1)) for m in _RELATIVE.finditer(text)]
    for raw in found:
        url = _trim(raw)
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out


def hop_url_key(url: str) -> Optional[str]:
    """Identity for comparing a requested hop with a fetched link: scheme and
    host case-folded, ``www.`` dropped, trailing slash and fragment ignored,
    query kept (it can name a different document)."""
    try:
        parts = urlsplit(str(url or "").strip())
        host = (parts.hostname or "").lower().rstrip(".")
        port = f":{parts.port}" if parts.port and parts.port not in (80, 443) else ""
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not host:
        return None
    if host.startswith("www."):
        host = host[4:]
    path = unquote(parts.path).rstrip("/") or "/"
    query = f"?{parts.query}" if parts.query else ""
    # http and https name the same page for this purpose.
    return f"{host}{port}{path}{query}"


def hop_allowlist(
    texts: Iterable[str | tuple[Optional[str], str]], fetched_urls: Iterable[str] = ()
) -> set[str]:
    """Keys of every URL Orion actually saw: links written in fetched text (each
    item a text, or ``(page_url, text)`` so relative links resolve), plus URLs a
    fetch reached with real content (not a status notice)."""
    keys: set[str] = set()
    for item in texts:
        base, text = item if isinstance(item, tuple) else (None, item)
        for url in urls_in_text(text, base):
            key = hop_url_key(url)
            if key:
                keys.add(key)
    for url in fetched_urls:
        key = hop_url_key(url)
        if key:
            keys.add(key)
    return keys


HOP_REFUSED_NOT_IN_FETCHED_CONTENT = "url_not_in_fetched_content"


def hop_refusal_reason(url: str, allowlist: set[str]) -> Optional[str]:
    """``None`` when ``url`` may re-enter Stage 1, else the refusal label.
    A parent read with no valid evidence never gets here: Stage 2 skips it
    (world_pulse_read_stage2.py), so it cannot start a hop chain."""
    key = hop_url_key(url)
    if key is None or key not in allowlist:
        return HOP_REFUSED_NOT_IN_FETCHED_CONTENT
    return None


def reached_pages(fetches: Iterable[Any]) -> tuple[list[str], list[tuple[str, str]]]:
    """A turn's fetches (``SourceFetchEvidenceV1``) that returned a page, not an
    HTTP error notice: ``(urls reached, [(url, text)] for those carrying text)``."""
    urls: list[str] = []
    texts: list[tuple[str, str]] = []
    for f in fetches:
        if fetch_http_error_status(f.tool_name, f.content_text) is not None:
            continue
        urls.append(f.url)
        if f.content_text:
            texts.append((f.url, f.content_text))
    return urls, texts


def parent_read_state(
    evidence: Sequence[Any], rows: Sequence[Any]
) -> tuple[list[tuple[str, str]], bool]:
    """Stage 1 evidence (``SourceFetchEvidenceV1``) and its retained rows
    (``RetainedTextV1``, loaded with notices kept) -> ``([(url, page text)], dead)``.

    Dead only when every evidence row is retained, every distinct text loaded,
    and every one is an HTTP error notice. A missing or hash-failed text is
    "unknown", never "dead": that costs hops, not a whole Stage 2 pass."""
    tool_by_sha = {e.content_sha256: e.tool_name for e in evidence if e.content_sha256}
    pages = [(r.url, r.text) for r in rows
             if fetch_http_error_status(tool_by_sha.get(r.sha256, ""), r.text) is None]
    all_retained = bool(evidence) and all(e.content_sha256 for e in evidence)
    all_loaded = {r.sha256 for r in rows} == set(tool_by_sha)
    return pages, bool(rows) and all_retained and all_loaded and not pages
