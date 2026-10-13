"""Publishers that refuse the reader's fetch tool. Skip them before a turn is spent.

A reading turn can only read through the harness fetch tool (Claude Code's
WebFetch, see orion/harness/reading_receipts.py). Some publishers refuse that
tool outright. Every seed on those sites then burns a full FCC turn (and, for
world_pulse, a Wallet A slot) and lands as terminal ``no_read_evidence``.

Live 2026-10-05..13: 35 of 37 non-YouTube ``no_read_evidence`` seeds were BBC
(18) or NPR (17); 0 BBC/NPR seeds were ever read. Raw tool_results from
``grammar_events`` (harness step atoms):

- BBC: ``Claude Code is unable to fetch from www.bbc.co.uk`` (26x) and
  ``... www.bbc.com`` (16x) -- the tool's own domain refusal, returned before
  any request; ``feeds.bbci.co.uk`` is refused the same way (2x). bbc.co.uk's
  robots.txt also disallows ClaudeBot / Claude-Web / anthropic-ai.
- NPR: ``timeout of 60000ms exceeded`` (172x). npr.org's robots.txt has
  ``User-agent: Claude-User / Disallow: /``, and its edge resets any request
  carrying that agent (``curl -A 'Claude-User ...'`` -> HTTP/2 INTERNAL_ERROR
  in 0.17s; the same URL with curl's default agent -> 200, 143 KB), so
  WebFetch hangs to its 60s timeout. text.npr.org disallows every agent.

These are deliberate publisher/tool policies, not transient failures. Orion
does not route around them (no other user agent, no archive mirror, no
lite-site scraping); the honest outcome is a clean skip that names the block.
A feed's own excerpt (1-3 sentences in world_pulse_article) is not a read and
is not promoted to one.

Add a site here only with a live refusal pulled from the tool trace.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import urlsplit

# Registrable site -> what the fetch tool actually returned (evidence above).
READER_BLOCKED_SITES: dict[str, str] = {
    "bbc.co.uk": "fetch_tool_domain_refusal",
    "bbc.com": "fetch_tool_domain_refusal",
    "bbci.co.uk": "fetch_tool_domain_refusal",
    "npr.org": "robots_disallows_claude_user",
}

SKIP_REASON_PREFIX = "source_blocks_reader"


def _host(url: str) -> str:
    try:
        return (urlsplit(str(url or "").strip()).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def reader_block_reason(url: str) -> Optional[str]:
    """``source_blocks_reader:<site>:<why>`` when the URL is on a site that
    refuses the reader's fetch tool, else None. Matches the site and any
    subdomain (www., feeds., amp.), never a lookalike suffix (notbbc.com)."""
    host = _host(url)
    if not host:
        return None
    for site, why in READER_BLOCKED_SITES.items():
        if host == site or host.endswith("." + site):
            return f"{SKIP_REASON_PREFIX}:{site}:{why}"
    return None
