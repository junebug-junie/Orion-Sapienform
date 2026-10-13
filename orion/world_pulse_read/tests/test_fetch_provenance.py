from orion.world_pulse_read.fetch_provenance import (
    HOP_REFUSED_NOT_IN_FETCHED_CONTENT,
    fetch_http_error_status,
    hop_allowlist,
    hop_refusal_reason,
    hop_url_key,
    urls_in_text,
)

_NOTICE_TAIL = (
    "\n\nThe response body was not retrieved. If this URL requires authentication, use an "
    "authenticated tool (e.g. `gh` for GitHub, or an MCP-provided fetch tool) instead of WebFetch."
)


def test_every_live_status_notice_is_an_http_error():
    # Codes seen in retained live WebFetch results, 2026-10-11..13 (401 from the
    # Falcon-ASR read, where the model quoted it).
    for line, code in [
        ("The server returned HTTP 404 Not Found.", 404),
        ("The server returned HTTP 403 Forbidden.", 403),
        ("The server returned HTTP 400 Bad Request.", 400),
        ("The server returned HTTP 429 Too Many Requests.", 429),
        ("The server returned HTTP 401 Unauthorized.", 401),
    ]:
        assert fetch_http_error_status("WebFetch", line + _NOTICE_TAIL) == code


def test_live_notice_lengths_match_the_unretained_rows():
    # Rows from before text retention carry only content_chars: 215 (404/403)
    # and 218 (401) are exactly these notices.
    assert len("The server returned HTTP 404 Not Found." + _NOTICE_TAIL) == 215
    assert len("The server returned HTTP 401 Unauthorized." + _NOTICE_TAIL) == 218


def test_a_page_is_not_an_http_error():
    assert fetch_http_error_status("WebFetch", "CoreWeave offers bare-metal NVIDIA GPUs.") is None
    # Quoting the notice further down a page does not make the page an error.
    assert fetch_http_error_status(
        "WebFetch", "Our docs explain: The server returned HTTP 404 Not Found. means gone."
    ) is None
    assert fetch_http_error_status("WebFetch", "The server returned HTTP 200 OK.") is None
    assert fetch_http_error_status("WebFetch", None) is None
    # Other tools have their own failure signals (reading_receipts.py).
    assert fetch_http_error_status("mcp__firecrawl__scrape", "The server returned HTTP 404 Not Found.") is None


def test_urls_in_text_trims_prose_punctuation_but_keeps_balanced_parens():
    text = (
        "See https://ex.com/a. Also (https://ex.com/b) and [x](https://ex.com/c?q=1), "
        "plus https://en.wikipedia.org/wiki/Foo_(bar) and `https://ex.com/d`."
    )
    assert urls_in_text(text) == [
        "https://ex.com/a", "https://ex.com/b", "https://ex.com/c?q=1",
        "https://en.wikipedia.org/wiki/Foo_(bar)", "https://ex.com/d",
    ]
    assert urls_in_text("") == [] and urls_in_text(None) == []


def test_hop_url_key_ignores_www_slash_scheme_and_fragment_only():
    assert hop_url_key("https://www.Ex.com/a/") == hop_url_key("http://ex.com/a#top")
    assert hop_url_key("https://ex.com/a?x=1") != hop_url_key("https://ex.com/a")
    assert hop_url_key("https://ex.com/a") != hop_url_key("https://ex.com/b")
    assert hop_url_key("ftp://ex.com/a") is None
    assert hop_url_key("not a url") is None


def test_hop_refusal_follows_the_allowlist():
    allow = hop_allowlist(["Read more at https://ex.com/paper."], ["https://ex.com/fetched"])
    assert hop_refusal_reason("https://www.ex.com/paper/", allow) is None
    assert hop_refusal_reason("https://ex.com/fetched", allow) is None
    assert hop_refusal_reason("https://ex.com/guessed", allow) == HOP_REFUSED_NOT_IN_FETCHED_CONTENT
    assert hop_refusal_reason("garbage", allow) == HOP_REFUSED_NOT_IN_FETCHED_CONTENT


def test_schemeless_links_in_a_digest_count_but_versions_do_not():
    text = ("Culminates in MAX POWER; details at nasa.gov/maxpower. Built on llama.cpp v1.5.2, "
            "see e.g. the docs; mail press@nasa.gov/x is not a link to follow.")
    assert urls_in_text(text) == ["https://nasa.gov/maxpower"]
    allow = hop_allowlist([text])
    assert hop_refusal_reason("https://www.nasa.gov/maxpower", allow) is None


def test_relative_links_resolve_against_the_fetched_page():
    text = 'See [the paper](/abs/2609.10712) and <a href="/pdf/2609.10712.pdf">pdf</a>.'
    allow = hop_allowlist([("https://arxiv.org/list/cs.AI", text)])
    assert hop_refusal_reason("https://arxiv.org/abs/2609.10712", allow) is None
    assert hop_refusal_reason("https://arxiv.org/pdf/2609.10712.pdf", allow) is None
    # Without the page URL a relative link names nothing.
    assert hop_refusal_reason("https://arxiv.org/abs/2609.10712", hop_allowlist([text])) is not None


def test_default_port_percent_encoding_and_html_entities_normalize():
    assert hop_url_key("https://ex.com:443/~a") == hop_url_key("https://ex.com/%7Ea")
    assert hop_url_key("https://ex.com:8443/a") != hop_url_key("https://ex.com/a")
    allow = hop_allowlist(['<a href="https://ex.com/s?a=1&amp;b=2">x</a>'])
    assert hop_refusal_reason("https://ex.com/s?a=1&b=2", allow) is None


def _ev(url, sha):
    from orion.schemas.reading import SourceFetchEvidenceV1

    return SourceFetchEvidenceV1(url=url, tool_name="WebFetch", content_chars=300, content_sha256=sha)


def _row(url, sha, text):
    from orion.world_pulse_read.fetch_text import RetainedTextV1

    return RetainedTextV1(sha256=sha, url=url, representation="tool_digest", text=text)


def test_parent_is_dead_only_when_every_retained_text_is_a_notice():
    from orion.world_pulse_read.fetch_provenance import parent_read_state

    notice = "The server returned HTTP 404 Not Found." + _NOTICE_TAIL
    u = "https://ex.com/a"
    pages, dead = parent_read_state([_ev(u, "a" * 64)], [_row(u, "a" * 64, notice)])
    assert pages == [] and dead
    # A page whose text did not load (missing / hash mismatch) is unknown, not dead.
    pages, dead = parent_read_state([_ev(u, "a" * 64), _ev(u, "b" * 64)], [_row(u, "a" * 64, notice)])
    assert pages == [] and not dead
    pages, dead = parent_read_state([_ev(u, "a" * 64), _ev(u, "b" * 64)],
                                    [_row(u, "a" * 64, notice), _row(u, "b" * 64, "real page " * 30)])
    assert pages == [(u, "real page " * 30)] and not dead
