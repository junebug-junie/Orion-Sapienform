from orion.schemas.reading import SourceFetchEvidenceV1
from orion.world_pulse_read.read_evidence import (
    MIN_SOURCE_CONTENT_CHARS,
    NO_READ_EVIDENCE,
    NO_READ_EVIDENCE_UNREPORTED,
    parse_source_fetches,
    same_site,
    source_read_evidence,
)
from orion.world_pulse_read.retry import is_transient_failure

SEED = "https://www.networkworld.com/article/3562856/nvidia-latest-news-and-insights.html"


def _f(url, chars=1000, tool="WebFetch"):
    return SourceFetchEvidenceV1(url=url, tool_name=tool, content_chars=chars)


def test_same_site_accepts_www_and_subdomains_only():
    assert same_site(SEED, "https://networkworld.com/article/3562856/")
    assert same_site("https://arxiv.org/abs/2310.19279", "https://export.arxiv.org/api/query?id_list=2310.19279")
    assert same_site("https://nvidianews.nvidia.com/news/rubin", "https://www.nvidia.com/en-us/rubin")
    assert not same_site(SEED, "https://www.tomshardware.com/news/nvidia")
    assert not same_site(SEED, "not a url")


def test_live_hollow_turn_has_no_evidence():
    # finding:60d59b10...:9b084fc0f1583da0 made zero tool calls.
    assert source_read_evidence(SEED, []) == []


def test_evidence_needs_the_source_site_and_real_content():
    fetches = [
        _f("https://www.networkworld.com/article/3562856/", MIN_SOURCE_CONTENT_CHARS),
        _f("https://www.networkworld.com/x", MIN_SOURCE_CONTENT_CHARS - 1),
        _f("https://other.example.org/story", 5000),
    ]
    assert [f.content_chars for f in source_read_evidence(SEED, fetches)] == [MIN_SOURCE_CONTENT_CHARS]


def test_parse_distinguishes_unreported_from_empty_and_drops_junk():
    assert parse_source_fetches(None) is None
    assert parse_source_fetches([]) == []
    assert parse_source_fetches("nope") == []
    parsed = parse_source_fetches([{"url": "https://a.org/x", "tool_name": "WebFetch", "content_chars": 3}, {"url": ""}])
    assert [p.url for p in parsed] == ["https://a.org/x"]


def test_only_the_unreported_gap_is_retried():
    assert not is_transient_failure(NO_READ_EVIDENCE)
    assert is_transient_failure(NO_READ_EVIDENCE_UNREPORTED)
