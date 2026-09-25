import pytest

from orion.world_pulse_read.url_filters import url_looks_like_section_index


def test_section_index_urls():
    assert url_looks_like_section_index("https://www.tomshardware.com/news")
    assert url_looks_like_section_index("https://www.tomshardware.com/news/")
    assert url_looks_like_section_index("https://example.com/blog")
    assert url_looks_like_section_index("https://example.com/category/ai")


def test_article_urls_are_not_indexes():
    assert not url_looks_like_section_index(
        "https://www.tomshardware.com/news/nvidia-rtx-5090-msrp-return"
    )
    assert not url_looks_like_section_index(
        "https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer"
    )
    assert not url_looks_like_section_index(
        "https://www.youtube.com/watch?v=-E_dQSaUccg&vl=en"
    )


# Live world_pulse_read_seed URLs, pulled read-only 2026-09-25. Each one is a
# case the pre-2026-09-25 filter got wrong.

LIVE_LISTINGS_MISSED = [
    # Roundup hub under /article/ -- read "done" 3 times, the last one a
    # metadata-only hollow success (finding:60d59b10...:9b084fc0f1583da0).
    "https://www.networkworld.com/article/3562856/nvidia-latest-news-and-insights.html",
    "https://wccftech.com/topic/hardware/",
    "https://www.silicondata.com/media",
    "https://www.usgs.gov/news/national-news-release",
    "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
]

LIVE_ARTICLES_WRONGLY_SKIPPED = [
    # 36 distinct BBC article URLs (5 skipped as section_index_url, 31 pending).
    "https://www.bbc.co.uk/news/articles/c17jqp0xzpzo?at_medium=RSS&at_campaign=rss",
    "https://www.bbc.co.uk/news/articles/cvgy5k4n07ko?at_medium=RSS&at_campaign=rss",
    "https://www.bbc.co.uk/news/videos/czxzwge59w8o?at_medium=RSS&at_campaign=rss",
]

LIVE_ARTICLES_STILL_ARTICLES = [
    "https://www.who.int/news/item/30-07-2025-paying-tribute-to-david-nabarro",
    "https://www.nasa.gov/news-release/nasa-selects-far-infrared-telescope-as-first-in-new-mission-class/",
    "https://www.npr.org/2026/09/23/nx-s1-5978055/congress-ai-regulation",
    "https://governor.utah.gov/press/nominees-announced-for-third-district-court-vacancy-2/",
    "https://huggingface.co/blog/tokenizers-v1",
    "https://arxiv.org/abs/2310.19279",
    "https://www.mdpi.com/1424-8220/24/15/4830",
    "https://www.apollo.com/institutional/insights-news/insights/2026/06/growing-compute-shortage",
    "https://www.reddit.com/r/hardware/comments/1pt4u0p/gamers_nexus_nvidia_wtf_combined_singlevideo/",
]

LIVE_LISTINGS_STILL_LISTINGS = [
    "https://hothardware.com/",
    "https://spectrum.ieee.org/tag/gpus",
    "https://www.tomshardware.com/pc-components/gpus/news",
]


@pytest.mark.parametrize("url", LIVE_LISTINGS_MISSED + LIVE_LISTINGS_STILL_LISTINGS)
def test_live_listing_roundup_and_feed_urls_are_skipped(url):
    assert url_looks_like_section_index(url)


@pytest.mark.parametrize("url", LIVE_ARTICLES_WRONGLY_SKIPPED + LIVE_ARTICLES_STILL_ARTICLES)
def test_live_article_urls_are_read(url):
    assert not url_looks_like_section_index(url)


def test_item_parent_does_not_rescue_a_roundup_slug():
    # /article/ is an item parent, but the roundup rule runs first.
    assert url_looks_like_section_index(
        "https://example.com/article/1/amd-latest-updates"
    )
