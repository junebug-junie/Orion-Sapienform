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
