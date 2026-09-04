from __future__ import annotations

from pathlib import Path

from app.crawl.ksl_adapter import (
    parse_category_page,
    parse_detail_description,
    parse_posted_date,
    parse_price,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _fixture_html() -> str:
    return (FIXTURES_DIR / "ksl_category_sample.html").read_text(encoding="utf-8")


def test_fixture_file_exists():
    assert (FIXTURES_DIR / "ksl_category_sample.html").is_file()


def test_parse_category_page_finds_all_real_cards():
    candidates = parse_category_page(_fixture_html(), category_url="https://classifieds.ksl.com/search/cat/Electronics")
    # 4 Electronics + 3 Computers + 2 FREE real cards, saved 2026-09-04.
    assert len(candidates) == 9


def test_parse_category_page_extracts_real_listing_id_title_price():
    candidates = parse_category_page(_fixture_html(), category_url="https://classifieds.ksl.com/search/cat/Electronics")
    by_id = {c.external_listing_id: c for c in candidates}
    gx1400 = by_id["77981153"]
    assert gx1400.title == "GX1400 TRANSCEIVER"
    assert gx1400.price == 100.0
    assert gx1400.url == "https://classifieds.ksl.com/listing/77981153"


def test_parse_category_page_handles_comma_formatted_price():
    candidates = parse_category_page(_fixture_html(), category_url="https://classifieds.ksl.com/search/cat/Electronics")
    by_id = {c.external_listing_id: c for c in candidates}
    pinball = by_id["81261397"]
    assert pinball.price == 17500.0


def test_parse_category_page_handles_free_listings():
    candidates = parse_category_page(_fixture_html(), category_url="https://classifieds.ksl.com/search/cat/FREE")
    free_candidates = [c for c in candidates if c.price_raw.upper() == "FREE"]
    assert len(free_candidates) >= 2
    assert all(c.price == 0.0 for c in free_candidates)


def test_parse_price_variants():
    assert parse_price("$100.00") == 100.0
    assert parse_price("$17,500.00") == 17500.0
    assert parse_price("FREE") == 0.0
    assert parse_price("Call for quote") is None


def test_parse_detail_description_extracts_seller_text():
    html = (
        '<div><h2 class="text-lg font-semibold undefined">Description</h2>'
        '<p class="mb-4 last-of-type:mb-0">STANDARD HORIZON 25 WATT VHF FM MARINE TRANSCEIVER. BLACK</p>'
        '<p class="mb-4 last-of-type:mb-0">NEW</p></div>'
    )
    description = parse_detail_description(html)
    assert description == "STANDARD HORIZON 25 WATT VHF FM MARINE TRANSCEIVER. BLACK\nNEW"


def test_parse_detail_description_none_when_absent():
    assert parse_detail_description("<div>no description tab here</div>") is None


def test_parse_posted_date_parses_real_format():
    html = '<span aria-label="Posted Aug 30, 2026">Aug 30, 2026</span>'
    posted = parse_posted_date(html)
    assert posted is not None
    assert (posted.year, posted.month, posted.day) == (2026, 8, 30)


def test_parse_posted_date_none_when_absent():
    assert parse_posted_date("<div>no posted stat</div>") is None
