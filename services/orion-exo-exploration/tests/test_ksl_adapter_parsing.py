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


def test_price_window_does_not_bleed_into_next_card():
    """Review finding, 2026-09-04: the price search window used to be a flat
    3000-char lookahead with a comment claiming that was "tight enough to
    never bleed into the next card" -- real captured cards are only
    2206-2751 chars apart, well under that constant. A card missing its own
    Price aria-label (this crawl has never observed one, but nothing
    guarantees it) would silently pick up the NEXT card's price instead of
    correctly reporting none. Built from two real cards' real spacing
    (~2300 chars of real filler between them), first card's price div
    deliberately stripped out.
    """
    real_html = _fixture_html()
    first = 'aria-label="No Price Card" data-item-id="11111111" href="https://classifieds.ksl.com/listing/11111111"'
    # ~2300 chars of real filler (borrowed from the real fixture) between
    # the two cards, matching the real observed gap -- NOT the price div
    # itself, which is deliberately omitted from the first card.
    filler = real_html[500:2800].replace('aria-label="Price', 'data-not-a-price-label="Price')
    second = (
        'aria-label="Second Card" data-item-id="22222222" '
        'href="https://classifieds.ksl.com/listing/22222222"'
        '</div></div><div aria-label="Price $250.00">$250.00</div>'
    )
    html = f"<a class=\"group\" {first}>{filler}<a class=\"group\" {second}"
    candidates = parse_category_page(html, category_url="https://classifieds.ksl.com/search/cat/Electronics")
    by_id = {c.external_listing_id: c for c in candidates}
    assert by_id["11111111"].price_raw == ""
    assert by_id["11111111"].price is None
    assert by_id["22222222"].price == 250.0


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
