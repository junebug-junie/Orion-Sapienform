from __future__ import annotations

from app.crawl.dedup import content_hash, external_listing_id_from_url, normalize_title


def test_external_listing_id_parses_canonical_url():
    assert external_listing_id_from_url("https://classifieds.ksl.com/listing/77981153") == "77981153"


def test_external_listing_id_parses_url_with_query_string():
    assert external_listing_id_from_url("https://classifieds.ksl.com/listing/77981153?ref=search") == "77981153"


def test_external_listing_id_none_for_non_listing_url():
    assert external_listing_id_from_url("https://classifieds.ksl.com/search/cat/Electronics") is None


def test_normalize_title_lowercases_and_collapses_whitespace():
    assert normalize_title("  GX1400   TRANSCEIVER \n") == "gx1400 transceiver"


def test_normalize_title_matches_sql_side_normalization_shape():
    # Must stay in lockstep with repository.list_current_by_normalized's
    # `lower(regexp_replace(title, '\s+', ' ', 'g'))` -- this is the Python
    # mirror of that exact transform.
    assert normalize_title("Iphone 12   Pro Max") == "iphone 12 pro max"


def test_content_hash_stable_for_same_inputs():
    a = content_hash(external_listing_id="1", title="X", price_raw="$1.00", description="d")
    b = content_hash(external_listing_id="1", title="X", price_raw="$1.00", description="d")
    assert a == b


def test_content_hash_changes_when_title_changes():
    a = content_hash(external_listing_id="1", title="X", price_raw="$1.00", description="d")
    b = content_hash(external_listing_id="1", title="Y", price_raw="$1.00", description="d")
    assert a != b


def test_content_hash_handles_none_fields():
    # Must not raise on a candidate with no price_raw/description yet.
    h = content_hash(external_listing_id="1", title="X", price_raw=None, description=None)
    assert isinstance(h, str) and len(h) == 64
