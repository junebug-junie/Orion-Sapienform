from orion.world_pulse_read.seeds import make_seed_id, seeds_from_digest_payload


def test_make_seed_id_is_stable_for_same_inputs():
    a = make_seed_id(kind="finding", run_id="r1", url="https://ex.com/a")
    b = make_seed_id(kind="finding", run_id="r1", url="https://ex.com/a")
    assert a == b
    assert a.startswith("finding:r1:")


def test_seeds_findings_before_digest_items():
    payload = {
        "run_id": "r1",
        "curiosity_followups": [
            {
                "section": "ai_technology",
                "articles": [
                    {"url": "https://ex.com/f1", "title": "F1", "description": "", "salience": 0.9}
                ],
            }
        ],
        "items": [
            {
                "item_id": "item-1",
                "run_id": "r1",
                "title": "Digest card",
                "category": "hardware_compute_gpu",
                "worth_reading": ["https://ex.com/d1"],
                "article_ids": [],
            }
        ],
    }
    seeds = seeds_from_digest_payload(payload)
    assert [s.kind for s in seeds] == ["finding", "digest_item"]
    assert seeds[0].url == "https://ex.com/f1"
    assert seeds[1].url == "https://ex.com/d1"
    assert seeds[1].item_id == "item-1"


def test_digest_item_resolves_url_via_article_map_when_worth_reading_empty():
    payload = {
        "run_id": "r1",
        "curiosity_followups": [],
        "items": [
            {
                "item_id": "item-2",
                "run_id": "r1",
                "title": "No worth_reading",
                "category": "ai_technology",
                "worth_reading": [],
                "article_ids": ["art-9"],
            }
        ],
    }
    seeds = seeds_from_digest_payload(
        payload, article_urls={"art-9": "https://ex.com/from-article"}
    )
    assert len(seeds) == 1
    assert seeds[0].url == "https://ex.com/from-article"


def test_digest_item_without_url_is_skipped():
    payload = {
        "run_id": "r1",
        "curiosity_followups": [],
        "items": [
            {
                "item_id": "item-3",
                "run_id": "r1",
                "title": "Orphan",
                "category": "ai_technology",
                "worth_reading": [],
                "article_ids": [],
            }
        ],
    }
    assert seeds_from_digest_payload(payload) == []
