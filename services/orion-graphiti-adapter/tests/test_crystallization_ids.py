import pytest

from app.crystallization_ids import validate_crystallization_id


def test_validate_accepts_crys_prefix():
    assert validate_crystallization_id("crys_test001") == "crys_test001"


def test_validate_accepts_raw_uuid():
    # Regression: real memory_crystallizations.crystallization_id values are bare UUIDs
    # (no crys_ prefix -- that prefix is only used for derived ids like chroma doc_id /
    # graphiti entity_id). The graphiti_core backend's ingest_episode() previously rejected
    # every real crystallization_id with invalid_crystallization_id, making /v1/episodes and
    # /v1/rebuild 500 for all live data (caught only by testing against real Postgres rows).
    cid = "191fdf0e-6e86-42e7-a661-86a517d491e1"
    assert validate_crystallization_id(cid) == cid


def test_validate_rejects_injection():
    with pytest.raises(ValueError, match="invalid_crystallization_id"):
        validate_crystallization_id("crys_x'} ) DELETE n //")
