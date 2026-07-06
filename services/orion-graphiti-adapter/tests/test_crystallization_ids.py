import pytest

from app.crystallization_ids import validate_crystallization_id


def test_validate_accepts_crys_prefix():
    assert validate_crystallization_id("crys_test001") == "crys_test001"


def test_validate_rejects_injection():
    with pytest.raises(ValueError, match="invalid_crystallization_id"):
        validate_crystallization_id("crys_x'} ) DELETE n //")
