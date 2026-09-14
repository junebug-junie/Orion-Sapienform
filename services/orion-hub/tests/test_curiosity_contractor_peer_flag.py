from app.settings import Settings


def test_contractor_peer_defaults_off() -> None:
    s = Settings()
    assert s.HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED is False
