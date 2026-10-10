from app.settings import Settings


def test_contractor_peer_defaults_on() -> None:
    # Code default matches .env_example and production (aligned 2026-10-10;
    # scripts/check_settings_defaults.py --example-drift gates the match).
    s = Settings()
    assert s.HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED is True
