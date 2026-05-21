"""Gateway must subscribe to cognition trace channel (spec §5.4 preflight)."""
from app.settings import settings


def test_organ_channels_includes_cognition_trace() -> None:
    patterns = settings.ORGAN_CHANNELS
    assert any(
        p in ("orion:cognition:trace", "orion:cognition:*") or p.endswith("cognition:*")
        for p in patterns
    ), f"ORGAN_CHANNELS missing cognition pattern: {patterns}"
