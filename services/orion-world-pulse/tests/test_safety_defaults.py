from __future__ import annotations

from app.settings import Settings


def test_world_pulse_defaults_are_conservative() -> None:
    cfg = Settings()
    assert cfg.world_pulse_enabled is False
    assert cfg.world_pulse_dry_run is True
    assert cfg.world_pulse_graph_enabled is False
    assert cfg.world_pulse_graph_dry_run is True
    assert cfg.world_pulse_stance_enabled is False



def test_direct_email_path_is_retired() -> None:
    """Retired 2026-09-30: World Pulse has no direct email route or email settings.

    The news digest reaches email only via orion-actions' Journal Pass
    (trigger_kind=world_pulse_digest). A resurrected route here would be a
    second, unreviewed email path.
    """
    from app.routers.publish import router

    paths = {getattr(route, "path", "") for route in router.routes}
    assert "/api/world-pulse/runs/{run_id}/publish-hub-message" in paths
    assert not any("email" in path for path in paths)
    assert not any("email" in name for name in Settings.model_fields)
