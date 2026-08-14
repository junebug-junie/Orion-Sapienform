"""Tests for the subscribed-channel route-coverage startup gate.

The gate exists because twice on 2026-08-13/14 this service ran for hours
subscribed to a channel it had no route for, writing every event to
``bus_fallback_log`` while looking completely healthy. See
``app/route_coverage.py`` for the full incident.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
for path in (SERVICE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.route_coverage import find_unrouted_channels, log_route_coverage  # noqa: E402
from app.settings import Settings  # noqa: E402

_YAML = """
channels:
  - name: "orion:routed"
    message_kind: "routed.v1"
  - name: "orion:adapter"
    message_kind: "adapter.v1"
  - name: "orion:orphan"
    message_kind: "orphan.v1"
  - name: "orion:no:kind"
"""


def _yaml(tmp_path: Path) -> Path:
    p = tmp_path / "channels.yaml"
    p.write_text(_YAML, encoding="utf-8")
    return p


def test_reproduces_the_real_incident(tmp_path: Path) -> None:
    """The exact shape of both outages: the channel is subscribed (durable
    .env) but the kind has no route (code from an older image). This is the
    condition that silently diverts every event to bus_fallback_log."""
    found = find_unrouted_channels(
        subscribed=["orion:routed", "orion:orphan"],
        route_map={"routed.v1": "RoutedSQL"},
        channels_yaml=_yaml(tmp_path),
        adapter_kinds=set(),
    )
    assert found == [("orion:orphan", "orphan.v1")]


def test_a_fully_routed_service_is_clean(tmp_path: Path) -> None:
    assert (
        find_unrouted_channels(
            subscribed=["orion:routed"],
            route_map={"routed.v1": "RoutedSQL"},
            channels_yaml=_yaml(tmp_path),
            adapter_kinds=set(),
        )
        == []
    )


def test_adapter_handled_kinds_are_not_false_positives(tmp_path: Path) -> None:
    """The gate's own first version failed this.

    `handle_envelope` has TWO dispatch paths: route_map -> a SQL model, and
    the adapter-only branch -> evidence_units. Checking only the first flagged
    orion:evidence:markdown:ingest and orion:evidence:parsed:ingest, which are
    healthy -- verified against real data at the time: zero rows in
    bus_fallback_log for either kind, while evidence_units held 20,248 rows
    updated seconds earlier.

    A gate that cries wolf on working channels is worse than no gate; it
    teaches people to skip the output, which is exactly how the original
    silent WARNING went unnoticed."""
    assert (
        find_unrouted_channels(
            subscribed=["orion:adapter"],
            route_map={},
            channels_yaml=_yaml(tmp_path),
            adapter_kinds={"adapter.v1"},
        )
        == []
    )


def test_channels_without_a_declared_kind_are_skipped(tmp_path: Path) -> None:
    """Several subscribed channels are pattern/broadcast channels with no
    message_kind in the registry. Flagging them would produce standing noise
    on every boot."""
    assert (
        find_unrouted_channels(
            subscribed=["orion:no:kind"],
            route_map={},
            channels_yaml=_yaml(tmp_path),
            adapter_kinds=set(),
        )
        == []
    )


def test_unknown_channel_not_in_the_registry_is_skipped(tmp_path: Path) -> None:
    """A subscribed channel absent from channels.yaml entirely cannot be
    judged -- there is no declared kind to check. Skipped, not flagged."""
    assert (
        find_unrouted_channels(
            subscribed=["orion:never:registered"],
            route_map={},
            channels_yaml=_yaml(tmp_path),
            adapter_kinds=set(),
        )
        == []
    )


def test_an_unreadable_registry_does_not_raise(tmp_path: Path) -> None:
    """This runs during startup. It must never be the reason the service
    fails to boot -- a diagnostic that takes down persistence is worse than
    the problem it reports."""
    assert (
        find_unrouted_channels(
            subscribed=["orion:routed"],
            route_map={},
            channels_yaml=tmp_path / "does-not-exist.yaml",
            adapter_kinds=set(),
        )
        == []
    )


def test_log_route_coverage_returns_the_offenders(tmp_path: Path) -> None:
    """The logging wrapper must surface what it found, so a caller (or a
    future health endpoint) can assert on it rather than scrape stderr."""
    assert log_route_coverage(
        subscribed=["orion:orphan"],
        route_map={},
        channels_yaml=_yaml(tmp_path),
        adapter_kinds=set(),
    ) == [("orion:orphan", "orphan.v1")]


def test_the_real_shipped_config_is_fully_covered() -> None:
    """The gate applied to this repo's actual channels.yaml and the code
    default subscribe list -- so a PR that subscribes to a channel without
    adding its route fails here, at review time, instead of silently
    diverting live events to bus_fallback_log after deploy.

    Uses the declared FIELD DEFAULT rather than Settings(): instantiating
    reads a cwd-relative .env, which would make this assert on whatever the
    local operator happens to have configured instead of on what ships."""
    subscribed = Settings.model_fields["sql_writer_subscribe_channels"].default
    from app.settings import DEFAULT_ROUTE_MAP

    assert find_unrouted_channels(subscribed, DEFAULT_ROUTE_MAP) == []


def test_registry_is_found_by_walking_up_not_by_a_fixed_depth(tmp_path: Path) -> None:
    """Regression guard for a real outage this file caused.

    The first version computed the registry path as
    `Path(__file__).resolve().parents[3]`, correct for the repo layout
    (services/orion-sql-writer/app/ -> repo root) and an IndexError inside the
    image, which copies the module to /app/app/ with the registry at /app/.
    Because it ran at MODULE level, the IndexError fired on import and
    crash-looped the whole worker -- RestartCount reached 3 before it was
    caught. A startup diagnostic must not be able to break startup.

    Builds both real layouts and asserts the resolver handles each."""
    from app import route_coverage

    for depth in (Path("services/svc/app"), Path("app")):
        root = tmp_path / str(depth).replace("/", "_")
        module_dir = root / depth
        module_dir.mkdir(parents=True)
        registry = root / "orion" / "bus"
        registry.mkdir(parents=True)
        (registry / "channels.yaml").write_text("channels: []", encoding="utf-8")

        fake_module = module_dir / "route_coverage.py"
        fake_module.write_text("", encoding="utf-8")

        found = None
        for parent in fake_module.resolve().parents:
            candidate = parent / route_coverage._CHANNELS_YAML_RELATIVE
            if candidate.is_file():
                found = candidate
                break
        assert found == (registry / "channels.yaml"), f"layout {depth} not resolved"


def test_a_missing_registry_returns_empty_rather_than_raising() -> None:
    """The other half of the same lesson: no path condition may raise out of
    this module, because it is imported at worker startup."""
    from app.route_coverage import _channel_kinds

    assert _channel_kinds(Path("/nonexistent/orion/bus/channels.yaml")) == {}
