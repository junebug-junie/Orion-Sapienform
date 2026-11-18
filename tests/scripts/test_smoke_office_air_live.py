from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from urllib.error import URLError

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "smoke_office_air_live", ROOT / "scripts" / "smoke_office_air_live.py"
)
assert SPEC and SPEC.loader
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)


def config(**overrides):
    values = {
        "hub_base": "http://hub.test",
        "db_container": "db",
        "db_name": "orion",
        "db_user": "postgres",
        "max_age_sec": 120.0,
        "fallback_window_hours": 24,
        "allow_warming": False,
        "timeout_sec": 5.0,
    }
    values.update(overrides)
    return smoke.Config(**values)


def good_api():
    return {
        "ok": True,
        "age_sec": 12.5,
        "sample": {
            "sensor_online": True,
            "office_eco2_ppm": 615.0,
            "office_tvoc_ppb": 32.0,
            "air_warming": False,
        },
        "sensor_stale": False,
        "publisher_stale": False,
    }


def psql_responses(*, fallback="0", latest=None, columns=None):
    latest = latest or {
        "age_sec": 13.0,
        "eco2": 615,
        "tvoc": 32,
        "stale": False,
        "warming": False,
    }
    columns = columns or sorted(smoke.REQUIRED_COLUMNS)

    def fake_psql(_config, sql):
        if "information_schema.columns" in sql:
            return json.dumps(columns)
        if "ORDER BY ts DESC" in sql:
            return json.dumps(latest)
        if "bus_fallback_log" in sql:
            return fallback
        raise AssertionError(sql)

    return fake_psql


def run(**overrides):
    payload = overrides.pop("payload", good_api())
    psql = overrides.pop("psql", psql_responses())
    asset = overrides.pop("asset", "CCS811 officeEnvEco2 officeEnvTvoc")
    page = overrides.pop(
        "page", '<section id="officeEnvStrip"><div id="officeEnvEco2"></div>'
        '<div id="officeEnvTvoc"></div><script src="/static/js/office-env.js"></script>'
    )
    cfg = overrides.pop("config", config())
    assert not overrides

    def text_get(url, _timeout):
        return asset if url.endswith("/static/js/office-env.js") else page

    return smoke.run_smoke(
        cfg,
        json_get=lambda _url, _timeout: payload,
        text_get=text_get,
        psql=psql,
    )


def test_live_path_passes_without_printing_fallback_payloads():
    result = run()
    assert result["ok"] is True
    assert result["hub"]["eco2_ppm"] == 615.0
    assert result["database"]["tvoc_ppb"] == 32.0
    assert result["fallback_count"] == 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"ok": False}, "not current"),
        ({"age_sec": 121.0}, "exceeds"),
        ({"sample": {"sensor_online": False}}, "sensor_online=false"),
    ],
)
def test_hub_stale_or_offline_fails(change, message):
    payload = good_api()
    payload.update(change)
    with pytest.raises(smoke.SmokeFailure, match=message):
        run(payload=payload)


def test_missing_asset_marker_fails():
    with pytest.raises(smoke.SmokeFailure, match="officeEnvTvoc"):
        run(asset="CCS811 officeEnvEco2")


def test_rendered_page_must_link_asset_and_contain_office_dom():
    page = '<section id="officeEnvStrip"><div id="officeEnvEco2"></div></section>'
    with pytest.raises(smoke.SmokeFailure, match="officeEnvTvoc, office-env.js"):
        run(page=page)


def test_http_transport_failure_is_operator_readable(monkeypatch):
    def fail_urlopen(*_args, **_kwargs):
        raise URLError("connection refused")

    monkeypatch.setattr(smoke, "urlopen", fail_urlopen)
    with pytest.raises(smoke.SmokeFailure, match="connection refused"):
        smoke.fetch_text("http://hub.test/", 1.0)


def test_schema_drift_fails_before_claiming_success():
    psql = psql_responses(columns=["office_eco2_ppm", "office_tvoc_ppb"])
    with pytest.raises(smoke.SmokeFailure, match="air columns differ"):
        run(psql=psql)


def test_destination_row_must_be_fresh_and_not_stale():
    latest = {"age_sec": 5, "eco2": 600, "tvoc": 20, "stale": True, "warming": False}
    with pytest.raises(smoke.SmokeFailure, match="reports stale"):
        run(psql=psql_responses(latest=latest))


def test_fallback_debt_fails_without_reading_payloads():
    with pytest.raises(smoke.SmokeFailure, match="3 office.env.sample.v1 row"):
        run(psql=psql_responses(fallback="3"))


def test_warming_requires_explicit_operator_allowance():
    payload = good_api()
    payload["sample"] = {"sensor_online": True, "air_warming": True}
    latest = {"age_sec": 5, "eco2": None, "tvoc": None, "stale": False, "warming": True}
    with pytest.raises(smoke.SmokeFailure, match="still warming"):
        run(payload=payload, psql=psql_responses(latest=latest))

    result = run(
        payload=payload,
        psql=psql_responses(latest=latest),
        config=config(allow_warming=True),
    )
    assert result["hub"]["warming"] is True
    assert result["database"]["warming"] is True


def test_cli_rejects_unbounded_fallback_windows():
    with pytest.raises(SystemExit):
        smoke.parse_args(["--fallback-window-hours", "0"])
    with pytest.raises(SystemExit):
        smoke.parse_args(["--fallback-window-hours", "169"])
