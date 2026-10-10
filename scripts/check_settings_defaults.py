#!/usr/bin/env python3
"""Verify a service's pydantic-settings `Settings` class gives every field a real
default -- unless the field is on that service's explicit `REQUIRED_NO_DEFAULT`
allowlist below.

Context: `services/orion-actions/docker-compose.yml` used to hand-duplicate almost
every `.env_example` key in a ~90-line `environment:` list, each entry usually
carrying its own `${KEY:-default}` compose-level fallback -- a second, competing
place a default could silently drift from `app/settings.py`'s `Field(...)` default.
The fix for orion-actions was to delete that redundant `environment:` list and rely
on `env_file: [.env]` (already present) to pass every key through untouched -- see
`scripts/check_service_env_compose_parity.py`, which already treats a compose file
with `env_file:` as N/A for key-by-key parity. That only holds up if every Settings
field has a real default: a required field is normally protected today by
docker-compose's own duplicate-with-fallback line, so removing that line requires
the underlying pydantic-settings model to be able to boot with the field genuinely
unset (bare tests, CI, a fresh host with an incomplete `.env`). This script makes
"every field has a default, except an explicit named allowlist for true secrets"
a gated, testable claim instead of an implicit assumption.

Pilot scope: `services/orion-actions` only (2026-07 env/settings single-source-of-
truth pilot). Non-goal: generalizing to the other ~84 services -- see
`REQUIRED_NO_DEFAULT` and `_SETTINGS_MODULE_PATHS` below, both scoped by an explicit
per-service entry, not a generic services/*/app/settings.py glob.

Usage:
    python scripts/check_settings_defaults.py orion-actions
    python scripts/check_settings_defaults.py orion-actions --json
    python scripts/check_settings_defaults.py orion-actions --report-only
    python scripts/check_settings_defaults.py orion-hub --example-drift
        (separate check: every Settings default and docker-compose ${KEY:-x}
        fallback must equal the service's .env_example -- see
        _EXAMPLE_DRIFT_SETTINGS_PATHS below; gated in orion-static-gates.yml)

Exit codes: 0 = every Settings field has a real default or is on the service's
                 REQUIRED_NO_DEFAULT allowlist.
            1 = at least one field has no default and isn't allowlisted (FAIL) --
                unless --report-only.
            2 = could not run the check (unknown service, missing settings module,
                import error, no `Settings` class found).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Running as `python scripts/check_settings_defaults.py` puts scripts/ on
# sys.path[0], which shadows stdlib modules (same issue documented in
# scripts/check_inner_state_registry.py / scripts/check_service_env_compose_parity.py).
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Explicit, per-service opt-in only -- this pilot deliberately does not glob
# services/*/app/settings.py. Add an entry here (and to REQUIRED_NO_DEFAULT below)
# only after auditing that service's Settings class the same way orion-actions was
# audited (see the PR that introduced this file).
_SETTINGS_MODULE_PATHS: dict[str, str] = {
    "orion-actions": "app/settings.py",
}

# Fields that are genuinely required secrets/config with no safe default, keyed by
# service then by the Settings attribute name (the model_fields key, not the env
# alias). Empty per service is the expected common case -- see this repo's
# no-keyword-cathedral rule: an allowlist entry must correspond to a real field
# that actually has no default, not a defensive placeholder.
REQUIRED_NO_DEFAULT: dict[str, frozenset[str]] = {
    "orion-actions": frozenset(),
}


# --- --example-drift mode -------------------------------------------------
#
# A second, separate claim about the same Settings class: every field's code
# default must equal the value the service's checked-in `.env_example` sets for
# it, and so must any `${KEY:-fallback}` in the service's docker-compose.yml.
# Otherwise a key that goes missing from a live `.env` silently changes
# behaviour. Found 2026-10-09: orion-hub's curiosity daily cap defaulted to 3 in
# code while `.env_example` and production ran 7; the cooldown was 14400s vs
# 1800s live; orion-gpu-pool's shed lever was False in code and in compose's
# fallback while `.env_example` and production ran true. 2026-10-10 sweep found
# 73 drifted Settings fields in orion-hub alone.
#
# Scoped per service, like the pilot above. Add a service only after aligning
# it (or listing its remaining drift below with a reason).
_EXAMPLE_DRIFT_SETTINGS_PATHS: dict[str, str] = {
    "orion-hub": "app/settings.py",
    "orion-gpu-pool": "app/settings.py",
}

# Keys whose `.env_example` value describes THIS host's topology (loopback /
# tailnet URLs, filesystem paths, DSNs, geographic location, node project name).
# Their code/compose default deliberately stays portable (docker DNS name or
# empty = feature unconfigured) instead of baking one machine's address into
# code. An entry here that is no longer drifted on any surface fails the check,
# so the list cannot rot into a blanket exemption.
HOST_SPECIFIC_EXAMPLE_KEYS: dict[str, frozenset[str]] = {
    "orion-hub": frozenset({
        "PROJECT",
        "TOPIC_FOUNDRY_BASE_URL",
        "WORLD_PULSE_BASE_URL",
        "FIELD_DIGESTER_BASE_URL",
        "HUB_EXO_EXPLORATION_BASE_URL",
        "HUB_PROPOSAL_REVIEW_API_URL",
        "HUB_CONTEXT_EXEC_API_URL",
        "HUB_FCC_ENV_PATH",
        "HUB_AITOWN_UI_URL",
        "HUB_AITOWN_CONVEX_URL",
        "SOCIAL_MEMORY_BASE_URL",
        "SELF_EXPERIMENTS_BASE_URL",
        "JUNIPER_AFFECTIVE_STATE_BASE_URL",
        "PERCEPT_STORE_BASE_URL",
        "CABINET_SENSORS_B_PATH",
        "CABINET_BOOT_B_PATH",
        "NOTIFY_BASE_URL",
        "HUB_READING_SEARCH_CHROMA_URL",
        "HUB_READING_SEARCH_EMBED_URL",
        "HUB_RECALL_SERVICE_URL",
        "RECALL_SERVICE_URL",
        "RECALL_PG_DSN",
        "FALKORDB_URI",
        "CRYSTALLIZER_EMBED_HOST_URL",
        "GRAPHITI_ADAPTER_URL",
        "ORION_SITUATION_LOCATION_LABEL",
        "ORION_SITUATION_LOCALITY",
        "ORION_SITUATION_REGION",
        "ORION_SITUATION_COUNTRY",
        "ORION_SITUATION_HOME_LOCATION",
        "ORION_SITUATION_PHYSICAL_LOCATION",
        "ORION_SITUATION_WEATHER_LAT",
        "ORION_SITUATION_WEATHER_LON",
        "RDF_STORE_BASE_URL",
        "RDF_STORE_QUERY_URL",
        "RDF_STORE_PASS",
        "MEMORY_GRAPH_DEFAULT_NAMED_GRAPH",
        "FIELD_PLASTICITY_SQL_DB_PATH",
        "AUTONOMY_GRAPH_QUERY_URL",
        "AUTONOMY_GRAPH_UPDATE_URL",
    }),
    "orion-gpu-pool": frozenset(),
}

# Behavioural keys still drifted on purpose, each with its reason: either
# waiting on a human decision, or behaviourally equivalent by construction.
# Same staleness rule as above: once aligned, the entry must be removed.
REASONED_DRIFT: dict[str, dict[str, str]] = {
    "orion-hub": {
        "HUB_PROPOSAL_REVIEW_ENABLED": (
            "on in .env_example and live, but its API (orion-context-exec :8096) "
            "is not deployed -- nothing listens on 8096 (2026-10-10). Live value "
            "looks like a mistake; code default left False pending Juniper."
        ),
        "CHAT_HISTORY_LOG_CHANNEL": (
            "equivalent: None is an override slot that falls back to "
            "CHANNEL_CHAT_HISTORY_LOG (default 'orion:chat:history:log', same "
            "as .env_example); a non-None default would disable that fallback."
        ),
    },
    "orion-gpu-pool": {
        "GPU_POOL_ORION_SHED_ENABLED": (
            "Orion's own self-shed. On in .env_example and live since 2026-10-01, but "
            ".env_example records that as Juniper's call with 'code default stays off'. "
            "A recorded decision, not drift to auto-fix; left for Juniper."
        ),
    },
}


class SettingsCheckError(ValueError):
    """Raised when the target service's Settings class can't be loaded/introspected."""


def _load_settings_module(service: str, paths: dict[str, str] | None = None) -> ModuleType:
    rel_path = (_SETTINGS_MODULE_PATHS if paths is None else paths).get(service)
    if rel_path is None:
        raise SettingsCheckError(
            f"'{service}' is not in check_settings_defaults.py's _SETTINGS_MODULE_PATHS "
            f"allowlist. This checker is scoped to explicitly audited services only "
            f"(pilot: orion-actions) -- add an entry there first."
        )

    settings_path = _REPO_ROOT / "services" / service / rel_path
    if not settings_path.is_file():
        raise SettingsCheckError(f"settings module not found at {settings_path}")

    module_name = f"_check_settings_defaults__{service.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, settings_path)
    if spec is None or spec.loader is None:
        raise SettingsCheckError(f"could not build an import spec for {settings_path}")

    module = importlib.util.module_from_spec(spec)
    # Registered before exec so pydantic can resolve `from __future__ import
    # annotations` string annotations (e.g. Literal[...]) against the module --
    # without it TypeAdapter(field.annotation) cannot rebuild the type.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - see fallback below before treating this as fatal
        # A settings module with a trailing top-level singleton (e.g. orion-actions'
        # `settings = get_settings()`) executes its `class Settings(...)` body FIRST,
        # then instantiates it as its last statement. If a field has no default and
        # the real env doesn't happen to supply a value in whatever process is
        # running this checker, that instantiation itself raises a pydantic
        # ValidationError -- exactly the condition this checker exists to catch.
        # `module.__dict__` is populated incrementally as exec_module runs, so
        # `Settings` is already fully defined and usable even though the exception
        # aborted the module before reaching its end. Prefer using that
        # already-defined class (so the field-by-field diagnostic below still
        # fires with exit 1 and real field names) over surfacing a raw traceback
        # as an exit-2 "couldn't run the check" -- confirmed live: without this,
        # introducing a genuinely-required field crashed the import before
        # `_find_missing_defaults` ever reached its `model_fields` walk.
        if not hasattr(module, "Settings"):
            raise SettingsCheckError(f"importing {settings_path} raised {exc!r}") from exc

    return module


def _find_missing_defaults(service: str) -> list[str]:
    """Returns the sorted list of Settings field names (attribute names, not env
    aliases) that have no default and aren't on the service's allowlist."""
    module = _load_settings_module(service)

    settings_cls = getattr(module, "Settings", None)
    if settings_cls is None:
        raise SettingsCheckError(
            f"{service}'s settings module has no top-level `Settings` class"
        )

    model_fields = getattr(settings_cls, "model_fields", None)
    if model_fields is None:
        raise SettingsCheckError(
            f"{service}'s Settings class has no `model_fields` -- is this really a "
            f"pydantic v2 BaseSettings subclass?"
        )

    allowlist = REQUIRED_NO_DEFAULT.get(service, frozenset())
    missing: list[str] = []
    for field_name, field_info in model_fields.items():
        is_required = getattr(field_info, "is_required", None)
        if not callable(is_required):
            # pydantic v2's FieldInfo.is_required() has existed since 2.0; if it's
            # missing, this isn't the pydantic v2 model this checker was built for.
            # NOTE: an earlier version of this fallback compared `field_info.default
            # is ...` (Ellipsis) -- wrong for pydantic v2, where an unset default is
            # the sentinel `PydanticUndefined`, not Ellipsis, so that comparison was
            # always False and would have silently reported every required field as
            # having a default. Fail loudly instead of guessing.
            raise SettingsCheckError(
                f"{service}'s Settings.{field_name} FieldInfo has no is_required() "
                f"method -- expected pydantic v2's BaseModel/BaseSettings FieldInfo."
            )
        if is_required() and field_name not in allowlist:
            missing.append(field_name)

    return sorted(missing)


_ENV_KEY_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_COMPOSE_FALLBACK = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-((?:[^{}$]|\$(?!\{))*)\}")


def _unquote(raw: str) -> str:
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    # Unquoted values: compose and python-dotenv both drop a ` #` inline comment.
    hash_at = s.find(" #")
    return s[:hash_at].rstrip() if hash_at >= 0 else s


def _parse_env_example(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export "):].lstrip()
        m = _ENV_KEY_LINE.match(stripped)
        if m:
            out[m.group(1)] = _unquote(m.group(2))  # last assignment wins, like dotenv
    return out


def _compose_fallbacks(path: Path) -> dict[str, str]:
    """`${KEY:-fallback}` literals in a compose file. Nested `${A:-${B}}`
    fallbacks are skipped -- their effective value is another variable."""
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for m in _COMPOSE_FALLBACK.finditer(path.read_text(encoding="utf-8")):
        out[m.group(1)] = _unquote(m.group(2))
    return out


def _field_env_name(field_name: str, field_info) -> str:
    alias = field_info.validation_alias or field_info.alias
    return alias if isinstance(alias, str) else field_name


def _coerce(adapter, raw: str):
    """Parse an env string the way pydantic-settings would for this field type."""
    try:
        return adapter.validate_python(raw)
    except Exception:  # noqa: BLE001 - complex types arrive as JSON text
        return adapter.validate_json(raw)


def _same(left, right) -> bool:
    if left == right:
        return True
    # An empty env string and a None default both mean "unset" to every reader
    # in these services; do not report that as drift.
    return {left, right} <= {None, ""} if not isinstance(left, (list, dict)) and not isinstance(right, (list, dict)) else False


def _same_text(left: str, right: str) -> bool:
    a, b = left.strip(), right.strip()
    if a.lower() == b.lower() and a.lower() in {"true", "false"}:
        return True
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def find_example_drift(service: str) -> dict:
    """Compare Settings defaults and compose fallbacks against `.env_example`.

    Returns {"drift": [...], "stale_exemptions": [...], "reasoned": [...],
    "host_specific": [...]}; each drift item is a dict with key/surface/default/
    example. Only `drift` and `stale_exemptions` fail the check.
    """
    from pydantic import TypeAdapter

    if service not in _EXAMPLE_DRIFT_SETTINGS_PATHS:
        raise SettingsCheckError(
            f"'{service}' is not in _EXAMPLE_DRIFT_SETTINGS_PATHS -- align it (or list "
            f"its remaining drift with a reason) and add an entry first."
        )
    service_dir = _REPO_ROOT / "services" / service
    example_path = service_dir / ".env_example"
    if not example_path.is_file():
        raise SettingsCheckError(f".env_example not found at {example_path}")
    example = _parse_env_example(example_path)

    module = _load_settings_module(service, _EXAMPLE_DRIFT_SETTINGS_PATHS)
    settings_cls = getattr(module, "Settings", None)
    if settings_cls is None or getattr(settings_cls, "model_fields", None) is None:
        raise SettingsCheckError(f"{service}'s settings module has no pydantic `Settings` class")

    host_specific = HOST_SPECIFIC_EXAMPLE_KEYS.get(service, frozenset())
    reasoned = REASONED_DRIFT.get(service, {})
    exempt = set(host_specific) | set(reasoned)

    raw_drift: list[dict] = []
    adapters: dict[str, object] = {}
    for field_name, field_info in settings_cls.model_fields.items():
        env_name = _field_env_name(field_name, field_info)
        adapter = TypeAdapter(field_info.annotation)
        adapters[env_name] = adapter
        if env_name not in example or field_info.is_required():
            continue
        default = field_info.get_default(call_default_factory=True)
        try:
            example_value = _coerce(adapter, example[env_name])
        except Exception:  # noqa: BLE001
            raw_drift.append({"key": env_name, "surface": "settings", "default": repr(default),
                              "example": "<unparseable for field type>"})
            continue
        if not _same(default, example_value):
            raw_drift.append({"key": env_name, "surface": "settings", "default": repr(default),
                              "example": repr(example_value)})

    for key, fallback in sorted(_compose_fallbacks(service_dir / "docker-compose.yml").items()):
        if key not in example:
            continue
        adapter = adapters.get(key)
        if adapter is not None:
            try:
                equal = _same(_coerce(adapter, fallback), _coerce(adapter, example[key]))
            except Exception:  # noqa: BLE001 - fall back to a textual compare
                equal = _same_text(fallback, example[key])
        else:
            equal = _same_text(fallback, example[key])
        if not equal:
            raw_drift.append({"key": key, "surface": "docker-compose", "default": repr(fallback),
                              "example": repr(example[key])})

    drifted_keys = {item["key"] for item in raw_drift}
    return {
        "drift": [item for item in raw_drift if item["key"] not in exempt],
        "stale_exemptions": sorted(exempt - drifted_keys),
        "reasoned": [item for item in raw_drift if item["key"] in reasoned],
        "host_specific": sorted(drifted_keys & set(host_specific)),
    }


def _main_example_drift(service: str, as_json: bool, report_only: bool) -> int:
    try:
        result = find_example_drift(service)
    except SettingsCheckError as exc:
        print(f"check_settings_defaults: {exc}", file=sys.stderr)
        return 2
    failing = bool(result["drift"] or result["stale_exemptions"])
    if as_json:
        print(json.dumps({"service": service, **result}))
    else:
        # Values are host/config literals from a committed template, never the
        # live .env, so printing them is safe -- except values on keys that look
        # secret-shaped, which are masked anyway.
        def _show(item: dict) -> str:
            secretish = any(t in item["key"] for t in ("PASS", "TOKEN", "SECRET", "KEY", "DSN"))
            ex = "<masked>" if secretish else item["example"]
            df = "<masked>" if secretish else item["default"]
            return f"    {item['key']} [{item['surface']}]: default={df} .env_example={ex}"

        if result["drift"]:
            print(f"check_settings_defaults: {service} has {len(result['drift'])} default(s) "
                  f"that differ from .env_example (a missing .env key would silently change behaviour):")
            for item in result["drift"]:
                print(_show(item))
        if result["stale_exemptions"]:
            print(f"check_settings_defaults: {service} exemption(s) no longer drifted -- remove them "
                  f"from HOST_SPECIFIC_EXAMPLE_KEYS / REASONED_DRIFT:")
            for key in result["stale_exemptions"]:
                print(f"    {key}")
        if result["reasoned"]:
            print(f"check_settings_defaults: {service} drift left open on purpose:")
            for item in result["reasoned"]:
                print(_show(item) + f"  -- {REASONED_DRIFT[service][item['key']]}")
        if not failing:
            print(f"check_settings_defaults: {service} OK -- settings defaults and compose fallbacks "
                  f"match .env_example ({len(result['host_specific'])} host-specific key(s) exempt).")
    if failing and not report_only:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("service", help="service directory name under services/, e.g. orion-actions")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of prose.")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="always exit 0, even if fields are missing defaults (report but don't gate).",
    )
    parser.add_argument(
        "--example-drift",
        action="store_true",
        help="instead: fail when a Settings default or compose ${KEY:-x} fallback differs from .env_example.",
    )
    args = parser.parse_args(argv)

    if args.example_drift:
        return _main_example_drift(args.service, args.json, args.report_only)

    try:
        missing = _find_missing_defaults(args.service)
    except SettingsCheckError as exc:
        print(f"check_settings_defaults: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({
            "service": args.service,
            "missing_defaults": missing,
        }))
    else:
        if missing:
            print(
                f"check_settings_defaults: {args.service} has {len(missing)} Settings "
                f"field(s) with no default and not on REQUIRED_NO_DEFAULT allowlist:"
            )
            for name in missing:
                print(f"    {name}")
        else:
            print(f"check_settings_defaults: {args.service} OK -- every Settings field has a default.")

    if missing and not args.report_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
