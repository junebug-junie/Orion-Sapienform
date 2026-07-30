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


class SettingsCheckError(ValueError):
    """Raised when the target service's Settings class can't be loaded/introspected."""


def _load_settings_module(service: str) -> ModuleType:
    rel_path = _SETTINGS_MODULE_PATHS.get(service)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("service", help="service directory name under services/, e.g. orion-actions")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of prose.")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="always exit 0, even if fields are missing defaults (report but don't gate).",
    )
    args = parser.parse_args(argv)

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
