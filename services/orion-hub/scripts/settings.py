from app.settings import Settings, get_settings, settings  # noqa: F401

print(f"--- LOADING SETTINGS FROM: {__file__} ---", flush=True)

# Mirror feature-flag access in scripts namespace for clarity in route handlers.
HUB_AUTO_DEFAULT_ENABLED = bool(getattr(settings, "HUB_AUTO_DEFAULT_ENABLED", False))
