from pathlib import Path
import sys
import types
import logging
import importlib.util

ROOT = Path(__file__).resolve().parents[3]
APP_PATH = ROOT / "services" / "orion-recall"
APP_DIR = APP_PATH / "app"
PACKAGE_NAME = "orion_recall"
APP_PACKAGE_NAME = f"{PACKAGE_NAME}.app"
LEGACY_APP_PACKAGE = "app"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

if PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(APP_PATH)]
    sys.modules[PACKAGE_NAME] = pkg
if APP_PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(APP_PACKAGE_NAME)
    pkg.__path__ = [str(APP_DIR)]
    sys.modules[APP_PACKAGE_NAME] = pkg
if LEGACY_APP_PACKAGE not in sys.modules:
    pkg = types.ModuleType(LEGACY_APP_PACKAGE)
    pkg.__path__ = [str(APP_DIR)]
    sys.modules[LEGACY_APP_PACKAGE] = pkg

sys.modules.setdefault("loguru", types.SimpleNamespace(logger=logging.getLogger("test")))

spec_settings = importlib.util.spec_from_file_location(f"{LEGACY_APP_PACKAGE}.settings", APP_DIR / "settings.py")
settings_mod = importlib.util.module_from_spec(spec_settings)
assert spec_settings and spec_settings.loader
sys.modules[spec_settings.name] = settings_mod
spec_settings.loader.exec_module(settings_mod)

spec = importlib.util.spec_from_file_location(f"{APP_PACKAGE_NAME}.profiles", APP_DIR / "profiles.py")
profiles = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = profiles
spec.loader.exec_module(profiles)

get_profile = profiles.get_profile
load_profiles = profiles.load_profiles


def test_profiles_load_default():
    load_profiles.cache_clear()
    profiles = load_profiles()
    assert "reflect.v1" in profiles
    assert "brain.recall.v1" in profiles
    prof = get_profile("reflect.v1")
    assert prof["profile"] == "reflect.v1"
    assert prof["max_total_items"] > 0


def test_self_factual_profile_loads():
    prof = get_profile("self.factual.v1")
    assert prof["profile"] == "self.factual.v1"
    assert prof["rdf_top_k"] > 0
    assert prof["enable_query_expansion"] is False
    assert prof["filters"]["allowed_sources"] == ["rdf", "sql_timeline"]
    assert prof["filters"]["exclude_tags_prefixes"] == ["trust:induced", "trust:reflective"]
    assert prof["policy"]["trust_tiers"] == ["authoritative"]
def test_dream_v1_profile_loads():
    profiles = load_profiles()
    assert "dream.v1" in profiles
    prof = get_profile("dream.v1")
    assert prof["profile"] == "dream.v1"
    assert prof["max_total_items"] >= 24
    assert int(prof.get("render_budget_tokens") or 0) >= 256


def test_recall_v1_hub_alias_maps_to_brain_recall():
    """Hub legacy label ``recall.v1`` must resolve to structured brain recall, not chat.general.v1."""
    load_profiles.cache_clear()
    prof = get_profile("recall.v1")
    assert prof.get("profile") == "brain.recall.v1"
    assert get_profile("RECALL.V1").get("profile") == "brain.recall.v1"


def test_brain_recall_v1_profile_is_sql_rdf_non_vector():
    load_profiles.cache_clear()
    prof = get_profile("brain.recall.v1")
    assert prof["profile"] == "brain.recall.v1"
    assert prof.get("enable_vector") is False
    assert int(prof.get("vector_top_k", -1)) == 0
    assert prof.get("enable_rdf") is True
    assert int(prof.get("rdf_top_k", 0)) > 0
    assert float(prof.get("relevance", {}).get("backend_weights", {}).get("vector", 1.0)) == 0.0


def test_chat_general_profile_is_narrower_than_reflect_profile():
    chat = get_profile("chat.general.v1")
    reflect = get_profile("reflect.v1")

    assert chat["enable_sql_timeline"] is False
    assert reflect["enable_sql_timeline"] is True
    assert chat["max_total_items"] < reflect["max_total_items"]
    assert int(chat.get("vector_top_k", 0)) == 0
    assert int(reflect.get("vector_top_k", 0)) == 0
    assert chat["rdf_top_k"] < reflect["rdf_top_k"]
    assert chat["sql_top_k"] < reflect["sql_top_k"]


def test_collapse_mirror_profile_is_lighter_than_reflect():
    collapse = get_profile("collapse_mirror.v1")
    reflect = get_profile("reflect.v1")
    assert collapse["profile"] == "collapse_mirror.v1"
    assert collapse["enable_query_expansion"] is False
    assert int(collapse.get("sql_top_k") or 0) < int(reflect.get("sql_top_k") or 0)
    assert int(collapse.get("sql_since_minutes") or 0) < int(reflect.get("sql_since_minutes") or 0)


def test_daily_metacog_profile_is_true_daily_window_not_weekly():
    load_profiles.cache_clear()
    prof = get_profile("journal.daily.metacog.grounded.v1")
    assert prof["profile"] == "journal.daily.metacog.grounded.v1"
    assert int(prof.get("sql_since_minutes") or 0) == 1440
    assert int(prof.get("sql_since_minutes") or 0) != 10080
    assert 8 <= int(prof.get("max_total_items") or 0) <= 12
    assert 40 <= int(prof.get("sql_top_k") or 0) <= 60
    assert int(prof.get("render_budget_tokens") or 0) == 256
    assert int(prof.get("max_render_snippet_chars") or 0) == 280
    assert prof.get("strict_prompt_budget") is True
    assert int(prof.get("render_char_budget") or 0) == 1280
    assert prof.get("prompt_safe_ctx") is True


def test_journal_daily_grounded_profile_keeps_legacy_render_semantics() -> None:
    load_profiles.cache_clear()
    prof = get_profile("journal.daily.grounded.v1")
    assert prof.get("strict_prompt_budget") is not True
    assert not int(prof.get("render_char_budget") or 0)
    assert prof.get("prompt_safe_ctx") is not True
