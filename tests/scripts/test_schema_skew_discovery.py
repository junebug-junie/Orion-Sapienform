"""Gate tests for mechanical discovery of cross-service strict schemas
(``orion/schema_skew_discovery.py``) and the batched live read in
``scripts/check_substrate_ladder_liveness.py``."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion import schema_skew_discovery as ssd  # noqa: E402
from orion import substrate_ladder_liveness as ll  # noqa: E402

SKEW_FIX = REPO / "tests" / "fixtures" / "schema_skew"


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "check_substrate_ladder_liveness", REPO / "scripts" / "check_substrate_ladder_liveness.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def real(real_schema_discovery):
    return real_schema_discovery


# ------------------------------------------------------------- real repo gate


def test_every_unresolved_candidate_is_declared(real):
    """A forbid model some service validates but no service is found writing
    must be declared in DECLARED_WRITERS (with its writer, or None + reason).
    Otherwise it would silently fall out of the live skew check."""
    assert real.unresolved == [], "\n".join(
        f"{u.key} ({u.path}) read by {', '.join(u.readers)}: declare its writer in DECLARED_WRITERS"
        for u in real.unresolved
    )


def test_declared_writers_are_not_stale(real):
    stale = set(ssd.DECLARED_WRITERS) - real.declared_used
    assert not stale, f"DECLARED_WRITERS entries that match nothing unresolved any more: {sorted(stale)}"
    services = {p.name for p in (REPO / "services").iterdir() if p.is_dir()}
    for key, (writer, reason) in ssd.DECLARED_WRITERS.items():
        assert reason.strip(), key
        assert writer is None or writer in services, (key, writer)


def test_real_repo_coverage_is_broad_but_bounded(real):
    strict = real.strict()
    files = {c.path for c in strict}
    # The 09-20 schema and the bus envelope (every service writes and reads it)
    assert "orion/schemas/field_state.py" in files
    assert "orion/core/bus/bus_schemas.py" in files
    assert len(files) >= 50
    # The registry import is not a read path: without that exclusion every
    # schema would reach ~60 services.
    for c in strict:
        assert all(r in {p.name for p in (REPO / "services").iterdir()} for r in c.readers)
        assert c.writer not in c.readers


def test_channels_yaml_producers_count_as_writers(real):
    hits = [c for c in real.candidates if any("channels.yaml producer" in e for e in c.evidence.get(c.writer, ()))]
    assert hits


# ------------------------------------------------------- synthetic discovery


def _write(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


BASE = (
    "from pydantic import BaseModel, ConfigDict, Field\n"
    "class StrictBase(BaseModel):\n"
    "    model_config = ConfigDict(extra='forbid')\n"
)


def _repo(tmp_path: Path) -> Path:
    _write(tmp_path, "orion/__init__.py", "")
    _write(tmp_path, "orion/schemas/__init__.py", "from orion.schemas.thing import ThingV1\n")
    _write(tmp_path, "orion/schemas/base.py", BASE)
    _write(
        tmp_path,
        "orion/schemas/thing.py",
        "from pydantic import BaseModel\n"
        "from orion.schemas.base import StrictBase\n"
        "class PartV1(StrictBase):\n    a: int\n"
        "class ThingV1(StrictBase):\n    parts: list['PartV1'] = []\n    n: int\n"
        "class LooseV1(BaseModel):\n    x: int = 0\n",
    )
    _write(
        tmp_path,
        "orion/lib/store.py",
        "from orion.schemas.thing import ThingV1\n"
        "def load(row):\n    return ThingV1.model_validate(row)\n"
        "def unrelated():\n    return 1\n",
    )
    _write(tmp_path, "orion/bus/channels.yaml", "channels: []\n")
    # writer: constructs with keywords
    _write(tmp_path, "services/svc-writer/app/w.py", "from orion.schemas.thing import ThingV1\nThingV1(n=1)\n")
    # reader through a shared-library function it calls
    _write(tmp_path, "services/svc-lib-reader/app/r.py", "from orion.lib.store import load\nload({})\n")
    # imports the library module but never calls the validating function
    _write(tmp_path, "services/svc-importer/app/r.py", "from orion.lib import store\nstore.unrelated()\n")
    # reader via a re-export and a ``**`` splat
    _write(tmp_path, "services/svc-splat/app/r.py", "from orion.schemas import ThingV1\nThingV1(**{})\n")
    # test files never count
    _write(tmp_path, "services/svc-tests/tests/test_x.py", "from orion.schemas.thing import ThingV1\nThingV1.model_validate({})\n")
    return tmp_path


def test_synthetic_discovery_roles_and_inheritance(tmp_path):
    d = ssd.discover(_repo(tmp_path))
    assert d.unresolved == []
    (c,) = [c for c in d.strict() if c.path == "orion/schemas/thing.py"]
    assert c.writer == "svc-writer"
    assert set(c.readers) == {"svc-lib-reader", "svc-splat"}
    # forbid is inherited from StrictBase in another file; the nested PartV1
    # is read wherever ThingV1 is.
    assert set(c.models) == {"ThingV1", "PartV1"}
    assert d.dependency_files["orion/schemas/thing.py"] == ("orion/schemas/base.py",)


def test_unresolved_reader_is_surfaced_then_declared(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    (root / "services/svc-writer/app/w.py").unlink()
    d = ssd.discover(root)
    assert {u.key for u in d.unresolved} >= {"orion.schemas.thing:ThingV1"}
    monkeypatch.setattr(ssd, "DECLARED_WRITERS", {"orion.schemas.thing:ThingV1": ("svc-declared", "test")})
    d = ssd.discover(root)
    assert d.unresolved == []
    (c,) = [c for c in d.strict() if c.path == "orion/schemas/thing.py"]
    assert c.writer == "svc-declared" and "PartV1" in c.models
    assert d.declared_used == {"orion.schemas.thing:ThingV1"}
    monkeypatch.setattr(ssd, "DECLARED_WRITERS", {"orion/schemas/thing.py": (None, "config file")})
    d = ssd.discover(root)
    assert d.unresolved == [] and not d.strict()


def test_channel_producer_is_a_writer_but_wildcards_are_not(tmp_path):
    root = _repo(tmp_path)
    (root / "services/svc-writer/app/w.py").unlink()
    _write(root, "orion/schemas/registry.py", "from orion.schemas.thing import ThingV1\n_REGISTRY = {\"ThingV1\": ThingV1}\n")
    _write(root, "services/svc-bus/app/x.py", "import orion\n")
    _write(
        root,
        "orion/bus/channels.yaml",
        "channels:\n  - name: \"a\"\n    schema_id: \"ThingV1\"\n    producer_services: [\"svc-bus\", \"*\"]\n    consumer_services: []\n",
    )
    d = ssd.discover(root)
    assert {c.writer for c in d.strict()} == {"svc-bus"}


def test_shapes_follow_cross_file_inheritance_and_required_fields():
    src = {
        "orion/schemas/base.py": BASE + "    common: str\n",
        "orion/schemas/t.py": (
            "from orion.schemas.base import StrictBase\n"
            "from pydantic import Field\n"
            "from typing import ClassVar\n"
            "class T(StrictBase):\n"
            "    K: ClassVar[int] = 1\n"
            "    _private: int = 0\n"
            "    a: int\n    b: int = 1\n    c: int = Field(...)\n    d: int = Field(default=2)\n"
            "    e: list[int] = Field(default_factory=list)\n"
            "class Old(StrictBase):\n    class Config:\n        extra = 'allow'\n"
        ),
    }
    shapes = ssd.shapes_from_sources(src)
    t = shapes["orion.schemas.t:T"]
    assert t.extra == "forbid"
    assert t.fields == {"common", "a", "b", "c", "d", "e"}
    assert t.required == {"common", "a", "c"}
    assert shapes["orion.schemas.t:Old"].extra == "allow"
    assert ssd.compare_models("orion.schemas.t", ["Gone"], shapes, shapes) is None


# --------------------------------------------------------- CLI batched read


def test_ref_file_shas_matches_git_show():
    cli = _load_cli()
    path = "orion/schemas/field_state.py"
    try:
        got = cli.ref_file_shas(str(REPO), "HEAD", [path, "orion/does/not/exist.py"])
    except Exception as exc:  # noqa: BLE001 - shallow clone without HEAD objects
        pytest.skip(f"git unavailable: {exc}")
    assert got[path] == hashlib.sha256((REPO / path).read_bytes()).hexdigest()
    assert got["orion/does/not/exist.py"] is None


def test_sidecar_without_orion_is_distinguished_from_a_failed_read(monkeypatch):
    cli = _load_cli()
    monkeypatch.setattr(cli, "_run", lambda cmd, timeout=0: json.dumps({"__orion__": False, "orion/x.py": None}))
    assert cli.container_sources("redis", ["orion/x.py"]) == cli.NO_ORION
    monkeypatch.setattr(cli, "_run", lambda cmd, timeout=0: json.dumps({"__orion__": True, "orion/x.py": "x = 1\n"}))
    assert cli.container_sources("svc", ["orion/x.py"]) == {"orion/x.py": "x = 1\n"}

    def boom(cmd, timeout=0):
        raise cli.subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(cli, "_run", boom)
    assert cli.container_sources("slow", ["orion/x.py"]) is None

    def restarting(cmd, timeout=0):
        raise cli.subprocess.CalledProcessError(1, cmd, stderr="Error response from daemon: Container x is restarting")

    monkeypatch.setattr(cli, "_run", restarting)
    assert cli.container_sources("restarting", ["orion/x.py"]) is None

    def no_python(cmd, timeout=0):
        raise cli.subprocess.CalledProcessError(
            127, cmd, stderr=""
        )

    monkeypatch.setattr(cli, "_run", no_python)
    assert cli.container_sources("redis", ["orion/x.py"]) == cli.NO_ORION


def test_check_skew_end_to_end_with_one_exec_per_container(monkeypatch):
    """Live path with docker faked: the discovered proposal-frame pair, the
    writer on the post-c95c8360c file, one reader stale, one sidecar."""
    cli = _load_cli()
    path = "orion/schemas/proposal_frame.py"
    after = (SKEW_FIX / "proposal_frame.after_c95c8360c.py.txt").read_text()
    before = (SKEW_FIX / "proposal_frame.before_c95c8360c.py.txt").read_text()
    sc = ll.StrictSchema(
        path, path, "orion-proposal-runtime",
        models=("ProposalCandidateV1", "ProposalFrameV1"),
        reader_models=(("orion-execution-dispatch-runtime", ("ProposalCandidateV1", "ProposalFrameV1")),),
    )
    disc = ssd.Discovery([], [], 1, {})
    monkeypatch.setattr(cli.ll, "discovered_schemas", lambda root: ([sc], disc))
    t = datetime(2026, 8, 21, tzinfo=timezone.utc)
    monkeypatch.setattr(cli, "list_containers", lambda services: [
        cli.ContainerMeta("prop", "orion-proposal-runtime", t, t),
        cli.ContainerMeta("disp", "orion-execution-dispatch-runtime", t - timedelta(days=9), t),
        cli.ContainerMeta("disp-redis", "orion-execution-dispatch-runtime", t - timedelta(days=400), t),
        cli.ContainerMeta("disp-2", "orion-execution-dispatch-runtime", t - timedelta(days=9), t),
    ])
    calls = []

    def fake_sources(name, paths):
        calls.append(name)
        return {"prop": {path: after}, "disp": {path: before}, "disp-redis": cli.NO_ORION, "disp-2": None}[name]

    monkeypatch.setattr(cli, "container_sources", fake_sources)
    monkeypatch.setattr(cli, "ref_file_shas", lambda repo, ref, paths: {path: hashlib.sha256(after.encode()).hexdigest()})
    args = cli.argparse.Namespace(repo=str(REPO), ref="HEAD", include_loose=True, docker_workers=2, verbose=False)
    results, notes = cli.check_skew(args)
    assert sorted(calls) == ["disp", "disp-2", "disp-redis", "prop"]  # one read per container
    by = {r.container: r for r in results}
    assert by["disp"].red and "expected_signal" in by["disp"].detail
    assert by["prop"].status == "ok"
    assert "disp-redis" not in by  # sidecar: skipped silently
    assert "disp-2" not in by  # failed read: not guessed at, but surfaced
    assert notes == ["skew: could not read schema files in container disp-2 (docker exec failed)"]
    # the production key form: schema file path, not a class name
    assert ll.LadderReport(skew=results).red_keys() == [f"skew:{path}:disp"]


def test_shapes_resolve_a_base_imported_through_a_re_export():
    src = {
        "orion/schemas/base.py": BASE + "    common: str\n",
        "orion/schemas/t.py": "from orion.schemas import StrictBase\nclass T(StrictBase):\n    a: int\n",
    }
    t = ssd.shapes_from_sources(src)["orion.schemas.t:T"]
    assert t.extra == "forbid" and t.fields == {"common", "a"}


def test_before_validator_reader_is_reported_not_red():
    """attention_frame.py's _drop_removed_legacy_fields pattern: the reader
    removed a field and strips it from old rows in a before-validator. That is
    the correct consumer-first order and must not page."""
    path = "orion/schemas/a.py"
    w = ssd.shapes_from_sources({path: BASE.replace("StrictBase", "A") + "    x: int\n    legacy: int = 0\n"})
    r = ssd.shapes_from_sources({path: BASE.replace("StrictBase", "A") + (
        "    x: int\n"
        "    @model_validator(mode='before')\n"
        "    @classmethod\n"
        "    def _drop(cls, v):\n        return v\n"
    )})
    (d,) = ssd.compare_models("orion.schemas.a", ["A"], w, r)
    assert not d.breaks and d.maybe_stripped == ("legacy",)
    sc = ll.StrictSchema(path, path, "w", models=("A",))
    got = {x.container: x for x in ll.evaluate_skew(
        sc, schema_commit_time=None, schema_sha256_on_ref=None, consumer_services=["r"],
        containers=[ll.RunningContainer("wc", "w", None, shapes=w), ll.RunningContainer("rc", "r", None, shapes=r)],
    )}
    assert got["rc"].status == "forbidden_unless_stripped" and not got["rc"].red


def test_unresolvable_or_ambiguous_base_forces_fallback_not_loose():
    """A forbid reader whose base cannot be resolved must not read as loose
    (that would turn the 09-20 shape into a warning)."""
    src = {"orion/schemas/t.py": "from orion.other import StrictBase\nclass T(StrictBase):\n    a: int\n"}
    assert "orion.schemas.t:T" not in ssd.shapes_from_sources(src)
    amb = {
        "orion/schemas/b1.py": BASE,
        "orion/schemas/b2.py": BASE,
        "orion/schemas/t.py": "from orion.schemas import StrictBase\nclass T(StrictBase):\n    a: int\n",
    }
    assert "orion.schemas.t:T" not in ssd.shapes_from_sources(amb)
    dotted = {
        "orion/schemas/base.py": BASE,
        "orion/schemas/t.py": "from orion.schemas import base\nclass T(base.StrictBase):\n    a: int\n",
    }
    assert ssd.shapes_from_sources(dotted)["orion.schemas.t:T"].extra == "forbid"


def test_extra_follows_mro_first_base_wins():
    src = {"orion/schemas/t.py": (
        "from pydantic import BaseModel, ConfigDict\n"
        "class F(BaseModel):\n    model_config = ConfigDict(extra='forbid')\n"
        "class I(BaseModel):\n    model_config = ConfigDict(extra='ignore')\n"
        "class M(F, I):\n    a: int\n"
    )}
    assert ssd.shapes_from_sources(src)["orion.schemas.t:M"].extra == "forbid"


def test_field_edge_cases():
    src = {"orion/schemas/t.py": (
        "from typing import Annotated\nfrom pydantic import BaseModel, Field\n"
        "class T(BaseModel):\n"
        "    a: Annotated[int, Field(default=1)]\n"
        "    b: int = Field(default=...)\n"
        "    c: int = Field(default=0, exclude=True)\n"
        "    class Config:\n        extra = 'forbid'\n"
        "class Config(BaseModel):\n    top: int\n"
    )}
    sh = ssd.shapes_from_sources(src)
    t = sh["orion.schemas.t:T"]
    assert t.required == {"b"} and t.excluded == {"c"} and t.extra == "forbid"
    # the nested ``class Config`` did not overwrite the top-level model
    assert sh["orion.schemas.t:Config"].fields == {"top"}


def test_init_schema_module_name():
    sc = ll.StrictSchema("orion/schemas/pkg/__init__.py", "x", "w")
    assert sc.module == "orion.schemas.pkg"


def test_container_parser_agrees_with_repo_index_for_every_candidate(real):
    """Parity gate: the host-side parser over exactly the files the live check
    fetches for each service must see the same fields and extra as discovery
    for every candidate model. A divergence means the live comparison would
    judge a different model than the one discovered."""
    index = real.index
    schemas = ll.schemas_from_discovery(real)
    files_by_svc: dict[str, set[str]] = {}
    for sc in schemas:
        for svc in {sc.producer_service, *(r for r, _ in sc.reader_models)}:
            files_by_svc.setdefault(svc, set()).update({sc.path, *sc.deps})
    parsed = {
        svc: ssd.shapes_from_sources({p: (REPO / p).read_text(encoding="utf-8") for p in files})
        for svc, files in files_by_svc.items()
    }
    by_module = {f.path: f.module for f in index.orion.values()}
    bad = []
    for sc in schemas:
        mod = by_module[sc.path]
        for svc in {sc.producer_service, *(r for r, _ in sc.reader_models)}:
            for name in sc.models:
                ref = (mod, name)
                want_fields = set().union(*(set(index.classes[b].fields) for b in index.base_closure(ref)))
                got = parsed[svc].get(f"{mod}:{name}")
                if got is None or got.fields != want_fields or got.extra != index.effective_extra(ref):
                    bad.append((sc.path, name, svc))
    assert not bad, bad[:10]
